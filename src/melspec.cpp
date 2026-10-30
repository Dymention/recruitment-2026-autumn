#include "melspec.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "mkl_dfti.h"

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

// ---------------------------------------------------------------------------
// Sparse Mel filterbank.
// ---------------------------------------------------------------------------
struct MelBank {
    int n_mels = 0, n_bins = 0;
    std::vector<double> w;   // dense, row-major [n_mels][n_bins]
    std::vector<int> lo, hi; // inclusive support per mel
};

void build_mel_bank(int sr, int n_fft, int n_mels, double f_min, double f_max,
                    MelBank& mb) {
    const int n_bins = n_fft / 2 + 1;
    mb.n_mels = n_mels;
    mb.n_bins = n_bins;
    mb.w.assign(static_cast<size_t>(n_mels) * n_bins, 0.0);
    mb.lo.assign(n_mels, n_bins);
    mb.hi.assign(n_mels, -1);

    std::vector<double> freq(n_bins);
    for (int k = 0; k < n_bins; ++k)
        freq[k] = static_cast<double>(k) * sr / static_cast<double>(n_fft);

    const double mel_min = 2595.0 * std::log10(1.0 + f_min / 700.0);
    const double mel_max = 2595.0 * std::log10(1.0 + f_max / 700.0);
    std::vector<double> hz(n_mels + 2);
    for (int i = 0; i <= n_mels + 1; ++i) {
        double mel = mel_min + i * (mel_max - mel_min) / (n_mels + 1);
        hz[i] = 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
    }

    for (int m = 0; m < n_mels; ++m) {
        const double fl = hz[m], fc = hz[m + 1], fr = hz[m + 2];
        const double enorm = 2.0 / (fr - fl);
        double* row = mb.w.data() + static_cast<size_t>(m) * n_bins;
        for (int k = 0; k < n_bins; ++k) {
            double rising = (freq[k] - fl) / (fc - fl);
            double falling = (fr - freq[k]) / (fr - fc);
            double val = std::max(0.0, std::min(rising, falling)) * enorm;
            row[k] = val;
            if (val != 0.0) {
                if (mb.lo[m] == n_bins) mb.lo[m] = k;
                mb.hi[m] = k;
            }
        }
    }
}

}  // namespace

double hz_to_mel(double f) {
    return 2595.0 * std::log10(1.0 + f / 700.0);
}

double mel_to_hz(double m) {
    return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0);
}

void compute_rfft(const std::vector<double>& frame,
                  std::vector<std::complex<double>>& out) {
    const int n = static_cast<int>(frame.size());
    DFTI_DESCRIPTOR_HANDLE h = NULL;
    DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 1, static_cast<MKL_LONG>(n));
    DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(h, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiCommitDescriptor(h);

    std::vector<double> in(frame.begin(), frame.end());
    std::vector<double> spec(static_cast<size_t>(n) + 2);
    DftiComputeForward(h, in.data(), spec.data());
    DftiFreeDescriptor(&h);

    out.resize(n / 2 + 1);
    for (int k = 0; k <= n / 2; ++k)
        out[k] = std::complex<double>(spec[2 * k], spec[2 * k + 1]);
}

std::vector<std::vector<double>> build_mel_filterbank(int sr, int n_fft,
                                                      int n_mels, double f_min,
                                                      double f_max) {
    MelBank mb;
    build_mel_bank(sr, n_fft, n_mels, f_min, f_max, mb);
    std::vector<std::vector<double>> result(
        n_mels, std::vector<double>(static_cast<size_t>(mb.n_bins), 0.0));
    for (int m = 0; m < n_mels; ++m)
        for (int k = 0; k < mb.n_bins; ++k)
            result[m][k] = mb.w[static_cast<size_t>(m) * mb.n_bins + k];
    return result;
}

void compute_melspectrogram(const std::vector<double>& y, int sr, int n_fft,
                            int hop_length, int n_mels, double f_min,
                            double f_max, std::vector<double>& output) {
    const int n_samples = static_cast<int>(y.size());
    const int n_frames = 1 + (n_samples - n_fft) / hop_length;
    const int n_bins = n_fft / 2 + 1;
    if (n_frames <= 0)
        throw std::runtime_error("Audio too short for given n_fft");

    // Periodic Hann window.
    std::vector<double> window(n_fft);
    for (int i = 0; i < n_fft; ++i)
        window[i] = 0.5 * (1.0 - std::cos(2.0 * kPi * i / n_fft));

    MelBank mb;
    build_mel_bank(sr, n_fft, n_mels, f_min, f_max, mb);

    // One-time DFTI plan for the real FFT (shared across threads).
    DFTI_DESCRIPTOR_HANDLE h = NULL;
    DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 1,
                         static_cast<MKL_LONG>(n_fft));
    DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(h, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(h, DFTI_THREAD_LIMIT, 1);
    DftiCommitDescriptor(h);

    output.assign(static_cast<size_t>(n_mels) * n_frames, 0.0);

    const double* yp = y.data();
    const double* wp = window.data();
    const double* melw = mb.w.data();
    const int* mlo = mb.lo.data();
    const int* mhi = mb.hi.data();
    double* outp = output.data();

#pragma omp parallel
    {
        std::vector<double> in(n_fft);
        std::vector<double> spec(static_cast<size_t>(n_fft) + 2);
        std::vector<double> power(n_bins);
        double* pw = power.data();

#pragma omp for schedule(static)
        for (int f = 0; f < n_frames; ++f) {
            const double* frame = yp + static_cast<size_t>(f) * hop_length;
            for (int i = 0; i < n_fft; ++i) in[i] = frame[i] * wp[i];

            DftiComputeForward(h, in.data(), spec.data());

            for (int k = 0; k < n_bins; ++k) {
                double re = spec[2 * k], im = spec[2 * k + 1];
                pw[k] = re * re + im * im;
            }

            for (int m = 0; m < n_mels; ++m) {
                const double* mw = melw + static_cast<size_t>(m) * n_bins;
                double acc = 0.0;
                int klo = mlo[m], khi = mhi[m];
                for (int k = klo; k <= khi; ++k) acc += mw[k] * pw[k];
                outp[static_cast<size_t>(m) * n_frames + f] = acc;
            }
        }
    }

    DftiFreeDescriptor(&h);
}

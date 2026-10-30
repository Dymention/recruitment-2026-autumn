#include "melspec.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

// ---------------------------------------------------------------------------
// Packed real FFT: a real frame of length n_fft (= 2*n2) is reduced to a
// complex FFT of size n2 by putting even samples in the real part and odd
// samples in the imaginary part, then "unpacking" the result. This halves the
// FFT work compared to a full complex FFT of the padded real input.
// ---------------------------------------------------------------------------
struct FftTables {
    int n2 = 0;                // n_fft / 2
    std::vector<int> rev;      // bit-reversal permutation for size n2
    std::vector<double> w_re;  // exp(-2*pi*i*k/n2), k in [0, n2)
    std::vector<double> w_im;
    std::vector<double> u_cos;  // cos(2*pi*k/n_fft), k in [0, n2]
    std::vector<double> u_sin;  // sin(2*pi*k/n_fft), k in [0, n2]
};

void build_fft_tables(int n2, FftTables& t) {
    t.n2 = n2;
    int logn = 0;
    while ((1 << logn) < n2) ++logn;

    t.rev.resize(n2);
    for (int i = 0; i < n2; ++i) {
        int r = 0;
        for (int b = 0; b < logn; ++b)
            if (i & (1 << b)) r |= (1 << (logn - 1 - b));
        t.rev[i] = r;
    }

    t.w_re.resize(n2);
    t.w_im.resize(n2);
    for (int k = 0; k < n2; ++k) {
        double ang = -2.0 * kPi * k / n2;
        t.w_re[k] = std::cos(ang);
        t.w_im[k] = std::sin(ang);
    }

    int nfft = 2 * n2;
    t.u_cos.resize(n2 + 1);
    t.u_sin.resize(n2 + 1);
    for (int k = 0; k <= n2; ++k) {
        double ang = 2.0 * kPi * k / nfft;
        t.u_cos[k] = std::cos(ang);
        t.u_sin[k] = std::sin(ang);
    }
}

// Iterative radix-2 DIT complex FFT over an interleaved [re, im] buffer.
inline void fft_radix2(double* buf, int n2, const FftTables& t) {
    const int* rev = t.rev.data();
    for (int i = 1; i < n2; ++i) {
        int j = rev[i];
        if (i < j) {
            double tr = buf[2 * i], ti = buf[2 * i + 1];
            buf[2 * i] = buf[2 * j];
            buf[2 * i + 1] = buf[2 * j + 1];
            buf[2 * j] = tr;
            buf[2 * j + 1] = ti;
        }
    }
    const double* wre = t.w_re.data();
    const double* wim = t.w_im.data();
    for (int len = 2; len <= n2; len <<= 1) {
        int half = len >> 1;
        int step = n2 / len;
        for (int i = 0; i < n2; i += len) {
            for (int j = 0; j < half; ++j) {
                int k = j * step;
                double wr = wre[k], wi = wim[k];
                int a = i + j, b = i + j + half;
                double ur = buf[2 * a], ui = buf[2 * a + 1];
                double br = buf[2 * b], bi = buf[2 * b + 1];
                double vr = br * wr - bi * wi;
                double vi = br * wi + bi * wr;
                buf[2 * a] = ur + vr;
                buf[2 * a + 1] = ui + vi;
                buf[2 * b] = ur - vr;
                buf[2 * b + 1] = ui - vi;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mel filterbank: stored densely plus the contiguous non-zero support [lo, hi]
// of each triangular filter, so the per-frame application can skip zero bins.
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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

double hz_to_mel(double f) {
    return 2595.0 * std::log10(1.0 + f / 700.0);
}

double mel_to_hz(double m) {
    return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0);
}

void compute_rfft(const std::vector<double>& frame,
                  std::vector<std::complex<double>>& out) {
    const int n = static_cast<int>(frame.size());
    FftTables t;
    build_fft_tables(n, t);
    std::vector<double> buf(static_cast<size_t>(2) * n, 0.0);
    for (int i = 0; i < n; ++i) buf[2 * i] = frame[i];
    fft_radix2(buf.data(), n, t);
    out.resize(n / 2 + 1);
    for (int k = 0; k <= n / 2; ++k)
        out[k] = std::complex<double>(buf[2 * k], buf[2 * k + 1]);
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

    const int n2 = n_fft / 2;

    // Periodic Hann window.
    std::vector<double> window(n_fft);
    for (int i = 0; i < n_fft; ++i)
        window[i] = 0.5 * (1.0 - std::cos(2.0 * kPi * i / n_fft));

    // One-time precomputation: FFT twiddles + sparse mel filterbank.
    FftTables ft;
    build_fft_tables(n2, ft);
    MelBank mb;
    build_mel_bank(sr, n_fft, n_mels, f_min, f_max, mb);

    output.assign(static_cast<size_t>(n_mels) * n_frames, 0.0);

    const double* yp = y.data();
    const double* wp = window.data();
    const double* melw = mb.w.data();
    const int* mlo = mb.lo.data();
    const int* mhi = mb.hi.data();
    const double* ucos = ft.u_cos.data();
    const double* usin = ft.u_sin.data();
    double* outp = output.data();

    // Parallel over frames: each frame is independent. Thread count follows
    // OMP_NUM_THREADS / number of cores (tune via OMP_NUM_THREADS on cluster).
#pragma omp parallel
    {
        std::vector<double> buf(static_cast<size_t>(2) * n2);
        std::vector<double> power(n_bins);
        double* b = buf.data();
        double* pw = power.data();

#pragma omp for schedule(static)
        for (int f = 0; f < n_frames; ++f) {
            const double* frame = yp + static_cast<size_t>(f) * hop_length;

            // Pack the windowed frame: even samples -> real, odd -> imag.
            for (int i = 0; i < n2; ++i) {
                b[2 * i] = frame[2 * i] * wp[2 * i];
                b[2 * i + 1] = frame[2 * i + 1] * wp[2 * i + 1];
            }
            fft_radix2(b, n2, ft);

            // Unpack to real FFT bins and form the power spectrum.
            double z0r = b[0], z0i = b[1];
            double x0 = z0r + z0i;    // X[0]  (DC)
            pw[0] = x0 * x0;
            double xnyq = z0r - z0i;  // X[n2] (Nyquist)
            pw[n2] = xnyq * xnyq;
            for (int k = 1; k < n2; ++k) {
                double zkr = b[2 * k], zki = b[2 * k + 1];
                double zmr = b[2 * (n2 - k)], zmi = b[2 * (n2 - k) + 1];
                double are = 0.5 * (zkr + zmr);
                double aim = 0.5 * (zki - zmi);
                double a = zkr - zmr;
                double bb = zki + zmi;
                double bre = 0.5 * bb;
                double bim = -0.5 * a;
                double wr = ucos[k], wi = -usin[k];
                double xr = are + wr * bre - wi * bim;
                double xi = aim + wr * bim + wi * bre;
                pw[k] = xr * xr + xi * xi;
            }

            // Apply the sparse mel filterbank.
            for (int m = 0; m < n_mels; ++m) {
                const double* mw = melw + static_cast<size_t>(m) * n_bins;
                double acc = 0.0;
                int klo = mlo[m], khi = mhi[m];
                for (int k = klo; k <= khi; ++k) acc += mw[k] * pw[k];
                outp[static_cast<size_t>(m) * n_frames + f] = acc;
            }
        }
    }
}

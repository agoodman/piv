// Window-pair cross-correlation, two ways:
//
// * correlate_fused: one threadgroup per window does everything (load, FFT, cross spectrum,
//   inverse FFT, peak) in threadgroup memory. The packed n × n complex plane must fit in 32 KB,
//   so n ≤ 64, and the FFT runs in place: radix-2, power-of-two sizes only. The forward
//   transform is decimation-in-time on bit-reversed input (written that way by the load), the
//   inverse is decimation-in-frequency, which leaves the correlation bit-reversed; the
//   compaction step undoes that.
//
// * The batched kernels keep the planes in device memory: load, forward 2-D FFT, cross spectrum
//   (two windows packed per plane), inverse 2-D FFT, peak. The FFTs are either fft_lines below
//   (all rows of all windows, then all columns, with a mixed-radix (2, 3, 4) Stockham FFT per
//   line in threadgroup memory; any n = 2^a·3^b up to 128) or MPSGraph's.

// MARK: - Fused path

struct TGBitrevStore {
    threadgroup float2* plane;
    uint n, shift;
    void operator()(uint r, uint c, float2 v) const {
        plane[(reverse_bits(r) >> shift) * n + (reverse_bits(c) >> shift)] = v;
    }
};

struct TGPlane {
    threadgroup float* corr;
    float operator()(uint i) const { return corr[i]; }
};

/// In-place radix-2 butterflies over all rows (`cols` = false) or all columns of the plane.
/// Forward: decimation in time, sign −1. Inverse: decimation in frequency, sign +1.
inline void fft_plane_pass(threadgroup float2* x, device const float2* tw, uint n, uint log2n,
                           bool cols, bool inverse, uint tid, uint T) {
    uint half_n = n / 2;
    for (uint st = 0; st < log2n; st++) {
        uint s = inverse ? log2n - 1 - st : st;
        uint m = 1u << s;
        uint tstride = n >> (s + 1);
        for (uint b = tid; b < n * half_n; b += T) {
            uint line = b / half_n, j = b % half_n;
            uint pos = j & (m - 1);
            uint k0 = ((j >> s) << (s + 1)) + pos;
            uint i0 = cols ? k0 * n + line : line * n + k0;
            uint i1 = cols ? i0 + m * n : i0 + m;
            float2 w = tw[pos * tstride];
            float2 u = x[i0], v = x[i1];
            if (!inverse) {
                float2 t = cmul(w, v);
                x[i0] = u + t;
                x[i1] = u - t;
            } else {
                x[i0] = u + v;
                x[i1] = cmul(u - v, conj(w));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void correlate_fused(device const float* imgA [[buffer(0)]],
                            device const float* imgB [[buffer(1)]],
                            device const float* statsA [[buffer(2)]],
                            device const float* statsB [[buffer(3)]],
                            device const float4* origins [[buffer(4)]],
                            constant CorrParams& p [[buffer(5)]],
                            device const float2* tw [[buffer(6)]],
                            device WindowPeak* out [[buffer(7)]],
                            threadgroup float2* plane [[threadgroup(0)]],
                            uint w [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint T [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint nsg [[simdgroups_per_threadgroup]]) {
    uint n = p.n, nn = n * n;
    // The plane is the only threadgroup memory; its tail doubles as reduction scratch while
    // it isn't holding data.
    threadgroup float* scratch1 = (threadgroup float*)plane + nn;
    threadgroup float2* scratch2 = plane + nn / 2;

    float2 energy;
    TGBitrevStore store{plane, n, 32 - p.log2n};
    uint status = load_window_pair(imgA, imgB, statsA, statsB, p, origins[w], store, energy,
                                   tid, T, scratch1, scratch2, lane, sg, nsg);
    // load_window_pair writes the plane after its last reduction; the reduction scratch above
    // is inside the plane, so wait for every thread's stores before anything else.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (status != STATUS_OK) {
        if (tid == 0) out[w] = WindowPeak{0, 0, 0, 0, status};
        return;
    }

    fft_plane_pass(plane, tw, n, p.log2n, false, false, tid, T);
    fft_plane_pass(plane, tw, n, p.log2n, true, false, tid, T);
    for (uint e = tid; e < nn; e += T) cross_spectrum_pair(plane, n, e);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fft_plane_pass(plane, tw, n, p.log2n, false, true, tid, T);
    fft_plane_pass(plane, tw, n, p.log2n, true, true, tid, T);

    // Real part, scaled to a correlation coefficient, compacted into the first half of the
    // plane in natural order. Read everything this thread owns before anyone writes.
    float scale = 1.0f / (float(nn) * sqrt(energy.x * energy.y));
    uint shift = 32 - p.log2n;
    // Each thread owns nn / T ≤ 8 cells (the host dispatches nn/4 threads, or nn/8 when the
    // pipeline can't run that many).
    float vals[8];
    uint dst[8];
    uint k = 0;
    for (uint e = tid; e < nn && k < 8; e += T, k++) {
        vals[k] = plane[e].x * scale;
        dst[k] = (reverse_bits(e / n) >> shift) * n + (reverse_bits(e % n) >> shift);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* corr = (threadgroup float*)plane;
    for (uint j = 0; j < k; j++) corr[dst[j]] = vals[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* sv = corr + nn;
    threadgroup uint* si = (threadgroup uint*)(corr + nn + 64);
    WindowPeak r = find_peak(TGPlane{corr}, n, tid, T, lane, sg, nsg, sv, si);
    if (tid == 0) out[w] = r;
}

// MARK: - Batched path

struct DeviceStore {
    device float2* plane;
    uint n;
    void operator()(uint r, uint c, float2 v) const { plane[r * n + c] = v; }
};

/// Loads each window pair into its device plane (natural order) and records the status and the
/// normalization scale. One threadgroup per window.
kernel void load_windows(device const float* imgA [[buffer(0)]],
                         device const float* imgB [[buffer(1)]],
                         device const float* statsA [[buffer(2)]],
                         device const float* statsB [[buffer(3)]],
                         device const float4* origins [[buffer(4)]],
                         constant CorrParams& p [[buffer(5)]],
                         device float2* planes [[buffer(6)]],
                         device WindowPeak* out [[buffer(7)]],
                         uint w [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint T [[threads_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch1[32];
    threadgroup float2 scratch2[32];
    uint nn = p.n * p.n;
    float2 energy;
    uint status = load_window_pair(imgA, imgB, statsA, statsB, p, origins[w],
                                   DeviceStore{planes + w * nn, p.n}, energy,
                                   tid, T, scratch1, scratch2, lane, sg, nsg);
    if (tid == 0) {
        // `height` carries the scale until the peak kernel replaces it.
        float scale = status == STATUS_OK ? 1.0f / (float(nn) * sqrt(energy.x * energy.y)) : 0;
        out[w] = WindowPeak{0, 0, 0, scale, status};
    }
}

/// Parameters for `fft_lines`. Mirrors `FFTLineParams` in Swift.
struct FFTLineParams {
    uint n;
    uint lines;            // total lines in the dispatch
    uint lines_per_window; // n
    uint window_stride;    // n * n
    uint line_stride;      // n for rows, 1 for columns
    uint elem_stride;      // 1 for rows, n for columns
    uint lines_per_tg;
    uint threads_per_line;
    uint inverse;
    uint stages;
    uint radix[8];
    uint check_status;     // skip planes whose window status isn't OK (not for packed planes)
};

/// Unnormalized DFT of every line (row or column) of every plane, in place in device memory.
/// Stockham autosort: each stage reads one threadgroup buffer and writes the other, so the
/// output is in natural order. `tw[k]` = exp(−2πik/n).
kernel void fft_lines(device float2* data [[buffer(0)]],
                      constant FFTLineParams& p [[buffer(1)]],
                      device const float2* tw [[buffer(2)]],
                      device const WindowPeak* status [[buffer(3)]],
                      threadgroup float2* buf [[threadgroup(0)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    uint n = p.n, tpl = p.threads_per_line;
    uint local = tid / tpl, lt = tid % tpl;
    uint line = tg * p.lines_per_tg + local;
    bool active = local < p.lines_per_tg && line < p.lines;
    uint window = line / p.lines_per_window;
    if (active && p.check_status && status[window].status != STATUS_OK) active = false;
    uint base = window * p.window_stride + (line % p.lines_per_window) * p.line_stride;
    threadgroup float2* x = buf + local * 2 * n;
    threadgroup float2* y = x + n;
    float sign = p.inverse ? 1.0f : -1.0f;

    if (active)
        for (uint i = lt; i < n; i += tpl) x[i] = data[base + i * p.elem_stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint ns = 1;
    for (uint s = 0; s < p.stages; s++) {
        uint R = p.radix[s];
        uint nr = n / R, tstride = n / (ns * R);
        if (active) {
            for (uint j = lt; j < nr; j += tpl) {
                uint jm = j % ns;
                float2 v[4];
                for (uint r = 0; r < R; r++) {
                    float2 t = tw[(jm * r * tstride) % n];
                    v[r] = cmul(x[j + r * nr], p.inverse ? conj(t) : t);
                }
                float2 o[4];
                if (R == 2) {
                    o[0] = v[0] + v[1];
                    o[1] = v[0] - v[1];
                } else if (R == 3) {
                    float2 t1 = v[1] + v[2], t2 = v[1] - v[2];
                    float2 m = v[0] - 0.5f * t1;
                    float2 q = mul_i(t2, sign) * 0.8660254037844386f;
                    o[0] = v[0] + t1;
                    o[1] = m + q;
                    o[2] = m - q;
                } else {
                    float2 t0 = v[0] + v[2], t1 = v[0] - v[2];
                    float2 t2 = v[1] + v[3], t3 = mul_i(v[1] - v[3], sign);
                    o[0] = t0 + t2;
                    o[1] = t1 + t3;
                    o[2] = t0 - t2;
                    o[3] = t1 - t3;
                }
                uint d = (j / ns) * ns * R + jm;
                for (uint r = 0; r < R; r++) y[d + r * ns] = o[r];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2* t = x;
        x = y;
        y = t;
        ns *= R;
    }

    if (active)
        for (uint i = lt; i < n; i += tpl) data[base + i * p.elem_stride] = x[i];
}

/// conj(A)·B for every window, two windows per output plane: P = C₀ + i·C₁ for windows 2j and
/// 2j + 1. Each Cᵢ is Hermitian (it is the spectrum of a real correlation), so the inverse
/// transform of P is c₀ + i·c₁ exactly, and one inverse FFT serves two windows. Windows that
/// weren't correlated (or a missing partner) contribute zero. One thread per element of each
/// output plane; the thread with the lower index of each (k, −k) pair writes both.
kernel void cross_spectrum_packed(device const float2* planes [[buffer(0)]],
                                  constant CorrParams& p [[buffer(1)]],
                                  device const WindowPeak* status [[buffer(2)]],
                                  device float2* packed [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]]) {
    uint n = p.n, nn = n * n;
    uint j = gid / nn, e = gid % nn;
    if (2 * j >= p.count) return;
    uint ky = e / n, kx = e % n;
    uint m = ((n - ky) % n) * n + (n - kx) % n;
    if (e > m) return;
    float2 c[2];
    for (uint h = 0; h < 2; h++) {
        uint w = 2 * j + h;
        c[h] = 0;
        if (w < p.count && status[w].status == STATUS_OK) {
            device const float2* z = planes + w * nn;
            float2 Z = z[e], Zm = conj(z[m]);
            float2 A = 0.5f * (Z + Zm);
            float2 B = mul_i(0.5f * (Z - Zm), -1.0f);
            c[h] = cmul(conj(A), B);
        }
    }
    device float2* out = packed + j * nn;
    out[e] = c[0] + mul_i(c[1], 1.0f);
    if (m != e) out[m] = conj(c[0]) + mul_i(conj(c[1]), 1.0f);
}

/// Window 2j's correlation is the real part of packed plane j, window 2j + 1's the imaginary.
struct DevicePlane {
    device const float2* plane;
    float scale;
    bool imag;
    float operator()(uint i) const { return (imag ? plane[i].y : plane[i].x) * scale; }
};

/// Peak, peak ratio and sub-pixel position of each window's correlation, from the packed
/// inverse transforms. One threadgroup per window.
kernel void find_peaks(device const float2* planes [[buffer(0)]],
                       constant CorrParams& p [[buffer(1)]],
                       device WindowPeak* out [[buffer(2)]],
                       uint w [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint T [[threads_per_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float sv[32];
    threadgroup uint si[32];
    WindowPeak in = out[w];
    if (in.status != STATUS_OK) return;
    uint nn = p.n * p.n;
    WindowPeak r = find_peak(DevicePlane{planes + (w / 2) * nn, in.height, (w & 1) != 0}, p.n, tid, T, lane, sg, nsg, sv, si);
    if (tid == 0) out[w] = r;
}

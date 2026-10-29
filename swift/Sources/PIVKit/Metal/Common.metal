// Shared definitions for the PIVKit kernels. The .metal files are concatenated in a fixed
// order (this one first) and compiled at runtime, so there are no #include lines between them.

#include <metal_stdlib>
using namespace metal;

// No fused multiply-add contraction: each a * b + c rounds twice, as in the Rust reference,
// so the CPU-ported stages (window placement, validation) reproduce it bit for bit.
#pragma METAL fp contract(off)

// MARK: - Complex arithmetic (float2 = re, im)

inline float2 cmul(float2 a, float2 b) { return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
inline float2 conj(float2 a) { return float2(a.x, -a.y); }
/// Multiplies by -i (forward) or +i (inverse): sign = -1 or +1.
inline float2 mul_i(float2 a, float sign) { return float2(-sign * a.y, sign * a.x); }

// MARK: - Threadgroup reductions

/// Sum of `v` over the threadgroup. Every thread gets the same total, adding the per-SIMD-group
/// partials in the same order, so the result is identical across threads. `scratch` needs one
/// slot per SIMD group. Must be called by every thread of the threadgroup.
template <typename T>
inline T tg_sum(T v, threadgroup T* scratch, uint lane, uint sg, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    T total = T(0);
    for (uint i = 0; i < nsg; i++) total += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

template <typename T>
inline T tg_max(T v, threadgroup T* scratch, uint lane, uint sg, uint nsg) {
    v = simd_max(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    T m = scratch[0];
    for (uint i = 1; i < nsg; i++) m = max(m, scratch[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

// MARK: - Interrogation windows

/// Per-pass parameters for the correlation kernels. Mirrors `CorrelationParams` in Swift.
struct CorrParams {
    uint n;          // window side, px
    uint width;      // image size, px
    uint height;
    uint count;      // windows in this dispatch
    float min_mean;  // signal mask thresholds on the stats image
    float min_std;
    uint log2n;      // fused power-of-two path only
};

/// One window's result. Mirrors `WindowPeak` in Swift.
struct WindowPeak {
    float dx;      // peak position relative to zero shift, px (sub-pixel)
    float dy;
    float ppr;     // primary / secondary peak ratio
    float height;  // normalized correlation at the peak
    uint status;   // 0 ok, 1 no signal (mask), 2 flat window (no intensity variation)
};

constant uint STATUS_OK = 0;
constant uint STATUS_MASKED = 1;
constant uint STATUS_FLAT = 2;
constant float MAX_PPR = 1000.0f;

/// A window placed at a (possibly fractional) top-left origin, sampled the way
/// `piv_core::correlate::load_window` does: an integer copy when both fractional parts are
/// below 1e-4, bilinear interpolation (in this exact operation order) otherwise.
struct WindowSampler {
    device const float* img;
    uint width;
    uint x0, y0;
    float tx, ty;
    float w00, w01, w10, w11;
    bool interp;

    WindowSampler(device const float* img, uint width, float x, float y) : img(img), width(width) {
        float fx = floor(x), fy = floor(y);
        x0 = uint(fx);
        y0 = uint(fy);
        tx = x - fx;
        ty = y - fy;
        interp = !(tx < 1e-4f && ty < 1e-4f);
        w00 = (1.0f - tx) * (1.0f - ty);
        w01 = tx * (1.0f - ty);
        w10 = (1.0f - tx) * ty;
        w11 = tx * ty;
    }

    float operator()(uint r, uint c) const {
        device const float* top = img + (y0 + r) * width + x0;
        if (!interp) return top[c];
        device const float* bot = top + width;
        return w00 * top[c] + w01 * top[c + 1] + w10 * bot[c] + w11 * bot[c + 1];
    }
};

/// Mean and standard deviation of the stats image over the n × n box at the rounded origin
/// (the window signal mask). Two-pass, so f32 is accurate enough to stand in for Rust's f64
/// integral image.
inline float2 box_stats(device const float* stats, uint width, float x, float y, uint n,
                        uint tid, uint T, threadgroup float* scratch, uint lane, uint sg, uint nsg) {
    uint bx = uint(round(x)), by = uint(round(y));
    float s = 0;
    for (uint e = tid; e < n * n; e += T) s += stats[(by + e / n) * width + bx + e % n];
    float mean = tg_sum(s, scratch, lane, sg, nsg) / float(n * n);
    float q = 0;
    for (uint e = tid; e < n * n; e += T) {
        float d = stats[(by + e / n) * width + bx + e % n] - mean;
        q += d * d;
    }
    float var = tg_sum(q, scratch, lane, sg, nsg) / float(n * n);
    return float2(mean, sqrt(max(var, 0.0f)));
}

/// Loads the window pair at `o` = (ax, ay, bx, by), checking the signal mask and removing each
/// window's mean. Calls `store(r, c, float2(a, b))` for every element and returns the status;
/// `energy` receives Σa², Σb² after mean removal. Must be called by every thread.
template <typename Store>
inline uint load_window_pair(device const float* imgA, device const float* imgB,
                             device const float* statsA, device const float* statsB,
                             constant CorrParams& p, float4 o, Store store, thread float2& energy,
                             uint tid, uint T, threadgroup float* scratch1, threadgroup float2* scratch2,
                             uint lane, uint sg, uint nsg) {
    uint n = p.n;
    float2 sa = box_stats(statsA, p.width, o.x, o.y, n, tid, T, scratch1, lane, sg, nsg);
    float2 sb = box_stats(statsB, p.width, o.z, o.w, n, tid, T, scratch1, lane, sg, nsg);
    if (!(sa.x >= p.min_mean && sa.y >= p.min_std && sb.x >= p.min_mean && sb.y >= p.min_std))
        return STATUS_MASKED;

    WindowSampler a(imgA, p.width, o.x, o.y), b(imgB, p.width, o.z, o.w);
    float2 s = 0;
    for (uint e = tid; e < n * n; e += T) s += float2(a(e / n, e % n), b(e / n, e % n));
    float2 mean = tg_sum(s, scratch2, lane, sg, nsg) / float(n * n);
    float2 q = 0;
    for (uint e = tid; e < n * n; e += T) {
        float2 d = float2(a(e / n, e % n), b(e / n, e % n)) - mean;
        q += d * d;
    }
    energy = tg_sum(q, scratch2, lane, sg, nsg);
    if (energy.x <= 1e-12f || energy.y <= 1e-12f) return STATUS_FLAT;
    // Stored only after the last reduction: in the fused kernel the scratch lives inside the
    // plane being written.
    for (uint e = tid; e < n * n; e += T)
        store(e / n, e % n, float2(a(e / n, e % n), b(e / n, e % n)) - mean);
    return STATUS_OK;
}

// MARK: - Spectra

/// Cross spectrum conj(A)·B from the transform Z of the packed window pair a + ib, in place.
/// A = (Z(k) + conj(Z(−k))) / 2 and B = (Z(k) − conj(Z(−k))) / 2i; since a and b are real,
/// the result at −k is the conjugate of the result at k, so each thread handles both.
template <typename Plane>
inline void cross_spectrum_pair(Plane z, uint n, uint e) {
    uint ky = e / n, kx = e % n;
    uint m = ((n - ky) % n) * n + (n - kx) % n;
    if (e > m) return;
    float2 Z = z[e], Zm = conj(z[m]);
    float2 A = 0.5f * (Z + Zm);
    float2 B = mul_i(0.5f * (Z - Zm), -1.0f);
    float2 C = cmul(conj(A), B);
    z[e] = C;
    if (m != e) z[m] = conj(C);
}

// MARK: - Peak finding

/// Three-point sub-pixel offset of a peak `c` between neighbours `m` (−1) and `p` (+1):
/// Gaussian when all three are positive, parabolic otherwise, clamped to ±0.5.
inline float subpixel(float m, float c, float p) {
    float d;
    if (m > 0 && c > 0 && p > 0) {
        float lm = log(m), lc = log(c), lp = log(p);
        d = (lm - lp) / (2.0f * (lm - 2.0f * lc + lp));
    } else {
        d = (m - p) / (2.0f * (m - 2.0f * c + p));
    }
    return isfinite(d) ? clamp(d, -0.5f, 0.5f) : 0.0f;
}

/// Port of `piv_core::correlate::find_peak` for a circular n × n plane, run by a whole
/// threadgroup. `at(i)` reads the (scaled) correlation at natural row-major index i. The
/// results are only valid in thread 0.
template <typename Plane>
inline WindowPeak find_peak(Plane at, uint n, uint tid, uint T, uint lane, uint sg, uint nsg,
                            threadgroup float* sv, threadgroup uint* si) {
    uint nn = n * n;
    // Highest value; ties go to the lowest index, as in a row-major scan with `>`.
    float best = -INFINITY;
    uint bi = 0xffffffffu;
    for (uint i = tid; i < nn; i += T) {
        float v = at(i);
        if (v > best) { best = v; bi = i; }
    }
    float m = simd_max(best);
    uint mi = simd_min(best == m ? bi : 0xffffffffu);
    if (lane == 0) { sv[sg] = m; si[sg] = mi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    best = sv[0];
    bi = si[0];
    for (uint i = 1; i < nsg; i++) {
        if (sv[i] > best || (sv[i] == best && si[i] < bi)) { best = sv[i]; bi = si[i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint by = bi / n, bx = bi % n;

    // Half-width at half maximum along ±x and ±y (one direction per thread).
    if (tid < 4) {
        uint2 step = tid == 0 ? uint2(1, 0) : tid == 1 ? uint2(n - 1, 0) : tid == 2 ? uint2(0, 1) : uint2(0, n - 1);
        uint k = 0;
        while (k < n / 2 && at(((by + k * step.x) % n) * n + (bx + k * step.y) % n) > 0.5f * best) k++;
        si[tid] = k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint hw = max(max(si[0], si[1]), max(si[2], si[3]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint exclude = max(uint(ceil(1.5f * float(hw))), 2u);

    // Highest local maximum outside the primary peak's footprint.
    float second = -INFINITY;
    for (uint i = tid; i < nn; i += T) {
        uint y = i / n, x = i % n;
        uint dy = y > by ? y - by : by - y, dx = x > bx ? x - bx : bx - x;
        if (max(min(dy, n - dy), min(dx, n - dx)) <= exclude) continue;
        float v = at(i);
        if (!(v > second)) continue;
        bool local_max = true;
        for (uint oy = n - 1, j = 0; j < 3; j++, oy++) {
            for (uint ox = n - 1, l = 0; l < 3; l++, ox++) {
                if (j == 1 && l == 1) continue;
                if (at(((y + oy) % n) * n + (x + ox) % n) > v) local_max = false;
            }
        }
        if (local_max) second = v;
    }
    second = tg_max(second, sv, lane, sg, nsg);

    WindowPeak r;
    r.status = STATUS_OK;
    r.height = best;
    r.ppr = second > 0 ? min(best / second, MAX_PPR) : MAX_PPR;
    float sx = subpixel(at(by * n + (bx + n - 1) % n), best, at(by * n + (bx + 1) % n));
    float sy = subpixel(at(((by + n - 1) % n) * n + bx), best, at(((by + 1) % n) * n + bx));
    r.dx = (bx >= n / 2 ? float(bx) - float(n) : float(bx)) + sx;
    r.dy = (by >= n / 2 ? float(by) - float(n) : float(by)) + sy;
    return r;
}

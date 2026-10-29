// Window placement and validation on the GPU, so a whole multi-pass pair runs in one command
// buffer. Ports of PivAnalyzer::pass (window placement, predictor sampling), the pair
// assembly, and piv_core::validate, with the same f32 operations in the same order. Fields are
// row-major [ny][nx]; one thread per grid cell.

constant uchar F_NO_SIGNAL = 1, F_LOW_PPR = 2, F_MEDIAN = 4, F_REPLACED = 8, F_OUT_OF_RANGE = 16;
constant uchar F_INVALID = 1 | 2 | 4 | 16;

/// Mirrors `FieldParams` in Swift.
struct FieldParams {
    uint nx, ny, step, window, x0, y0;            // this pass's grid
    uint pnx, pny, pstep, pwindow, px0, py0;      // the predictor's grid (previous pass)
    uint has_predictor;
    float max_x, max_y;                           // largest window origin with room to interpolate
    float max_displacement;                       // validation limits and thresholds
    float ppr_threshold, median_threshold, median_eps;
    uint replace;
};

inline float center(uint origin0, uint i, uint step, uint window) {
    return float(origin0 + i * step) + (float(window) - 1.0f) / 2.0f;
}

/// `VectorField::sample`: bilinear interpolation of the predictor at pixel (x, y), clamped to
/// the grid, skipping NaN samples; (0, 0) if every neighbour is NaN.
inline float2 sample_predictor(device const float* dx, device const float* dy, constant FieldParams& p, float x, float y) {
    float fx = clamp((x - center(p.px0, 0, p.pstep, p.pwindow)) / float(p.pstep), 0.0f, float(p.pnx - 1));
    float fy = clamp((y - center(p.py0, 0, p.pstep, p.pwindow)) / float(p.pstep), 0.0f, float(p.pny - 1));
    uint ix = uint(floor(fx)), iy = uint(floor(fy));
    float tx = fx - float(ix), ty = fy - float(iy);
    uint ix1 = min(ix + 1, p.pnx - 1), iy1 = min(iy + 1, p.pny - 1);
    uint rr[4] = {iy, iy, iy1, iy1};
    uint cc[4] = {ix, ix1, ix, ix1};
    float ww[4] = {(1.0f - tx) * (1.0f - ty), tx * (1.0f - ty), (1.0f - tx) * ty, tx * ty};
    float au = 0, av = 0, aw = 0;
    for (uint j = 0; j < 4; j++) {
        float u = dx[rr[j] * p.pnx + cc[j]], v = dy[rr[j] * p.pnx + cc[j]];
        if (ww[j] > 0 && isfinite(u) && isfinite(v)) {
            au = au + ww[j] * u;
            av = av + ww[j] * v;
            aw = aw + ww[j];
        }
    }
    return aw > 0 ? float2(au / aw, av / aw) : float2(0);
}

/// Splits the predicted shift symmetrically between the two windows (moving both inward
/// together near the border). Returns (a0, a0 + shift).
inline float2 place(uint origin, float shift, float maxv) {
    float s = clamp(shift, -maxv, maxv);
    float a0 = clamp(float(origin) - s / 2.0f, max(-s, 0.0f), min(maxv - s, maxv));
    return float2(a0, a0 + s);
}

kernel void place_windows(device const float* pdx [[buffer(0)]],
                          device const float* pdy [[buffer(1)]],
                          constant FieldParams& p [[buffer(2)]],
                          device float4* origins [[buffer(3)]],
                          uint k [[thread_position_in_grid]]) {
    if (k >= p.nx * p.ny) return;
    uint ix = k % p.nx, iy = k / p.nx;
    float2 shift = p.has_predictor
        ? sample_predictor(pdx, pdy, p, center(p.x0, ix, p.step, p.window), center(p.y0, iy, p.step, p.window))
        : float2(0);
    float2 x = place(p.x0 + ix * p.step, shift.x, p.max_x);
    float2 y = place(p.y0 + iy * p.step, shift.y, p.max_y);
    origins[k] = float4(x.x, y.x, x.y, y.y);
}

/// Window results → displacement field, plus the per-vector tests of `validate` (out of range,
/// low peak ratio). `raw_flags` gets the flags before validation (for tracing).
kernel void assemble_field(device const float4* origins [[buffer(0)]],
                           device const WindowPeak* peaks [[buffer(1)]],
                           constant FieldParams& p [[buffer(2)]],
                           device float* dx [[buffer(3)]],
                           device float* dy [[buffer(4)]],
                           device float* ppr [[buffer(5)]],
                           device uchar* flags [[buffer(6)]],
                           uint k [[thread_position_in_grid]]) {
    if (k >= p.nx * p.ny) return;
    WindowPeak pk = peaks[k];
    if (pk.status != STATUS_OK) {
        dx[k] = NAN;
        dy[k] = NAN;
        ppr[k] = 0;
        flags[k] = F_NO_SIGNAL;
        return;
    }
    float4 o = origins[k];
    float u = o.z - o.x + pk.dx, v = o.w - o.y + pk.dy;
    dx[k] = u;
    dy[k] = v;
    ppr[k] = pk.ppr;
    uchar f = 0;
    if (sqrt(u * u + v * v) > p.max_displacement) f |= F_OUT_OF_RANGE;
    if (pk.ppr < p.ppr_threshold) f |= F_LOW_PPR;
    flags[k] = f;
}

/// Sorts a few values in place and returns their median (mean of the middle two if even).
inline float small_median(thread float* v, uint n) {
    for (uint i = 1; i < n; i++) {
        float x = v[i];
        uint j = i;
        while (j > 0 && v[j - 1] > x) { v[j] = v[j - 1]; j--; }
        v[j] = x;
    }
    return n % 2 == 1 ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

/// Usable neighbours of (ix, iy) within Chebyshev `radius` (excluding itself).
inline uint gather(device const float* dx, device const float* dy, device const uchar* flags, constant FieldParams& p,
                   uint ix, uint iy, uint radius, thread float* nu, thread float* nv) {
    uint n = 0;
    uint y0 = iy > radius ? iy - radius : 0, y1 = min(iy + radius, p.ny - 1);
    uint x0 = ix > radius ? ix - radius : 0, x1 = min(ix + radius, p.nx - 1);
    for (uint y = y0; y <= y1; y++)
        for (uint x = x0; x <= x1; x++) {
            uint j = y * p.nx + x;
            if ((x != ix || y != iy) && (flags[j] & F_INVALID) == 0) {
                nu[n] = dx[j];
                nv[n] = dy[j];
                n++;
            }
        }
    return n;
}

/// Normalized median residual of `value` against its neighbours (which get reordered).
inline float residual(float value, thread float* nb, uint n, float eps) {
    float med = small_median(nb, n);
    float dev[24];
    for (uint i = 0; i < n; i++) dev[i] = abs(nb[i] - med);
    return abs(value - med) / (small_median(dev, n) + eps);
}

/// Normalized median test (Westerweel & Scarano) against the 8-neighbourhood, using the flags
/// after the per-vector tests.
kernel void median_test(device const float* dx [[buffer(0)]],
                        device const float* dy [[buffer(1)]],
                        device const uchar* flags_in [[buffer(2)]],
                        device uchar* flags_out [[buffer(3)]],
                        constant FieldParams& p [[buffer(4)]],
                        uint k [[thread_position_in_grid]]) {
    if (k >= p.nx * p.ny) return;
    uchar f = flags_in[k];
    flags_out[k] = f;
    if ((f & F_INVALID) != 0) return;
    float nu[24], nv[24];
    uint n = gather(dx, dy, flags_in, p, k % p.nx, k / p.nx, 1, nu, nv);
    if (n < 2) return;
    float ru = residual(dx[k], nu, n, p.median_eps);
    float rv = residual(dy[k], nv, n, p.median_eps);
    if (sqrt(ru * ru + rv * rv) > p.median_threshold) flags_out[k] = f | F_MEDIAN;
}

/// Replaces each rejected vector with the median of its usable neighbours (radius 1, else 2),
/// or NaN if there are none. Reads the fields before replacement and writes new ones, so the
/// threads don't see each other's replacements.
kernel void replace_outliers(device const float* dx [[buffer(0)]],
                             device const float* dy [[buffer(1)]],
                             device const uchar* flags_in [[buffer(2)]],
                             device float* dx_out [[buffer(3)]],
                             device float* dy_out [[buffer(4)]],
                             device uchar* flags_out [[buffer(5)]],
                             constant FieldParams& p [[buffer(6)]],
                             uint k [[thread_position_in_grid]]) {
    if (k >= p.nx * p.ny) return;
    uchar f = flags_in[k];
    dx_out[k] = dx[k];
    dy_out[k] = dy[k];
    flags_out[k] = f;
    bool rejected = (f & F_NO_SIGNAL) == 0 && (f & F_INVALID) != 0;
    if (!p.replace || !rejected) return;
    float nu[24], nv[24];
    for (uint radius = 1; radius <= 2; radius++) {
        uint n = gather(dx, dy, flags_in, p, k % p.nx, k / p.nx, radius, nu, nv);
        if (n > 0) {
            dx_out[k] = small_median(nu, n);
            dy_out[k] = small_median(nv, n);
            flags_out[k] = f | F_REPLACED;
            return;
        }
    }
    dx_out[k] = NAN;
    dy_out[k] = NAN;
}

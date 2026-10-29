// Frame preparation: port of piv_core::preprocess (PreparedFrame::new, gaussian_blur,
// local_normalize, BackgroundEstimator). Images are dense row-major float buffers.

struct ImageParams {
    uint width;      // output (ROI) size
    uint height;
    uint src_x;      // ROI origin in the source texture
    uint src_y;
    uint radius;     // blur / normalize radius
    float eps;       // local normalize ε
    uint channel;    // upload: 0 = plane value (r), 1 = green of a BGRA texture
    uint flag;       // upload: 1 = subtract background; accumulate: 0 min, 1 sum, 2 first frame
};

/// Texture plane → float image (the ROI), normalized to 0–1 by the texture format (v/255 for
/// 8-bit, v/65535 for 16-bit storage). With a background, also writes the background-subtracted
/// image clamped at 0, the base for the signal mask and for preprocessing.
kernel void upload_frame(texture2d<float, access::read> src [[texture(0)]],
                         constant ImageParams& p [[buffer(0)]],
                         device float* raw [[buffer(1)]],
                         device float* base [[buffer(2)]],
                         device const float* background [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height) return;
    float4 t = src.read(uint2(gid.x + p.src_x, gid.y + p.src_y));
    float v = p.channel == 1 ? t.g : t.r;
    uint i = gid.y * p.width + gid.x;
    raw[i] = v;
    base[i] = p.flag == 1 ? max(v - background[i], 0.0f) : v;
}

/// Horizontal pass of the separable Gaussian, edges clamped. `kernel` has 2·radius + 1
/// weights summing to 1; terms are added in kernel order, as in Rust.
kernel void blur_rows(device const float* src [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      constant ImageParams& p [[buffer(2)]],
                      constant float* weights [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height) return;
    int r = int(p.radius), w = int(p.width);
    device const float* row = src + gid.y * p.width;
    float s = 0;
    for (int k = 0; k <= 2 * r; k++) s += weights[k] * row[clamp(int(gid.x) + k - r, 0, w - 1)];
    dst[gid.y * p.width + gid.x] = s;
}

kernel void blur_cols(device const float* src [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      constant ImageParams& p [[buffer(2)]],
                      constant float* weights [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height) return;
    int r = int(p.radius), h = int(p.height);
    float s = 0;
    for (int k = 0; k <= 2 * r; k++) s += weights[k] * src[clamp(int(gid.y) + k - r, 0, h - 1) * p.width + gid.x];
    dst[gid.y * p.width + gid.x] = s;
}

/// Local normalization, first pass: Σv and Σv² over the clipped horizontal run
/// [x − r, x + r] of each row.
kernel void box_rows(device const float* src [[buffer(0)]],
                     device float2* sums [[buffer(1)]],
                     constant ImageParams& p [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height) return;
    uint r = p.radius;
    uint x0 = gid.x > r ? gid.x - r : 0, x1 = min(gid.x + r + 1, p.width);
    device const float* row = src + gid.y * p.width;
    float2 s = 0;
    for (uint x = x0; x < x1; x++) s += float2(row[x], row[x] * row[x]);
    sums[gid.y * p.width + gid.x] = s;
}

/// Local normalization, second pass: subtract the mean and divide by the standard deviation
/// (plus ε) over the (2r+1)² box clipped at the image edges, from column sums of the first
/// pass. var = E[v²] − mean² in f32: for 0–1 intensities over ≤ 81 pixels this stays within
/// ~1e-6 of Rust's f64 integral image.
kernel void normalize_cols(device const float* src [[buffer(0)]],
                           device const float2* sums [[buffer(1)]],
                           device float* dst [[buffer(2)]],
                           constant ImageParams& p [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height) return;
    uint r = p.radius;
    uint x0 = gid.x > r ? gid.x - r : 0, x1 = min(gid.x + r + 1, p.width);
    uint y0 = gid.y > r ? gid.y - r : 0, y1 = min(gid.y + r + 1, p.height);
    float2 s = 0;
    for (uint y = y0; y < y1; y++) s += sums[y * p.width + gid.x];
    float n = float((x1 - x0) * (y1 - y0));
    float mean = s.x / n;
    float sd = sqrt(max(s.y / n - mean * mean, 0.0f));
    dst[gid.y * p.width + gid.x] = (src[gid.y * p.width + gid.x] - mean) / (sd + p.eps);
}

/// Background accumulation: per-pixel minimum, or a running sum for the mean (divided by the
/// frame count afterwards with `scale_image`). `flag` 2 copies the first frame.
kernel void accumulate_background(device float* acc [[buffer(0)]],
                                  device const float* frame [[buffer(1)]],
                                  constant ImageParams& p [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= p.width * p.height) return;
    float v = frame[gid];
    acc[gid] = p.flag == 2 ? v : p.flag == 0 ? min(acc[gid], v) : acc[gid] + v;
}

kernel void scale_image(device float* img [[buffer(0)]],
                        constant ImageParams& p [[buffer(1)]],
                        constant float& factor [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= p.width * p.height) return;
    img[gid] /= factor;
}

// Quiver rendering (port of piv_io::render::render_quiver): the analysed frame in gray, then
// one arrow per vector, coloured by magnitude (magenta for replaced vectors). Arrows are
// instanced quads around each stroke, antialiased by distance to the stroke in the fragment
// shader. Image coordinates follow the grid's pixel-centre convention (pixel i spans
// [i − 0.5, i + 0.5]); the framebuffer's pixel i is centred at i + 0.5.

struct QuiverParams {
    uint width, height;      // image (ROI) size; the target may be larger (padded to even)
    uint target_w, target_h; // render target size
    float gain;              // background intensity multiplier
    float arrow_scale;       // arrow length per px of displacement
    float color_max;         // displacement at the top of the colour map
    float half_width;        // stroke half-width, px
    uint nx, every;          // grid columns; draw every n-th vector in each direction
};

struct QuiverVector {
    float2 center;  // grid point, image px
    float2 d;       // displacement, px (NaN = missing)
    uint flags;
};

struct BackgroundOut {
    float4 position [[position]];
};

vertex BackgroundOut quiver_background_vertex(uint vid [[vertex_id]]) {
    // One triangle covering the target.
    float2 p = float2((vid << 1) & 2, vid & 2);
    return BackgroundOut{float4(p * 2.0f - 1.0f, 0, 1)};
}

/// Gray background: min(raw · gain, 1) · 180, truncated to 8 bits as in Rust. Black padding.
fragment float4 quiver_background_fragment(BackgroundOut in [[stage_in]],
                                           device const float* raw [[buffer(0)]],
                                           constant QuiverParams& p [[buffer(1)]]) {
    uint x = uint(in.position.x), y = uint(in.position.y);
    if (x >= p.width || y >= p.height) return float4(0, 0, 0, 1);
    float g = floor(min(raw[y * p.width + x] * p.gain, 1.0f) * 180.0f) / 255.0f;
    return float4(g, g, g, 1);
}

/// Light blue → yellow → red.
inline float3 quiver_colormap(float t) {
    t = clamp(t, 0.0f, 1.0f);
    float3 c0 = float3(90, 170, 255), c1 = float3(255, 230, 80), c2 = float3(255, 70, 50);
    float3 c = t < 0.5f ? mix(c0, c1, t / 0.5f) : mix(c1, c2, (t - 0.5f) / 0.5f);
    return round(c) / 255.0f;
}

struct StrokeOut {
    float4 position [[position]];
    float2 pixel;              // framebuffer pixel coordinates of this vertex
    float2 a [[flat]];         // stroke endpoints, framebuffer pixels
    float2 b [[flat]];
    float3 color [[flat]];
};

/// Instance = vector · 3 + stroke (0 shaft, 1 and 2 the head's sides); 4 vertices per instance
/// (triangle strip) around the stroke, padded for antialiasing. Missing vectors, skipped grid
/// points and heads of very short arrows collapse to nothing.
vertex StrokeOut quiver_stroke_vertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                      device const QuiverVector* vectors [[buffer(0)]],
                                      constant QuiverParams& p [[buffer(1)]]) {
    StrokeOut out;
    uint k = iid / 3, part = iid % 3;
    QuiverVector v = vectors[k];
    uint ix = k % p.nx, iy = k / p.nx;
    bool skip = !isfinite(v.d.x) || !isfinite(v.d.y) || ix % p.every != 0 || iy % p.every != 0;
    float2 from = v.center, to = v.center + v.d * p.arrow_scale;
    float2 dv = to - from;
    float len = length(dv);
    float2 a = from, b = to;
    if (part > 0) {
        if (len < 2.0f) skip = true;
        float head = clamp(0.3f * len, 2.0f, 8.0f);
        float2 u = dv / max(len, 1e-6f);
        float side = part == 1 ? -1.0f : 1.0f;
        float c = cos(0.42f), s = side * sin(0.42f);
        float2 back = float2(-(u.x * c - u.y * s), -(u.x * s + u.y * c));
        a = to;
        b = to + head * back;
    }
    // To framebuffer pixels.
    a += 0.5f;
    b += 0.5f;
    float2 dir = b - a;
    float l = length(dir);
    float2 t = l > 1e-6f ? dir / l : float2(1, 0);
    float2 nrm = float2(-t.y, t.x);
    float pad = p.half_width + 1.0f;
    float2 corner = (vid & 1 ? b + t * pad : a - t * pad) + (vid & 2 ? nrm : -nrm) * pad;
    if (skip) corner = float2(-10, -10);
    out.pixel = corner;
    out.position = float4(corner.x / float(p.target_w) * 2.0f - 1.0f, 1.0f - corner.y / float(p.target_h) * 2.0f, 0, 1);
    out.a = a;
    out.b = b;
    out.color = (v.flags & 8) != 0 ? float3(1, 0, 1) : quiver_colormap(length(v.d) / p.color_max);
    return out;
}

fragment float4 quiver_stroke_fragment(StrokeOut in [[stage_in]], constant QuiverParams& p [[buffer(0)]]) {
    float2 ab = in.b - in.a;
    float h = clamp(dot(in.pixel - in.a, ab) / max(dot(ab, ab), 1e-6f), 0.0f, 1.0f);
    float d = length(in.pixel - in.a - h * ab);
    float coverage = clamp(p.half_width + 0.5f - d, 0.0f, 1.0f);
    return float4(in.color, coverage);
}

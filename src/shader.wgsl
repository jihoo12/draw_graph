// shader.wgsl
struct Camera {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    let res: u32 = 50u;
    let size: f32 = 10.0;
    let step: f32 = size / f32(res - 1u);
    let offset: f32 = size / 2.0;

    let total_v_per_dir = res * (res - 1u) * 2u;
    var x_idx: u32;
    var z_idx: u32;

    if (vertex_idx < total_v_per_dir) {
        // 가로선 (X축 방향으로 이어지는 선들)
        let line_idx = vertex_idx / 2u;
        z_idx = line_idx / (res - 1u);
        x_idx = line_idx % (res - 1u);
        if (vertex_idx % 2u == 1u) { x_idx += 1u; }
    } else {
        // 세로선 (Z축 방향으로 이어지는 선들)
        let i = vertex_idx - total_v_per_dir;
        let line_idx = i / 2u;
        x_idx = line_idx / (res - 1u);
        z_idx = line_idx % (res - 1u);
        if (i % 2u == 1u) { z_idx += 1u; }
    }

    let px = f32(x_idx) * step - offset;
    let pz = f32(z_idx) * step - offset;

    // --- 여기서 함수를 정의하세요 ---
    let d = sqrt(px * px + pz * pz);
    let py = sin(d);
    // ----------------------------

    var out: VertexOutput;
    out.clip_pos = camera.view_proj * vec4<f32>(px, py, pz, 1.0);

    // 높이에 따른 색상 변화 (파란색 -> 흰색)
    let color_val = (py + 1.0) / 2.0;
    out.color = vec3<f32>(color_val * 0.5, color_val, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

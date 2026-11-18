@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var scene: texture_2d<f32>;

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
	var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
	var o: VsOut;
	let xy = p[vi];
	o.pos = vec4f(xy, 0.0, 1.0);
	o.uv = vec2f(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);
	return o;
}

fn to_srgb1(c: f32) -> f32 {
	let x = clamp(c, 0.0, 1.0);
	if (x <= 0.0031308) {
		return x * 12.92;
	}
	return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

// The scene is premultiplied linear. The canvas is premultiplied sRGB, composited over the page by the browser.
@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	let s = textureSampleLevel(scene, samp, in.uv, 0.0);
	if (s.a <= 1e-5) {
		return vec4f(0.0);
	}
	let c = clamp(s.rgb / s.a, vec3f(0.0), vec3f(1.0));
	let srgb = vec3f(to_srgb1(c.r), to_srgb1(c.g), to_srgb1(c.b));
	return vec4f(srgb * s.a, s.a);
}

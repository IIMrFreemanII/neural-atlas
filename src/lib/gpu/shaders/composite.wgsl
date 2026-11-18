struct Params {
	strength: f32,
	pad0: f32,
	pad1: f32,
	pad2: f32,
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var bloom: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: Params;

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

const K: f32 = 0.8;

fn tonemap1(c: f32) -> f32 {
	if (c < K) {
		return c;
	}
	return K + (1.0 - K) * tanh((c - K) / (1.0 - K));
}

fn to_srgb1(c: f32) -> f32 {
	let x = clamp(c, 0.0, 1.0);
	if (x <= 0.0031308) {
		return x * 12.92;
	}
	return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	let s = textureSampleLevel(scene, samp, in.uv, 0.0).rgb;
	let b = textureSampleLevel(bloom, samp, in.uv, 0.0).rgb;
	var c = max(s + b * params.strength, vec3f(0.0));
	c = vec3f(tonemap1(c.r), tonemap1(c.g), tonemap1(c.b));
	return vec4f(to_srgb1(c.r), to_srgb1(c.g), to_srgb1(c.b), 1.0);
}

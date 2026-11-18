@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;

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

fn tex(uv: vec2f) -> vec3f {
	return textureSampleLevel(src, samp, uv, 0.0).rgb;
}

const THRESHOLD: f32 = 0.8;
const KNEE: f32 = 0.4;

fn soft_threshold(c: vec3f) -> vec3f {
	let b = max(c.r, max(c.g, c.b));
	let s = clamp(b - THRESHOLD + KNEE, 0.0, 2.0 * KNEE);
	let soft = s * s / (4.0 * KNEE + 1e-5);
	let contrib = max(soft, b - THRESHOLD) / max(b, 1e-5);
	return c * contrib;
}

// Reads the full-resolution scene, writes level 0 (half resolution).
@fragment
fn fs_prefilter(in: VsOut) -> @location(0) vec4f {
	let t = 1.0 / vec2f(textureDimensions(src));
	var c = vec3f(0.0);
	// Threshold each 2x2 bilinear tap separately to limit fireflies.
	c += soft_threshold(tex(in.uv + t * vec2f(-1.0, -1.0)));
	c += soft_threshold(tex(in.uv + t * vec2f(1.0, -1.0)));
	c += soft_threshold(tex(in.uv + t * vec2f(-1.0, 1.0)));
	c += soft_threshold(tex(in.uv + t * vec2f(1.0, 1.0)));
	return vec4f(c * 0.25, 1.0);
}

@fragment
fn fs_down(in: VsOut) -> @location(0) vec4f {
	let t = 1.0 / vec2f(textureDimensions(src));
	var c = tex(in.uv + t * vec2f(-1.0, -1.0));
	c += tex(in.uv + t * vec2f(1.0, -1.0));
	c += tex(in.uv + t * vec2f(-1.0, 1.0));
	c += tex(in.uv + t * vec2f(1.0, 1.0));
	return vec4f(c * 0.25, 1.0);
}

// 3x3 tent filter, additively blended onto the destination level.
@fragment
fn fs_up(in: VsOut) -> @location(0) vec4f {
	let t = 1.0 / vec2f(textureDimensions(src));
	var c = tex(in.uv) * 4.0;
	c += tex(in.uv + t * vec2f(-1.0, 0.0)) * 2.0;
	c += tex(in.uv + t * vec2f(1.0, 0.0)) * 2.0;
	c += tex(in.uv + t * vec2f(0.0, -1.0)) * 2.0;
	c += tex(in.uv + t * vec2f(0.0, 1.0)) * 2.0;
	c += tex(in.uv + t * vec2f(-1.0, -1.0));
	c += tex(in.uv + t * vec2f(1.0, -1.0));
	c += tex(in.uv + t * vec2f(-1.0, 1.0));
	c += tex(in.uv + t * vec2f(1.0, 1.0));
	return vec4f(c / 16.0, 0.0);
}

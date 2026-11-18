struct ViewU {
	viewProj: mat4x4f,
	viewport: vec2f,
	dpr: f32,
	time: f32,
};
@group(0) @binding(0) var<uniform> view: ViewU;

struct VsIn {
	@builtin(vertex_index) vi: u32,
	@location(0) posR: vec4f,   // x, y, z, radius (css px)
	@location(1) color: vec4f,  // linear, straight alpha
	@location(2) misc: vec4f,   // ring (css px), softness, 0, 0
};

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) local: vec2f,                 // device px from center
	@location(1) color: vec4f,
	@location(2) @interpolate(flat) shape: vec4f, // radius dev px, ring dev px, softness, 0
};

@vertex
fn vs(in: VsIn) -> VsOut {
	var corners = array<vec2f, 6>(
		vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
		vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
	);
	let corner = corners[in.vi];
	let radius = in.posR.w * view.dpr;
	let extent = radius + 1.0;
	let clip = view.viewProj * vec4f(in.posR.xyz, 1.0);
	let offsetNdc = corner * extent * 2.0 / view.viewport;

	var o: VsOut;
	o.pos = vec4f(clip.xy + offsetNdc * clip.w, clip.z, clip.w);
	o.local = corner * extent;
	o.color = in.color;
	o.shape = vec4f(radius, in.misc.x * view.dpr, in.misc.y, 0.0);
	return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	let d = length(in.local);
	let r = max(in.shape.x, 0.001);
	let ringW = in.shape.y;
	let soft = clamp(in.shape.z, 0.0, 1.0);

	var crisp = clamp(r - d + 0.5, 0.0, 1.0);
	if (ringW > 0.0) {
		crisp = crisp * clamp(d - (r - ringW) + 0.5, 0.0, 1.0);
	}

	let x = clamp(d / r, 0.0, 1.0);
	let e4 = exp(-4.0);
	let glow = (exp(-4.0 * x * x) - e4) / (1.0 - e4);

	let cov = mix(crisp, glow, soft);
	let a = in.color.a * cov;
	return vec4f(in.color.rgb * a, a);
}

struct ViewU {
	viewProj: mat4x4f,
	viewport: vec2f,
	dpr: f32,
	time: f32,
};
@group(0) @binding(0) var<uniform> view: ViewU;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

struct VsIn {
	@builtin(vertex_index) vi: u32,
	@location(0) originEdge: vec4f, // origin xyz, edgePx (css)
	@location(1) uOpacity: vec4f,   // u xyz, opacity
	@location(2) v: vec4f,          // v xyz, 0
	@location(3) fill: vec4f,       // linear, straight alpha
	@location(4) edge: vec4f,       // linear, straight alpha
};

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) fill: vec4f,
	@location(2) @interpolate(flat) edge: vec4f,
	@location(3) @interpolate(flat) misc: vec2f, // edge device px, opacity
};

@vertex
fn vs(in: VsIn) -> VsOut {
	var corners = array<vec2f, 6>(
		vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
		vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
	);
	let c = corners[in.vi];
	let w = in.originEdge.xyz + in.uOpacity.xyz * c.x + in.v.xyz * c.y;
	var o: VsOut;
	o.pos = view.viewProj * vec4f(w, 1.0);
	o.uv = c;
	o.fill = in.fill;
	o.edge = in.edge;
	o.misc = vec2f(in.originEdge.w * view.dpr, in.uOpacity.w);
	return o;
}

// Premultiplied "inkOver" of a straight-alpha color onto a premultiplied destination.
fn inkOver(dst: vec4f, rgb: vec3f, a: f32) -> vec4f {
	return vec4f(rgb * a + dst.rgb * (1.0 - a), a + dst.a * (1.0 - a));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	// Derivatives first, in uniform control flow.
	let gu = vec2f(dpdx(in.uv.x), dpdy(in.uv.x));
	let gv = vec2f(dpdx(in.uv.y), dpdy(in.uv.y));
	let t = textureSample(tex, samp, in.uv);

	var col = vec4f(in.fill.rgb * in.fill.a, in.fill.a);
	col = inkOver(col, t.rgb, t.a);

	// Distance inside the border, in device px, along each sheet axis.
	let pxU = min(in.uv.x, 1.0 - in.uv.x) / max(length(gu), 1e-6);
	let pxV = min(in.uv.y, 1.0 - in.uv.y) / max(length(gv), 1e-6);
	let d = min(pxU, pxV);
	let ecov = clamp(in.misc.x - d + 0.5, 0.0, 1.0) * select(0.0, 1.0, in.misc.x > 0.0);
	col = inkOver(col, in.edge.rgb, in.edge.a * ecov);

	return col * in.misc.y;
}

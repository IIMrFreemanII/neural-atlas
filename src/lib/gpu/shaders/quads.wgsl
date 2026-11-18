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
	@location(1) size: vec4f,   // w, h (world), 0, 0
	@location(2) color: vec4f,  // linear, straight alpha
};

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) local: vec2f,                      // device px from center
	@location(1) color: vec4f,
	@location(2) @interpolate(flat) shape: vec3f,   // half width px, half height px, radius px
};

@vertex
fn vs(in: VsIn) -> VsOut {
	var corners = array<vec2f, 6>(
		vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
		vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
	);
	let corner = corners[in.vi];
	let half_vp = view.viewport * 0.5;
	let c = in.posR.xyz;
	let hw = in.size.x * 0.5;
	let hh = in.size.y * 0.5;

	let cc = view.viewProj * vec4f(c, 1.0);
	let cx = view.viewProj * vec4f(c + vec3f(hw, 0.0, 0.0), 1.0);
	let cy = view.viewProj * vec4f(c + vec3f(0.0, hh, 0.0), 1.0);
	let pc = cc.xy / cc.w * half_vp;
	let vx = cx.xy / cx.w * half_vp - pc;
	let vy = cy.xy / cy.w * half_vp - pc;
	let lx = length(vx);
	let ly = length(vy);
	let dx = select(vec2f(1.0, 0.0), vx / max(lx, 1e-6), lx > 1e-6);
	let dy = select(vec2f(0.0, 1.0), vy / max(ly, 1e-6), ly > 1e-6);

	let pad = 1.0;
	let ex = lx + pad;
	let ey = ly + pad;
	let p = pc + dx * corner.x * ex + dy * corner.y * ey;

	var o: VsOut;
	o.pos = vec4f(p / half_vp * cc.w, cc.z, cc.w);
	o.local = vec2f(corner.x * ex, corner.y * ey);
	o.color = in.color;
	o.shape = vec3f(lx, ly, min(in.posR.w * view.dpr, min(lx, ly)));
	return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	let r = in.shape.z;
	let q = abs(in.local) - (in.shape.xy - vec2f(r));
	let d = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
	let cov = clamp(0.5 - d, 0.0, 1.0);
	let a = in.color.a * cov;
	return vec4f(in.color.rgb * a, a);
}

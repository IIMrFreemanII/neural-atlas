struct ViewU {
	viewProj: mat4x4f,
	viewport: vec2f,
	dpr: f32,
	time: f32,
};
@group(0) @binding(0) var<uniform> view: ViewU;

struct ShadowU {
	color: vec4f, // rgb linear, a = cast alpha
};
@group(1) @binding(0) var<uniform> sh: ShadowU;

struct VsIn {
	@builtin(vertex_index) vi: u32,
	// blob: center xyz, kind 0          quad: center xyz, kind 1
	@location(0) a: vec4f,
	// blob: dir x, dir z, semiMajor, semiMinor      quad: c0 x, c0 z, c1 x, c1 z
	@location(1) b: vec4f,
	// blob: unused                                  quad: c2 x, c2 z, c3 x, c3 z
	@location(2) c: vec4f,
	// alpha, blur, half extent of the drawn square, 0
	@location(3) d: vec4f,
};

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) xz: vec2f,
	@location(1) @interpolate(flat) a: vec4f,
	@location(2) @interpolate(flat) b: vec4f,
	@location(3) @interpolate(flat) c: vec4f,
	@location(4) @interpolate(flat) d: vec4f,
};

@vertex
fn vs(in: VsIn) -> VsOut {
	var corners = array<vec2f, 6>(
		vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
		vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
	);
	let k = corners[in.vi] * in.d.z;
	let w = vec3f(in.a.x + k.x, in.a.y, in.a.z + k.y);
	var o: VsOut;
	o.pos = view.viewProj * vec4f(w, 1.0);
	o.xz = w.xz;
	o.a = in.a;
	o.b = in.b;
	o.c = in.c;
	o.d = in.d;
	return o;
}

// Signed distance from p to the convex quad c0..c3 (ring order), negative inside.
fn edgeDist(p: vec2f, p0: vec2f, p1: vec2f, centroid: vec2f) -> f32 {
	let e = p1 - p0;
	var n = normalize(vec2f(e.y, -e.x));
	if (dot(n, centroid - p0) > 0.0) {
		n = -n;
	}
	return dot(p - p0, n);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	var cov = 0.0;
	if (in.a.w < 0.5) {
		// Blob: gaussian-ish falloff over an ellipse, zero at the edge.
		let rel = in.xz - in.a.xz;
		let dir = in.b.xy;
		let along = dot(rel, dir) / max(in.b.z, 1e-5);
		let across = dot(rel, vec2f(-dir.y, dir.x)) / max(in.b.w, 1e-5);
		let q2 = min(along * along + across * across, 1.0);
		let e3 = exp(-3.0);
		cov = (exp(-3.0 * q2) - e3) / (1.0 - e3);
	} else {
		let p0 = in.b.xy;
		let p1 = in.b.zw;
		let p2 = in.c.xy;
		let p3 = in.c.zw;
		let cen = (p0 + p1 + p2 + p3) * 0.25;
		let d = max(
			max(edgeDist(in.xz, p0, p1, cen), edgeDist(in.xz, p1, p2, cen)),
			max(edgeDist(in.xz, p2, p3, cen), edgeDist(in.xz, p3, p0, cen))
		);
		// Soft edge centred on the border, about one blur unit each side.
		let blur = max(in.d.y, 1e-4);
		let x = clamp(0.5 - d / (2.0 * blur), 0.0, 1.0);
		cov = x * x * (3.0 - 2.0 * x);
	}
	let a = sh.color.a * in.d.x * cov;
	return vec4f(sh.color.rgb * a, a);
}

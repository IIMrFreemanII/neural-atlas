struct ViewU {
	viewProj: mat4x4f,
	viewport: vec2f,
	dpr: f32,
	time: f32,
};
@group(0) @binding(0) var<uniform> view: ViewU;

struct MeshU {
	rect: vec4f,      // x0, z0, x1, z1
	colorLo: vec4f,
	colorHi: vec4f,
	params: vec4f,    // range lo, range hi, bands, opacity
	dims: vec4f,      // cols, rows, 0, 0
	inkFill: vec4f,   // paper colour, straight
	ink: vec4f,       // mesh + hatch ink, straight (a = density)
	contourInk: vec4f,
	flags: vec4f,     // style (1 = ink), gridEvery, hatch, contours
	hatchP: vec4f,    // hatch spacing css px, 0, 0, 0
};
@group(1) @binding(0) var<uniform> mesh: MeshU;

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) world: vec3f,
	@location(1) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @location(0) height: f32) -> VsOut {
	let cols = u32(mesh.dims.x);
	let col = vi % cols;
	let row = vi / cols;
	let u = f32(col) / (mesh.dims.x - 1.0);
	let v = f32(row) / (mesh.dims.y - 1.0);
	let w = vec3f(
		mix(mesh.rect.x, mesh.rect.z, u),
		height,
		mix(mesh.rect.y, mesh.rect.w, v)
	);
	var o: VsOut;
	o.pos = view.viewProj * vec4f(w, 1.0);
	o.world = w;
	o.uv = vec2f(u, v);
	return o;
}

// Premultiplied "inkOver" of a straight-alpha colour onto a premultiplied destination.
fn inkOver(dst: vec4f, rgb: vec3f, a: f32) -> vec4f {
	return vec4f(rgb * a + dst.rgb * (1.0 - a), a + dst.a * (1.0 - a));
}

// Coverage of a periodic diagonal line family: coord in device px along (x + sign*y), duty = ink fraction of the period.
fn hatchCov(coord: f32, spacingDev: f32, duty: f32) -> f32 {
	let period = spacingDev * 0.70710678; // perpendicular distance between lines
	let f = fract(coord / spacingDev);
	let dist = min(f, 1.0 - f) * spacingDev * 0.70710678;
	return clamp(duty * 0.5 * period - dist + 0.5, 0.0, 1.0) * select(0.0, 1.0, duty > 0.0);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	let range = mesh.params.xy;
	let bands = mesh.params.z;
	let span = max(range.y - range.x, 1e-6);
	let t = (in.world.y - range.x) / span;
	let s = t * bands;
	let fws = max(fwidth(s), 1e-6);
	let gridEvery = mesh.flags.y;
	let gcoord = in.uv * (mesh.dims.xy - 1.0) / max(gridEvery, 1.0);
	let gw = max(fwidth(gcoord), vec2f(1e-6));

	var n = normalize(cross(dpdx(in.world), dpdy(in.world)));
	if (n.y < 0.0) {
		n = -n;
	}
	let light = normalize(vec3f(-0.5, 0.8, 0.35));
	let lambert = max(dot(n, light), 0.0);

	let a = mesh.params.w;

	if (mesh.flags.x > 0.5) {
		// Line-art: paper fill, hatching by N.L, mesh lines, contours, all in ink.
		var col = vec4f(mesh.inkFill.rgb * mesh.inkFill.a, mesh.inkFill.a);
		let inkRgb = mesh.ink.rgb;

		if (mesh.flags.z > 0.5) {
			// 0..4 steps of coverage as the surface turns from the light.
			var level = 0;
			if (lambert < 0.75) { level = 1; }
			if (lambert < 0.55) { level = 2; }
			if (lambert < 0.35) { level = 3; }
			if (lambert < 0.15) { level = 4; }
			let sp = max(mesh.hatchP.x, 1.0) * view.dpr;
			let ca = in.pos.x + in.pos.y;
			let cb = in.pos.x - in.pos.y;
			var h = 0.0;
			if (level == 1) { h = hatchCov(ca, sp, 0.12); }
			if (level == 2) { h = hatchCov(ca, sp, 0.22); }
			if (level == 3) { h = max(hatchCov(ca, sp, 0.22), hatchCov(cb, sp, 0.12)); }
			if (level == 4) { h = max(hatchCov(ca, sp, 0.26), hatchCov(cb, sp, 0.26)); }
			col = inkOver(col, inkRgb, mesh.ink.a * h);
		}

		if (gridEvery > 0.5) {
			let g = abs(fract(gcoord - 0.5) - 0.5) / gw; // device px to the nearest grid line, per axis
			let d = min(g.x, g.y);
			let cov = clamp(0.5 * view.dpr - d + 0.5, 0.0, 1.0);
			col = inkOver(col, inkRgb, mesh.ink.a * cov);
		}

		if (bands > 0.5 && mesh.flags.w > 0.5) {
			let k = round(s);
			let dist = abs(s - k) / fws;
			var cov = clamp(0.5 * view.dpr * 1.2 - dist + 0.5, 0.0, 1.0);
			if (k < 0.0 || k > bands) {
				cov = 0.0;
			}
			col = inkOver(col, mesh.contourInk.rgb, mesh.contourInk.a * cov);
		}
		return col * a;
	}

	let base = mix(mesh.colorLo.rgb, mesh.colorHi.rgb, clamp(t, 0.0, 1.0));
	var col = base * (0.35 + 0.65 * lambert);

	if (bands > 0.5) {
		let k = round(s);
		let dist = abs(s - k) / fws;
		var cov = clamp(0.8 - dist + 0.5, 0.0, 1.0);
		if (k < 0.0 || k > bands) {
			cov = 0.0;
		}
		let line = col * 1.5 + vec3f(0.04);
		col = mix(col, line, cov);
	}
	return vec4f(col * a, a);
}

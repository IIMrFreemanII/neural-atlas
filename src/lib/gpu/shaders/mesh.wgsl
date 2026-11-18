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
};
@group(1) @binding(0) var<uniform> mesh: MeshU;

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) world: vec3f,
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
	return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	let range = mesh.params.xy;
	let bands = mesh.params.z;
	let span = max(range.y - range.x, 1e-6);
	let t = (in.world.y - range.x) / span;
	let s = t * bands;
	let fws = max(fwidth(s), 1e-6);

	var n = normalize(cross(dpdx(in.world), dpdy(in.world)));
	if (n.y < 0.0) {
		n = -n;
	}
	let light = normalize(vec3f(-0.5, 0.8, 0.35));
	let lambert = max(dot(n, light), 0.0);

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
	// Keep the surface below the bloom threshold.
	col = min(col, vec3f(0.65));
	let a = mesh.params.w;
	return vec4f(col * a, a);
}

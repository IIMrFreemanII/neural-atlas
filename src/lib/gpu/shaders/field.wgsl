struct ViewU {
	viewProj: mat4x4f,
	viewport: vec2f,
	dpr: f32,
	time: f32,
};
@group(0) @binding(0) var<uniform> view: ViewU;

struct FieldU {
	rect: vec4f,      // x0, y0, x1, y1
	colorLo: vec4f,
	colorHi: vec4f,
	isoColor: vec4f,
	params: vec4f,    // opacity, iso, isoWidth (css px), 0
	dims: vec4f,      // cols, rows, 0, 0
};
@group(1) @binding(0) var<uniform> field: FieldU;
@group(1) @binding(1) var grid: texture_2d<f32>;

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
	var corners = array<vec2f, 6>(
		vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
		vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
	);
	let uv = corners[vi];
	let w = mix(field.rect.xy, field.rect.zw, uv);
	var o: VsOut;
	o.pos = view.viewProj * vec4f(w, 0.0, 1.0);
	o.uv = uv;
	return o;
}

fn texel(c: vec2i, maxc: vec2i) -> f32 {
	return textureLoad(grid, clamp(c, vec2i(0), maxc), 0).x;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	let dims = field.dims.xy;
	let maxc = vec2i(dims) - vec2i(1);
	// Samples sit at cell centers.
	let g = clamp(in.uv * dims - vec2f(0.5), vec2f(0.0), dims - vec2f(1.0));
	let base = floor(g);
	let f = g - base;
	let c = vec2i(base);
	let v00 = texel(c, maxc);
	let v10 = texel(c + vec2i(1, 0), maxc);
	let v01 = texel(c + vec2i(0, 1), maxc);
	let v11 = texel(c + vec2i(1, 1), maxc);
	let v = mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
	let fw = max(fwidth(v), 1e-6);

	let base_c = mix(field.colorLo, field.colorHi, clamp(v, 0.0, 1.0));
	let aBase = base_c.a * field.params.x;
	var rgb = base_c.rgb * aBase;
	var a = aBase;

	let iso = field.params.y;
	if (iso >= 0.0) {
		let halfW = field.params.z * view.dpr * 0.5;
		let dist = abs(v - iso) / fw;
		let cov = clamp(halfW - dist + 0.5, 0.0, 1.0);
		let ia = field.isoColor.a * cov * field.params.x;
		rgb = rgb * (1.0 - ia) + field.isoColor.rgb * ia;
		a = a * (1.0 - ia) + ia;
	}
	return vec4f(rgb, a);
}

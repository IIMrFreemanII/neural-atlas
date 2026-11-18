struct ViewU {
	viewProj: mat4x4f,
	viewport: vec2f,
	dpr: f32,
	time: f32,
};
@group(0) @binding(0) var<uniform> view: ViewU;

struct RevealU {
	reveal: vec4f,  // x: 0..1 pen position along the arc length (1 = everything)
};
@group(1) @binding(0) var<uniform> rev: RevealU;

struct VsIn {
	@builtin(vertex_index) vi: u32,
	@location(0) a: vec4f,      // ax, ay, az, width (css px)
	@location(1) b: vec4f,      // bx, by, bz, t0
	@location(2) color: vec4f,
	@location(3) pulse: vec4f,  // speed, count, gain, t1
};

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) local: vec2f,                      // (along, across) device px, along measured from a
	@location(1) color: vec4f,
	@location(2) @interpolate(flat) geom: vec4f,    // length px, half width px, t0, t1
	@location(3) @interpolate(flat) pulse: vec3f,   // speed, count, gain
};

@vertex
fn vs(in: VsIn) -> VsOut {
	// x: 0 = at a, 1 = at b; y: side
	var corners = array<vec2f, 6>(
		vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(0.0, 1.0),
		vec2f(0.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
	);
	let corner = corners[in.vi];
	let half_vp = view.viewport * 0.5;

	let ca = view.viewProj * vec4f(in.a.xyz, 1.0);
	let cb = view.viewProj * vec4f(in.b.xyz, 1.0);
	let pa = ca.xy / ca.w * half_vp;
	let pb = cb.xy / cb.w * half_vp;

	let delta = pb - pa;
	let len = length(delta);
	var dir = vec2f(1.0, 0.0);
	if (len > 1e-4) {
		dir = delta / len;
	}
	let perp = vec2f(-dir.y, dir.x);
	let halfW = in.a.w * view.dpr * 0.5;
	let pad = halfW + 1.0;

	let atB = corner.x > 0.5;
	let base = select(pa, pb, atB);
	let along = select(-pad, len + pad, atB);
	let p = base + dir * select(-pad, pad, atB) + perp * corner.y * pad;
	let c = select(ca, cb, atB);

	var o: VsOut;
	o.pos = vec4f(p / half_vp * c.w, c.z, c.w);
	o.local = vec2f(along, corner.y * pad);
	o.color = in.color;
	o.geom = vec4f(len, halfW, in.b.w, in.pulse.w);
	o.pulse = in.pulse.xyz;
	return o;
}

fn bump(phase: f32) -> f32 {
	let x = abs(phase - 0.5) / 0.075;
	if (x >= 1.0) {
		return 0.0;
	}
	let k = 1.0 - x * x;
	return k * k;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	let len = in.geom.x;
	let halfW = in.geom.y;
	let s = in.local.x;
	let sx = clamp(s, 0.0, len);
	let dist = length(vec2f(s - sx, in.local.y)) - halfW;
	var cov = clamp(0.5 - dist, 0.0, 1.0);

	var f = 0.0;
	if (len > 1e-4) {
		f = sx / len;
	}
	let t = mix(in.geom.z, in.geom.w, f);

	let r = rev.reveal.x;
	if (r < 1.0) {
		// Soft ~1.5 device-px edge at the pen tip, measured along the segment.
		let tPerPx = (in.geom.w - in.geom.z) / max(len, 1e-4);
		let e = max(tPerPx * 1.5, 1e-6);
		let pen = clamp((r - t) / e, 0.0, 1.0);
		if (pen <= 0.0) {
			discard;
		}
		cov = cov * pen;
	}

	var mult = 1.0;
	let count = in.pulse.y;
	if (count > 0.0) {
		let phase = fract(t * count - view.time * in.pulse.x);
		mult = 1.0 + in.pulse.z * bump(phase);
	}

	let a = in.color.a * cov;
	return vec4f(in.color.rgb * mult * a, a);
}

// Particle cloud: one compute pass eases positions and colors, one render pass draws soft discs.

struct Particle {
	pos: vec2f,
	group: f32,
	seed: f32,
	color: vec4f,
};

struct SimU {
	rates: vec4f,    // dt, time, follow, colorFollow
	misc: vec4f,     // jitter, groupCount, count, teleportCount
	neutral: vec4f,
	centers: array<vec4f, 4>, // two centers per entry: (x0, y0, x1, y1)
	palette: array<vec4f, 8>,
};

@group(0) @binding(0) var<uniform> sim: SimU;
@group(0) @binding(1) var<storage, read> targets: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> state: array<Particle>;
@group(0) @binding(3) var<storage, read> teleport: array<vec2f>;

fn centerAt(i: u32) -> vec2f {
	let c = sim.centers[i / 2u];
	return select(c.xy, c.zw, (i & 1u) == 1u);
}

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) gid: vec3u) {
	let i = gid.x;
	if (f32(i) >= sim.misc.z) {
		return;
	}
	var p = state[i];
	let tgt = targets[i];
	if (f32(i) < sim.misc.w) {
		p.pos = teleport[i];
	}

	let dt = sim.rates.x;
	let time = sim.rates.y;
	let jitter = sim.misc.x;
	let drift = jitter * vec2f(sin(time * 0.7 + p.seed * 61.0), cos(time * 0.55 + p.seed * 97.0));
	let kp = 1.0 - exp(-sim.rates.z * dt);
	p.pos = p.pos + (tgt + drift - p.pos) * kp;

	let groups = u32(sim.misc.y + 0.5);
	var goal = sim.neutral;
	var g = -1.0;
	if (groups > 0u) {
		var best = 1e30;
		var bi = 0u;
		for (var c = 0u; c < groups; c = c + 1u) {
			let d = tgt - centerAt(c);
			let d2 = dot(d, d);
			if (d2 < best) {
				best = d2;
				bi = c;
			}
		}
		g = f32(bi);
		goal = sim.palette[bi];
	}
	p.group = g;
	let kc = 1.0 - exp(-sim.rates.w * dt);
	p.color = p.color + (goal - p.color) * kc;
	state[i] = p;
}

struct ViewU {
	viewProj: mat4x4f,
	viewport: vec2f,
	dpr: f32,
	time: f32,
};
@group(0) @binding(0) var<uniform> view: ViewU;

struct VsIn {
	@builtin(vertex_index) vi: u32,
	@location(0) posMisc: vec4f, // x, y, group, seed
	@location(1) color: vec4f,
};

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) local: vec2f,
	@location(1) color: vec4f,
	@location(2) @interpolate(flat) shape: vec2f, // radius dev px, softness
};

// radius (css px) and softness come from the render uniform.
struct RenderU {
	radius: f32,
	softness: f32,
	pad0: f32,
	pad1: f32,
};
@group(1) @binding(0) var<uniform> style: RenderU;

@vertex
fn vs(in: VsIn) -> VsOut {
	var corners = array<vec2f, 6>(
		vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
		vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
	);
	let corner = corners[in.vi];
	let radius = style.radius * view.dpr;
	let extent = radius + 1.0;
	let clip = view.viewProj * vec4f(in.posMisc.xy, 0.0, 1.0);
	let offsetNdc = corner * extent * 2.0 / view.viewport;

	var o: VsOut;
	o.pos = vec4f(clip.xy + offsetNdc * clip.w, clip.z, clip.w);
	o.local = corner * extent;
	o.color = in.color;
	o.shape = vec2f(radius, style.softness);
	return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	let d = length(in.local);
	let r = max(in.shape.x, 0.001);
	let crisp = clamp(r - d + 0.5, 0.0, 1.0);
	let x = clamp(d / r, 0.0, 1.0);
	let e4 = exp(-4.0);
	let glow = (exp(-4.0 * x * x) - e4) / (1.0 - e4);
	let cov = mix(crisp, glow, clamp(in.shape.y, 0.0, 1.0));
	let a = in.color.a * cov;
	return vec4f(in.color.rgb * a, a);
}

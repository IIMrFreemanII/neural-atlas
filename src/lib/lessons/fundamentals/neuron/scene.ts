import { Camera2D } from '../../../gpu/camera.ts';
import { paperFill, tone } from '../../../gpu/colors.ts';
import { Lines, bezier } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { ACTIVATION_NAMES, activate, type ActivationId } from '../../../ml/activations.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { Pulses, cumulativeLengths, pointAtFraction, type PulsePath } from '../../shared/pulses.ts';

const WORLD: WorldRect = { x0: -2.1, x1: 3.4, y0: -1.1, y1: 1.5 };
/** World shift that centres the diagram (x from -2.1 to 1.8) while the activation plot is hidden. */
const CENTER_SHIFT = 0.8;
const IN_X = -1.8;
const NEURON = { x: 0, y: 0 };
const BIAS = { x: 0, y: 1.0 };
const OUT = { x: 1.5, y: 0 };
const BOX = { x0: 2.1, x1: 3.3, y0: -0.6, y1: 0.6 };
const Z_RANGE = 4;
const CURVE_SEGS = 48;
const DIM = 0.3;
/** Smoothstep progress along the input edges where the weight labels are anchored (0.5 = middle). */
const WS = 0.5;
/** A param must hold still this long (seconds) before the signal wave runs: slider released, step tween done. */
const SETTLE = 0.15;
/** The wave: inputs travel this part of their edge, then the output edge does. */
const IN_SPAN = [0.15, 0.78] as const;
const OUT_SPAN = [0.26, 0.84] as const;
const WAVE_IN = 0.55;
const WAVE_OUT = 0.45;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const fmt = (v: number) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
const round2 = (v: number) => Math.round(v * 100) / 100 || 0;
const sign = (v: number, pos: Tone): Tone => (v < 0 ? 'neg' : pos);

type Pt = [number, number];

/** The curve an input edge follows into the neuron (also what the signal pulses travel). */
function inputCurve(from: { x: number; y: number }): Pt[] {
	const dx = NEURON.x - from.x;
	return bezier(
		[from.x, from.y],
		[from.x + dx * 0.5, from.y],
		[NEURON.x - dx * 0.5, NEURON.y],
		[NEURON.x, NEURON.y],
		24
	);
}

/** The part of a polyline between two arc-length fractions. */
function span(pts: Pt[], [f0, f1]: readonly [number, number]): Pt[] {
	const cum = cumulativeLengths(pts);
	const out: Pt[] = [];
	for (let k = 0; k <= 10; k++) {
		const q = pointAtFraction(pts, cum, f0 + ((f1 - f0) * k) / 10);
		out.push([q[0], q[1]]);
	}
	return out;
}

export const createNeuronScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let edges: Lines;
	let nodes: Points;
	let plotLines: Lines;
	let dots: Points;
	let pulses: Pulses;

	let p: Record<string, number> = {};
	let hl: Tone | null = null;
	let dirty = true;
	let lastW = -1;
	let lastH = -1;
	let lastZ = NaN;
	let lastOut = NaN;
	let paramsSeen = false;
	let settleT = 0;
	let settling = false;

	const dim = (t: Tone) => (hl === null || hl === t ? 1 : DIM);

	function rebuild() {
		const v = ctx.view;
				const showX2 = clamp(p.showX2 ?? 0, 0, 1);
		const showBias = clamp(p.showBias ?? 0, 0, 1);
		const showAct = clamp(p.showAct ?? 0, 0, 1);
		const act = Math.round(clamp(p.act ?? 0, 0, 2)) as ActivationId;
		const x1 = p.x1 ?? 0;
		const x2 = p.x2 ?? 0;
		const w1 = p.w1 ?? 0;
		const w2 = p.w2 ?? 0;
		const b = (p.b ?? 0) * showBias;

		const z = w1 * x1 + w2 * x2 * showX2 + b;
		const out = lerp(z, activate(act, z), showAct);

		const y1 = lerp(0, 0.6, showX2);
		const y2 = -0.6;

		const labels: Label[] = [];
		const label = (
			id: string,
			x: number,
			y: number,
			dx: number,
			dy: number,
			text: string,
			opts: { tone?: Tone; mono?: boolean; align?: 'left' | 'center' | 'right'; opacity: number }
		) => {
			const opacity = opts.opacity * dim(opts.tone ?? 'text');
			if (opacity < 0.02) return;
			const [px, py] = v.project(x, y);
			labels.push({
				id,
				x: px + dx,
				y: py + dy,
				text,
				tone: opts.tone,
				mono: opts.mono,
				align: opts.align,
				opacity
			});
		};

		nodes.count = 0;
		edges.count = 0;
		plotLines.count = 0;
		dots.count = 0;
		let ni = 0;

		// A value node: a paper disc hides the edges behind it; the ring carries tone and value.
		const valueNode = (x: number, y: number, r: number, value: number, opacity: number) => {
			if (opacity < 0.02) return;
			const t = sign(value, 'data');
			const f = opacity * dim(t);
			const k = Math.min(Math.abs(value), 1.5) / 1.5;
			nodes.set(ni++, { x, y, radius: r, color: paperFill(opacity) });
			nodes.set(ni++, { x, y, radius: r, ring: 1.5 + k, color: tone(t, (0.55 + 0.45 * k) * f) });
		};

		// ---- edges ----
		let n = 0;
		const weightEdge = (from: { x: number; y: number }, w: number, opacity: number) => {
			if (opacity < 0.02) return;
			const t = sign(w, 'param');
			const f = opacity * dim(t);
			const dx = NEURON.x - from.x;
			const pts = inputCurve(from);
			n = edges.setPolyline(n, pts, {
				width: clamp(1 + 3 * Math.abs(w), 1, 7),
				color: tone(t, 0.75 * f)
			});
		};
		weightEdge({ x: IN_X, y: y1 }, w1, 1);
		weightEdge({ x: IN_X, y: y2 }, w2, showX2);

		if (showBias >= 0.02) {
			const t = sign(b, 'param');
			const f = showBias * dim(t);
			edges.set(n++, {
				a: [BIAS.x, BIAS.y],
				b: [NEURON.x, NEURON.y],
				width: clamp(1 + 3 * Math.abs(b), 1, 7),
				color: tone(t, 0.75 * f)
			});
		}

		{
			const t = sign(z, 'data');
			const f = dim(t);
			edges.set(n++, {
				a: [NEURON.x, NEURON.y],
				b: [OUT.x, OUT.y],
				width: 3,
				color: tone(t, 0.75 * f)
			});
		}
		edges.count = n;

		// ---- nodes ----
		valueNode(IN_X, y1, 12, x1, 1);
		valueNode(IN_X, y2, 12, x2 * showX2, showX2);
		valueNode(NEURON.x, NEURON.y, 24, z, 1);
		valueNode(OUT.x, OUT.y, 14, out, 1);
		if (showBias >= 0.02) {
			const t = sign(b, 'param');
			const f = showBias * dim(t);
			const k = Math.min(Math.abs(b), 1.5) / 1.5;
			nodes.set(ni++, { x: BIAS.x, y: BIAS.y, radius: 8, color: paperFill(showBias) });
			nodes.set(ni++, {
				x: BIAS.x,
				y: BIAS.y,
				radius: 8,
				ring: 1.5 + k,
				color: tone(t, (0.55 + 0.45 * k) * f)
			});
		}
		nodes.count = ni;

		// ---- labels ----
		const dataT = (value: number) => sign(value, 'data');
		label('n-x1', IN_X, y1, 0, -26, 'x₁', { opacity: 1 });
		label('v-x1', IN_X, y1, 0, 26, fmt(x1), { tone: dataT(x1), mono: true, opacity: 1 });
		label('n-x2', IN_X, y2, 0, -26, 'x₂', { opacity: showX2 });
		label('v-x2', IN_X, y2, 0, 26, fmt(x2 * showX2), {
			tone: dataT(x2 * showX2),
			mono: true,
			opacity: showX2
		});
		label('n-sum', NEURON.x, NEURON.y, 0, 0, 'Σ', { opacity: 1 });
		label('v-sum', NEURON.x, NEURON.y, 0, 40, fmt(z), { tone: dataT(z), mono: true, opacity: 1 });
		label('n-out', OUT.x, OUT.y, 0, -28, 'output', { opacity: 1 });
		label('v-out', OUT.x, OUT.y, 0, 28, fmt(out), { tone: dataT(out), mono: true, opacity: 1 });
		label('w1', IN_X + 1.8 * WS, y1 * (1 - WS), 30, -18, `w₁ ${fmt(w1)}`, {
			tone: sign(w1, 'param'),
			mono: true,
			opacity: 1
		});
		label('w2', IN_X + 1.8 * WS, y2 * (1 - WS), 30, 18, `w₂ ${fmt(w2)}`, {
			tone: sign(w2, 'param'),
			mono: true,
			opacity: showX2
		});
		label('n-bias', BIAS.x, BIAS.y, 0, -20, 'bias', { opacity: showBias });
		label('v-bias', BIAS.x, BIAS.y, 16, 0, `b ${fmt(b)}`, {
			tone: sign(b, 'param'),
			mono: true,
			align: 'left',
			opacity: showBias
		});

		// ---- activation plot ----
		if (showAct >= 0.02) {
			const [vlo, vhi] = act === 1 ? [-0.1, 1.1] : [-Z_RANGE, Z_RANGE];
			const px = (zz: number) =>
				BOX.x0 + ((clamp(zz, -Z_RANGE, Z_RANGE) + Z_RANGE) / (2 * Z_RANGE)) * (BOX.x1 - BOX.x0);
			const py = (vv: number) => BOX.y0 + clamp((vv - vlo) / (vhi - vlo), 0, 1) * (BOX.y1 - BOX.y0);
			const fm = showAct * dim('muted');
			const axis = tone('muted', 0.5 * fm);
			plotLines.set(0, { a: [BOX.x0, py(0)], b: [BOX.x1, py(0)], width: 1, color: axis });
			plotLines.set(1, { a: [px(0), BOX.y0], b: [px(0), BOX.y1], width: 1, color: axis });
			const fd = showAct * dim('data');
			const curve: Array<[number, number]> = [];
			for (let i = 0; i <= CURVE_SEGS; i++) {
				const zz = -Z_RANGE + (2 * Z_RANGE * i) / CURVE_SEGS;
				curve.push([px(zz), py(activate(act, zz))]);
			}
			let m = plotLines.setPolyline(2, curve, { width: 2, color: tone('data', 0.9 * fd) });
			const zt = sign(z, 'data');
			const fz = showAct * dim(zt);
			plotLines.set(m++, {
				a: [px(z), BOX.y0],
				b: [px(z), BOX.y1],
				width: 1,
				color: tone(zt, 0.4 * fz)
			});
			plotLines.count = m;

			const ay = activate(act, z);
			const at = sign(ay, 'data');
			const fa = showAct * dim(at);
			// A solid ink dot with a paper ring so it reads on top of the curve.
			dots.set(0, { x: px(z), y: py(ay), radius: 6.5, color: paperFill(showAct) });
			dots.set(1, { x: px(z), y: py(ay), radius: 4.5, color: tone(at, fa) });
			dots.count = 2;

			const cx = (BOX.x0 + BOX.x1) / 2;
			label('n-act', cx, BOX.y1, 0, -16, ACTIVATION_NAMES[act], { opacity: showAct });
			label('p-z', cx, BOX.y0, 0, 18, `z ${fmt(z)}`, {
				tone: dataT(z),
				mono: true,
				opacity: showAct
			});
			label('p-out', cx, BOX.y0, 0, 38, `out ${fmt(out)}`, {
				tone: dataT(out),
				mono: true,
				opacity: showAct
			});
		}

		edges.upload();
		nodes.upload();
		plotLines.upload();
		dots.upload();
		ctx.labels.set(labels);

		const rz = round2(z);
		const ro = round2(out);
		if (rz !== lastZ) {
			lastZ = rz;
			ctx.emit('z', rz);
		}
		if (ro !== lastOut) {
			lastOut = ro;
			ctx.emit('out', ro);
		}
	}

	/** The inputs' values travel to the neuron, then the output travels on: a computation, shown once. */
	function wave() {
		const showX2 = clamp(p.showX2 ?? 0, 0, 1);
		const showBias = clamp(p.showBias ?? 0, 0, 1);
		const t = (v: number): Tone => sign(v, 'data');
		const w1 = p.w1 ?? 0;
		const w2 = p.w2 ?? 0;
		const b = (p.b ?? 0) * showBias;
		const x1 = p.x1 ?? 0;
		const x2 = p.x2 ?? 0;
		const inputs: PulsePath[] = [
			{ points: span(inputCurve({ x: IN_X, y: lerp(0, 0.6, showX2) }), IN_SPAN), tone: t(x1 * w1) }
		];
		if (showX2 > 0.5) inputs.push({ points: span(inputCurve({ x: IN_X, y: -0.6 }), IN_SPAN), tone: t(x2 * w2) });
		if (showBias > 0.5)
			inputs.push({ points: span([[BIAS.x, BIAS.y], [NEURON.x, NEURON.y]], IN_SPAN), tone: t(b) });
		const z = w1 * x1 + w2 * x2 * showX2 + b;
		const act = Math.round(clamp(p.act ?? 0, 0, 2)) as ActivationId;
		const out = lerp(z, activate(act, z), clamp(p.showAct ?? 0, 0, 1));
		pulses.fire(inputs, WAVE_IN, () => {
			pulses.fire(
				[{ points: span([[NEURON.x, NEURON.y], [OUT.x, OUT.y]], OUT_SPAN), tone: t(out) }],
				WAVE_OUT
			);
		});
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			edges = new Lines(c, { capacity: 128, blend: 'alpha' });
			nodes = new Points(c, { capacity: 16, blend: 'alpha' });
			plotLines = new Lines(c, { capacity: 96, blend: 'alpha' });
			dots = new Points(c, { capacity: 4, blend: 'alpha' });
			pulses = new Pulses(c, { capacity: 4 });
		},
		setParams(params) {
			const changed = Object.keys(params).some((k) => Math.abs(params[k] - (p[k] ?? 0)) > 1e-9);
			p = { ...params };
			dirty = true;
			// Any change (a step's tween, a slider drag, a new activation) restarts the settle timer.
			if (paramsSeen && changed) {
				settling = true;
				settleT = 0;
			}
			paramsSeen = true;
		},
		highlight(t) {
			if (t !== hl) {
				hl = t;
				dirty = true;
			}
		},
		update(frame) {
			const v = ctx.view;
			// showAct is tweened with the other params, so the diagram glides from centred to the plot layout.
			const shift = CENTER_SHIFT * (1 - clamp(p.showAct ?? 0, 0, 1));
			fitCamera(camera, v, { ...WORLD, x0: WORLD.x0 - shift, x1: WORLD.x1 - shift }, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));
			if (v.width !== lastW || v.height !== lastH) {
				lastW = v.width;
				lastH = v.height;
				dirty = true;
			}
			if (dirty) {
				dirty = false;
				rebuild();
			}
			if (settling) {
				settleT += frame.dt;
				if (settleT >= SETTLE) {
					settling = false;
					if (!ctx.reducedMotion) wave();
				}
			}
			pulses.update(frame.dt);
		},
		draw(pass) {
			edges.draw(pass);
			nodes.draw(pass);
			plotLines.draw(pass);
			dots.draw(pass);
			pulses.draw(pass);
		},
		dispose() {
			edges.destroy();
			nodes.destroy();
			plotLines.destroy();
			dots.destroy();
			pulses.dispose();
		}
	};
	return scene;
};

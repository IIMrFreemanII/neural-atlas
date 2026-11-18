import { Camera2D } from '../../../gpu/camera.ts';
import { VOID, tone } from '../../../gpu/colors.ts';
import { Lines, bezier } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { ACTIVATION_NAMES, activate, type ActivationId } from '../../../ml/activations.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';

const WORLD: WorldRect = { x0: -2.1, x1: 3.4, y0: -1.1, y1: 1.5 };
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

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const fmt = (v: number) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
const round2 = (v: number) => Math.round(v * 100) / 100 || 0;
const sign = (v: number, pos: Tone): Tone => (v < 0 ? 'neg' : pos);

export const createNeuronScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let edges: Lines;
	let nodes: Points;
	let glows: Points;
	let plotLines: Lines;
	let dots: Points;

	let p: Record<string, number> = {};
	let hl: Tone | null = null;
	let dirty = true;
	let lastW = -1;
	let lastH = -1;
	let lastZ = NaN;
	let lastOut = NaN;

	const dim = (t: Tone) => (hl === null || hl === t ? 1 : DIM);

	function rebuild() {
		const v = ctx.view;
		const motion = ctx.reducedMotion ? 0 : 1;
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
		glows.count = 0;
		edges.count = 0;
		plotLines.count = 0;
		dots.count = 0;
		let ni = 0;
		let gi = 0;

		// A value node: dark disc + crisp ring + additive glow scaled by the value.
		const valueNode = (x: number, y: number, r: number, value: number, opacity: number) => {
			if (opacity < 0.02) return;
			const t = sign(value, 'data');
			const f = opacity * dim(t);
			const k = Math.min(Math.abs(value), 1.5) / 1.5;
			nodes.set(ni++, { x, y, radius: r, color: [VOID[0], VOID[1], VOID[2], opacity] });
			nodes.set(ni++, { x, y, radius: r, ring: 2, color: tone(t, 1.3 * f, f) });
			glows.set(gi++, {
				x,
				y,
				radius: r * 3.2,
				softness: 1,
				color: tone(t, (0.15 + 2.1 * k) * f, (0.1 + 0.6 * k) * f)
			});
		};

		// ---- edges ----
		let n = 0;
		const weightEdge = (
			from: { x: number; y: number },
			w: number,
			x: number,
			opacity: number
		) => {
			if (opacity < 0.02) return;
			const t = sign(w, 'param');
			const f = opacity * dim(t);
			const dx = NEURON.x - from.x;
			const pts = bezier(
				[from.x, from.y],
				[from.x + dx * 0.5, from.y],
				[NEURON.x - dx * 0.5, NEURON.y],
				[NEURON.x, NEURON.y],
				24
			);
			n = edges.setPolyline(n, pts, {
				width: clamp(1 + 3 * Math.abs(w), 1, 7),
				color: tone(t, 1.2 * f, 0.7 * f),
				pulseSpeed: 0.3 * motion,
				pulseCount: 2,
				pulseGain: 3 * Math.min(1, Math.abs(x * w)) * motion
			});
		};
		weightEdge({ x: IN_X, y: y1 }, w1, x1, 1);
		weightEdge({ x: IN_X, y: y2 }, w2, x2 * showX2, showX2);

		if (showBias >= 0.02) {
			const t = sign(b, 'param');
			const f = showBias * dim(t);
			edges.set(n++, {
				a: [BIAS.x, BIAS.y],
				b: [NEURON.x, NEURON.y],
				width: clamp(1 + 3 * Math.abs(b), 1, 7),
				color: tone(t, 1.2 * f, 0.7 * f),
				pulseSpeed: 0.3 * motion,
				pulseCount: 1,
				pulseGain: 3 * Math.min(1, Math.abs(b)) * motion
			});
		}

		{
			const t = sign(z, 'data');
			const f = dim(t);
			edges.set(n++, {
				a: [NEURON.x, NEURON.y],
				b: [OUT.x, OUT.y],
				width: 3,
				color: tone(t, 1.2 * f, 0.7 * f),
				pulseSpeed: 0.3 * motion,
				pulseCount: 2,
				pulseGain: 3 * Math.min(1, Math.abs(out)) * motion
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
			nodes.set(ni++, { x: BIAS.x, y: BIAS.y, radius: 8, color: [VOID[0], VOID[1], VOID[2], showBias] });
			nodes.set(ni++, { x: BIAS.x, y: BIAS.y, radius: 8, ring: 2, color: tone(t, 1.3 * f, f) });
			glows.set(gi++, {
				x: BIAS.x,
				y: BIAS.y,
				radius: 26,
				softness: 1,
				color: tone(t, (0.15 + 2.1 * k) * f, (0.1 + 0.6 * k) * f)
			});
		}
		nodes.count = ni;
		glows.count = gi;

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
			const axis = tone('muted', 1.2 * fm, 0.7 * fm);
			plotLines.set(0, { a: [BOX.x0, py(0)], b: [BOX.x1, py(0)], width: 1, color: axis });
			plotLines.set(1, { a: [px(0), BOX.y0], b: [px(0), BOX.y1], width: 1, color: axis });
			const fd = showAct * dim('data');
			const curve: Array<[number, number]> = [];
			for (let i = 0; i <= CURVE_SEGS; i++) {
				const zz = -Z_RANGE + (2 * Z_RANGE * i) / CURVE_SEGS;
				curve.push([px(zz), py(activate(act, zz))]);
			}
			let m = plotLines.setPolyline(2, curve, { width: 2, color: tone('data', 1.4 * fd, 0.9 * fd) });
			const zt = sign(z, 'data');
			const fz = showAct * dim(zt);
			plotLines.set(m++, {
				a: [px(z), BOX.y0],
				b: [px(z), BOX.y1],
				width: 1,
				color: tone(zt, 1.2 * fz, 0.4 * fz)
			});
			plotLines.count = m;

			const ay = activate(act, z);
			const at = sign(ay, 'data');
			const fa = showAct * dim(at);
			dots.set(0, {
				x: px(z),
				y: py(ay),
				radius: 18,
				softness: 1,
				color: tone(at, 2 * fa, 0.7 * fa)
			});
			dots.set(1, { x: px(z), y: py(ay), radius: 4.5, color: tone(at, 3 * fa, fa) });
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
		glows.upload();
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

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			edges = new Lines(c, { capacity: 128, blend: 'alpha' });
			nodes = new Points(c, { capacity: 16, blend: 'alpha' });
			glows = new Points(c, { capacity: 16, blend: 'additive' });
			plotLines = new Lines(c, { capacity: 96, blend: 'alpha' });
			dots = new Points(c, { capacity: 4, blend: 'additive' });
		},
		setParams(params) {
			p = { ...params };
			dirty = true;
		},
		highlight(t) {
			if (t !== hl) {
				hl = t;
				dirty = true;
			}
		},
		update() {
			const v = ctx.view;
			fitCamera(camera, v, WORLD, lessonInsets(v));
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
		},
		draw(pass) {
			edges.draw(pass);
			nodes.draw(pass);
			glows.draw(pass);
			plotLines.draw(pass);
			dots.draw(pass);
		},
		dispose() {
			edges.destroy();
			nodes.destroy();
			glows.destroy();
			plotLines.destroy();
			dots.destroy();
		}
	};
	return scene;
};

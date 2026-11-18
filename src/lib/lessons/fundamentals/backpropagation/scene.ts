import { Camera2D } from '../../../gpu/camera.ts';
import { tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { applyGrads, backward, cloneMLP, createMLP, forward, type MLP } from '../../../ml/mlp.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { NetworkDiagram } from '../../shared/network-diagram.ts';

const SIZES = [2, 2, 1];
const SEED = 17;
const WORLD: WorldRect = { x0: -1.8, x1: 2.8, y0: -1.2, y1: 1.2 };
const NODE_R = 15;
/** World x of the target scale, to the right of the output node. */
const TARGET_X = 2.05;
/** World units of bar length per unit of error. */
const ERR_SCALE = 1.0;
const RUN_RATE = 6;
const DONE_LOSS = 0.001;
const WEIGHT_VIS = 0.6;
const DIM = 0.35;

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const sgn = (v: number, s: string) => (v < 0 && Number(s) !== 0 ? '−' : '') + s.replace('-', '');
const fmt2 = (v: number) => sgn(v, v.toFixed(2));
/** Two significant-ish digits: small numbers get three decimals. */
const fmtSig = (v: number) => sgn(v, Math.abs(v) >= 0.095 ? v.toFixed(2) : v.toFixed(3));
const fmtPlus = (v: number) => (v >= 0 && Number(Math.abs(v).toFixed(3)) !== 0 ? '+' : '') + fmtSig(v);

/** Point on the diagram's edge curve (cubic bezier with horizontal tangents) at parameter t. */
function edgePoint(a: [number, number], b: [number, number], t: number): [number, number] {
	const mid = (b[0] - a[0]) * 0.5;
	const p = [a, [a[0] + mid, a[1]], [b[0] - mid, b[1]], b] as const;
	const u = 1 - t;
	const k = [u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t];
	return [
		k[0] * p[0][0] + k[1] * p[1][0] + k[2] * p[2][0] + k[3] * p[3][0],
		k[0] * p[0][1] + k[1] * p[1][1] + k[2] * p[2][1] + k[3] * p[3][1]
	];
}

export const createBackpropScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let diagram: NetworkDiagram;
	let lines: Lines;
	let glows: Points;
	let dots: Points;

	const initial: MLP = createMLP(SIZES, 'tanh', 'sigmoid', SEED);
	let net: MLP = cloneMLP(initial);

	// params
	let x1 = 0.8;
	let x2 = -0.4;
	let target = 1;
	let lr = 0.5;
	let phase = 0;
	let hl: Tone | null = null;

	// simulation
	let running = false;
	let acc = 0;
	let updates = 0;

	// eased visibility values (0..1) and emphasis levels
	const vis = { marker: 0, haloOut: 0, haloHid: 0, edgeLabel: 0 };
	const emph = { ih: 1, e0: 1, e1: 1 };

	let lastPred = NaN;
	let lastTarget = NaN;
	let lastLoss = NaN;
	let lastUpdates = -1;

	function reset() {
		net = cloneMLP(initial);
		running = false;
		acc = 0;
		updates = 0;
	}

	function updateOnce() {
		const g = backward(net, forward(net, [x1, x2]), [target], 'mse');
		applyGrads(net, g, lr);
		updates++;
	}

	const mulTone = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			diagram = new NetworkDiagram(c, SIZES, { nodeRadius: NODE_R });
			lines = new Lines(c, { capacity: 16, blend: 'alpha' });
			glows = new Points(c, { capacity: 8, blend: 'additive' });
			dots = new Points(c, { capacity: 16, blend: 'alpha' });
		},
		setParams(p) {
			x1 = p.x1 ?? x1;
			x2 = p.x2 ?? x2;
			target = (p.target ?? target) >= 0.5 ? 1 : 0;
			lr = p.lr ?? lr;
			const next = Math.round(p.phase ?? phase);
			if (next < phase) reset();
			phase = next;
		},
		update(frame) {
			const v = ctx.view;
			fitCamera(camera, v, WORLD, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));

			if (running && frame.dt > 0) {
				acc += frame.dt * RUN_RATE;
				let n = Math.floor(acc);
				acc -= n;
				if (n > 5) {
					n = 5;
					acc = 0;
				}
				for (let i = 0; i < n && running; i++) {
					updateOnce();
					const l = backward(net, forward(net, [x1, x2]), [target], 'mse').loss;
					if (l < DONE_LOSS) running = false;
				}
			}

			// ease visibility
			const k = frame.dt > 0 ? 1 - Math.exp(-frame.dt * 9) : 0;
			const ease = (cur: number, goal: number) => cur + (goal - cur) * k;
			vis.marker = ease(vis.marker, phase >= 1 ? 1 : 0);
			vis.haloOut = ease(vis.haloOut, phase >= 2 ? 1 : 0);
			vis.haloHid = ease(vis.haloHid, phase >= 3 ? 1 : 0);
			vis.edgeLabel = ease(vis.edgeLabel, phase >= 4 ? 1 : 0);
			emph.ih = ease(emph.ih, phase === 2 ? DIM : 1);
			emph.e0 = ease(emph.e0, phase === 2 || phase === 3 ? DIM : 1);
			emph.e1 = ease(emph.e1, phase === 2 ? DIM : 1);

			// recompute everything from the current params and working copy
			const tr = forward(net, [x1, x2]);
			const g = backward(net, tr, [target], 'mse');
			const pred = tr.a[2][0];
			const dOut = g.delta[1][0];
			const dHid = [g.delta[0][0], g.delta[0][1]];

			// ---- nodes
			const nodeMul = hl === null || hl === 'data' ? 1 : 0.3;
			for (let l = 0; l < SIZES.length; l++) {
				for (let i = 0; i < SIZES[l]; i++) {
					const s = diagram.node(l, i);
					s.value = tr.a[l][i];
					s.label = fmt2(tr.a[l][i]);
					s.emphasis = (l === 2 ? 1 : emph.ih) * nodeMul;
					if (l === 0) s.name = i === 0 ? 'x₁' : 'x₂';
					if (l === 2) s.name = 'ŷ';
				}
			}

			// ---- edges
			let maxG = 0.01;
			for (let l = 0; l < 2; l++) for (const x of g.W[l]) maxG = Math.max(maxG, Math.abs(x));
			let maxBack = 1e-6;
			for (let i = 0; i < 2; i++) maxBack = Math.max(maxBack, Math.abs(net.W[1][i] * dOut));

			const edgeLabels: Label[] = [];
			for (let l = 0; l < 2; l++) {
				for (let i = 0; i < SIZES[l]; i++) {
					for (let j = 0; j < SIZES[l + 1]; j++) {
						const idx = j * SIZES[l] + i;
						const w = net.W[l][idx];
						const gr = g.W[l][idx];
						const e = diagram.edge(l, i, j);
						e.label = undefined;
						e.reverse = false;
						e.tone = undefined;
						e.weight = w * WEIGHT_VIS;
						e.signal = Math.abs(tr.a[l][i] * w);
						let showsGrad = false;
						if (phase === 3 && l === 1) {
							showsGrad = true;
							e.signal = Math.abs(w * dOut) / maxBack;
						} else if (phase === 4) {
							showsGrad = true;
							e.signal = Math.abs(gr) / maxG;
						}
						if (showsGrad) {
							e.tone = 'grad';
							e.reverse = true;
							if (phase === 4) e.weight = (gr / maxG) * 2;
						}
						const levelEmph = l === 0 ? emph.e0 : emph.e1;
						const m = showsGrad ? mulTone('grad') : mulTone('param');
						e.emphasis = levelEmph * m;

						if (phase >= 4 && vis.edgeLabel > 0.02) {
							const a = diagram.nodePos(l, i);
							const b = diagram.nodePos(l + 1, j);
							const [wx, wy] = edgePoint(a, b, l === 0 ? 0.4 : 0.5);
							const [px, py] = v.project(wx, wy);
							const text = phase === 4 ? `∂L/∂w ${fmtSig(gr)}` : `−η·∂L/∂w = ${fmtPlus(-lr * gr)}`;
							edgeLabels.push({
								id: `ew-${l}.${i}.${j}`,
								x: px,
								y: py - 13,
								text,
								tone: 'grad',
								mono: true,
								opacity: vis.edgeLabel * mulTone('grad')
							});
						}
					}
				}
			}
			diagram.commit();

			// ---- overlay primitives
			lines.count = 0;
			glows.count = 0;
			dots.count = 0;
			let nl = 0;
			let nd = 0;
			let ng = 0;
			const labels: Label[] = [...diagram.labels(v), ...edgeLabels];

			const [ox, oy] = diagram.nodePos(2, 0);
			const [opx, opy] = v.project(ox, oy);
			const wpp = 1 / Math.max(1e-6, Math.abs(v.project(1, 0)[0] - v.project(0, 0)[0]));

			// target marker and error bar
			if (vis.marker > 0.02) {
				const mk = vis.marker;
				const ty = oy + (target - pred) * ERR_SCALE;
				const mOk = mk * mulTone('ok');
				const mGrad = mk * mulTone('grad');
				const mData = mk * mulTone('data');
				const xt = TARGET_X;
				lines.set(nl++, {
					a: [ox + (NODE_R + 4) * wpp, oy],
					b: [xt, oy],
					width: 1,
					color: tone('muted', 1, 0.45 * mk)
				});
				lines.set(nl++, {
					a: [xt, oy],
					b: [xt, ty],
					width: 4,
					color: tone('grad', 1.8, 0.95 * mGrad)
				});
				dots.set(nd++, { x: xt, y: oy, radius: 4.5, color: tone('data', 2, mData) });
				glows.set(ng++, { x: xt, y: ty, radius: 34, softness: 1, color: tone('ok', 1.5, 0.45 * mOk) });
				dots.set(nd++, { x: xt, y: ty, radius: 9, ring: 2.2, color: tone('ok', 1.8, mOk) });

				const [tpx, tpy] = v.project(xt, ty);
				const above = target >= pred;
				labels.push({
					id: 'bp-target',
					x: tpx,
					y: tpy + (above ? -28 : 28),
					text: `target ${target.toFixed(2)}`,
					tone: 'ok',
					mono: true,
					opacity: mOk
				});
				const [, mpy] = v.project(xt, (oy + ty) / 2);
				labels.push({
					id: 'bp-loss',
					x: tpx + 20,
					y: mpy,
					text: `loss ${g.loss.toFixed(3)}`,
					tone: 'grad',
					mono: true,
					align: 'left',
					opacity: mGrad
				});
			}

			// halos
			const haloAt = (x: number, y: number, delta: number, a: number) => {
				if (a <= 0.02) return;
				const r = NODE_R + 4 + 9 * clamp(Math.abs(delta) / 0.3, 0, 1);
				glows.set(ng++, { x, y, radius: r * 2, softness: 1, color: tone('grad', 1.2, 0.3 * a) });
				dots.set(nd++, { x, y, radius: r, ring: 2.4, color: tone('grad', 2, 0.9 * a) });
			};
			const gOut = vis.haloOut * mulTone('grad');
			const gHid = vis.haloHid * mulTone('grad');
			haloAt(ox, oy, dOut, gOut);
			if (gOut > 0.02) {
				labels.push({
					id: 'bp-d-out',
					x: opx,
					y: opy + NODE_R + 40,
					text: `δ ${fmtSig(dOut)}`,
					tone: 'grad',
					mono: true,
					opacity: gOut
				});
			}
			for (let j = 0; j < 2; j++) {
				const [hx, hy] = diagram.nodePos(1, j);
				haloAt(hx, hy, dHid[j], gHid);
				if (gHid > 0.02) {
					const [hpx, hpy] = v.project(hx, hy);
					labels.push({
						id: `bp-d-h${j}`,
						x: hpx,
						y: hpy - NODE_R - 30,
						text: `δ ${fmtSig(dHid[j])}`,
						tone: 'grad',
						mono: true,
						opacity: gHid * emph.ih
					});
				}
			}

			lines.count = nl;
			dots.count = nd;
			glows.count = ng;
			lines.upload();
			dots.upload();
			glows.upload();
			ctx.labels.set(labels);

			if (pred !== lastPred) {
				lastPred = pred;
				ctx.emit('pred', pred);
			}
			if (target !== lastTarget) {
				lastTarget = target;
				ctx.emit('target', target);
			}
			if (g.loss !== lastLoss) {
				lastLoss = g.loss;
				ctx.emit('loss', g.loss);
			}
			if (updates !== lastUpdates) {
				lastUpdates = updates;
				ctx.emit('updates', updates);
			}
		},
		draw(pass) {
			diagram.draw(pass);
			lines.draw(pass);
			glows.draw(pass);
			dots.draw(pass);
		},
		command(id) {
			if (id === 'update') {
				updateOnce();
			} else if (id === 'run') {
				if (running) running = false;
				else {
					running = true;
					acc = 0;
				}
			} else if (id === 'reset') {
				reset();
			}
		},
		highlight(t) {
			hl = t;
		},
		dispose() {
			diagram.destroy();
			lines.destroy();
			glows.destroy();
			dots.destroy();
		}
	};
	return scene;
};

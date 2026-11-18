import { Camera2D } from '../../../gpu/camera.ts';
import { diverging, sheetFill, tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Quads } from '../../../gpu/primitives/quads.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { forward } from '../../../ml/mlp.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { NetworkDiagram } from '../../shared/network-diagram.ts';
import { SIZES, buildNet } from './network.ts';

const NET: WorldRect = { x0: -1.5, x1: 1.5, y0: -1.0, y1: 1.0 };
const NET_STACKED: WorldRect = { x0: -1.5, x1: 1.5, y0: -0.8, y1: 0.8 };
const WORLD_Y0 = -1.45;
const WORLD_Y1 = 1.3;
const WORLD_Y0_PANEL = -3.1;
const WORLD_Y1_PANEL = 1.05;
const WORLD_X_HIDDEN = 1.7;
const WORLD_X_PANEL = 2.35;
const PANEL_Y = -2.3;
const SEND_SECONDS = 2.4;
const DIM = 0.3;
const WEIGHT_SCALE = 0.6;

// Matrix panel geometry (world units).
const CELL = 0.42;
const PITCH = 0.48;
const X_IN = -2.0;
const X_TIMES = -1.6;
const X_W0 = -1.2;
const X_PLUS = 0.2;
const X_B = 0.6;
const ARROW = { x0: 1.0, x1: 1.45 };
const X_OUT = 1.85;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const fmt = (v: number) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2)).replace('-', '−');
const round2 = (v: number) => Math.round(v * 100) / 100 || 0;
const paren = (v: number) => (v < 0 && fmt(v) !== '0.00' ? `(${fmt(v)})` : fmt(v));

type Kind = 'data' | 'param';

export const createForwardPassScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let diagram: NetworkDiagram;
	let quads: Quads;
	let arrow: Lines;
	const net = buildNet();
	const last = SIZES.length - 1;

	let p: Record<string, number> = {};
	let hl: Tone | null = null;
	let lastParamWave = NaN;
	let holdWave: number | null = null;
	let sendT: number | null = null;
	let lastOut = NaN;

	/** Dim factor for hover highlighting. */
	const dimFor = (kind: Kind, v: number) => {
		if (hl === null) return 1;
		if (hl === 'neg') return v < 0 ? 1 : DIM;
		if (hl === 'param') return kind === 'param' ? 1 : DIM;
		if (hl === 'data') return kind === 'data' && v >= 0 ? 1 : DIM;
		return 1;
	};

	function rebuild(dt: number) {
		const v = ctx.view;
		if (sendT !== null) {
			sendT += dt;
			const u = (3 * sendT) / SEND_SECONDS;
			// Reduced motion: the wave lights up one whole layer at a time instead of sweeping.
			holdWave = Math.min(3, ctx.reducedMotion ? Math.floor(u) : u);
			if (sendT >= SEND_SECONDS) sendT = null;
		}
		const wave = clamp(holdWave ?? p.wave ?? 3, 0, 3);
		const x1 = p.x1 ?? 0;
		const x2 = p.x2 ?? 0;
		const focus = (p.focus ?? -1) >= 0 ? Math.round(clamp(p.focus, 0, 2)) : -1;
		const matrix = clamp(p.matrix ?? 0, 0, 1);
		const layer = Math.round(clamp(p.layer ?? 1, 1, 3));

		// Camera: lerp the world rectangle with the matrix reveal.
		const world: WorldRect = {
			x0: -lerp(WORLD_X_HIDDEN, WORLD_X_PANEL, matrix),
			x1: lerp(WORLD_X_HIDDEN, WORLD_X_PANEL, matrix),
			y0: lerp(WORLD_Y0, WORLD_Y0_PANEL, matrix),
			y1: lerp(WORLD_Y1, WORLD_Y1_PANEL, matrix)
		};
		diagram.rect = {
			x0: NET.x0,
			x1: NET.x1,
			y0: lerp(NET.y0, NET_STACKED.y0, matrix),
			y1: lerp(NET.y1, NET_STACKED.y1, matrix)
		};
		fitCamera(camera, v, world, lessonInsets(v));
		v.setViewProj(camera.matrix(v.aspect));

		const tr = forward(net, [x1, x2]);
		const computed = (l: number) => (l === 0 ? 1 : clamp(wave - (l - 1), 0, 1));

		// ---- diagram state ----
		for (let l = 0; l <= last; l++) {
			const c = computed(l);
			for (let i = 0; i < SIZES[l]; i++) {
				const s = diagram.node(l, i);
				const val = tr.a[l][i] * c;
				s.value = val;
				s.label = c > 0.5 ? fmt(tr.a[l][i]) : undefined;
				s.name = l === 0 ? (i === 0 ? 'x₁' : 'x₂') : l === last ? 'ŷ' : undefined;
				let f = 0.5 + 0.5 * c;
				if (focus >= 0 && !(l === 0 || (l === 1 && i === focus))) f *= 0.2;
				s.emphasis = f * dimFor('data', val);
			}
		}
		for (let l = 0; l < last; l++) {
			const mid = clamp(wave - l, 0, 1); // 0 before, 1 once computed
			const flowing = wave > l && wave < l + 1;
			for (let i = 0; i < SIZES[l]; i++) {
				for (let j = 0; j < SIZES[l + 1]; j++) {
					const w = net.W[l][j * SIZES[l] + i];
					const e = diagram.edge(l, i, j);
					e.weight = w * WEIGHT_SCALE;
					const mag = Math.min(1, Math.abs(tr.a[l][i] * w) * 1.5 + 0.3);
					e.signal = flowing ? 1 : mid >= 1 ? 0.25 * mag : 0;
					let f = 1;
					if (focus >= 0 && !(l === 0 && j === focus)) f = 0.2;
					if (matrix > 0) f *= lerp(1, l === layer - 1 ? 1 : DIM, matrix);
					e.emphasis = f * dimFor('param', w);
					e.label = focus >= 0 && l === 0 && j === focus ? `w ${fmt(w)}` : undefined;
				}
			}
		}
		diagram.commit();
		const labels: Label[] = diagram.labels(v);

		// ---- focus arithmetic ----
		if (focus >= 0) {
			const z = tr.z[0][focus];
			const bias = net.b[0][focus];
			const w0 = net.W[0][focus * 2];
			const w1 = net.W[0][focus * 2 + 1];
			const [px, py] = v.project(...diagram.nodePos(1, focus));
			const [, by] = v.project(0, diagram.rect.y0);
			const baseY = Math.max(by + 62, py + 62);
			const c = computed(1);
			labels.push({
				id: 'focus-sum',
				x: px,
				y: baseY,
				text: `${paren(w0)}·${paren(x1)} + ${paren(w1)}·${paren(x2)} + ${paren(bias)} = ${fmt(z)}`,
				tone: 'text',
				mono: true,
				opacity: c
			});
			labels.push({
				id: 'focus-act',
				x: px,
				y: baseY + 26,
				text: `tanh(${fmt(z)}) = ${fmt(Math.tanh(z))}`,
				tone: tr.a[1][focus] < 0 ? 'neg' : 'data',
				mono: true,
				opacity: c
			});
		}

		// ---- matrix panel ----
		quads.count = 0;
		arrow.count = 0;
		if (matrix > 0.02) {
			const nIn = SIZES[layer - 1];
			const nOut = SIZES[layer];
			const W = net.W[layer - 1];
			const b = net.b[layer - 1];
			const aIn = tr.a[layer - 1];
			const aOut = tr.a[layer];
			const cIn = computed(layer - 1);
			const cOut = computed(layer);
			const rowY = (j: number, n: number) => PANEL_Y + ((n - 1) / 2) * PITCH - j * PITCH;
			const { ppu } = pxPerUnit();
			const fits = CELL * ppu >= 36;
			let qi = 0;
			let ln = 0;
			// A cell is a sheet-colored box with a value wash on top and a thin muted outline.
			const quad = (x: number, y: number, v: number, f: number) => {
				quads.set(qi++, { x, y, w: CELL, h: CELL, color: sheetFill(f) });
				quads.set(qi++, { x, y, w: CELL, h: CELL, color: diverging(v, 1, f) });
				const h = CELL / 2;
				ln = arrow.setPolyline(
					ln,
					[
						[x - h, y - h],
						[x + h, y - h],
						[x + h, y + h],
						[x - h, y + h],
						[x - h, y - h]
					],
					{ width: 1, color: tone('muted', 0.5 * f) }
				);
			};
			const lab = (
				id: string,
				x: number,
				y: number,
				text: string,
				o: { tone?: Tone; dy?: number; opacity?: number; mono?: boolean } = {}
			) => {
				const op = (o.opacity ?? 1) * matrix;
				if (op < 0.02) return;
				const [lx, ly] = v.project(x, y);
				labels.push({
					id,
					x: lx,
					y: ly + (o.dy ?? 0),
					text,
					tone: o.tone ?? 'text',
					mono: o.mono ?? true,
					opacity: op
				});
			};
			const topY = (n: number) => rowY(0, n) + CELL / 2;
			const head = (id: string, x: number, n: number, text: string) =>
				lab(id, x, topY(n), text, { tone: 'muted', dy: -18 });

			for (let i = 0; i < nIn; i++) {
				const val = aIn[i] * cIn;
				const f = matrix * dimFor('data', val) * (0.5 + 0.5 * cIn);
				const y = rowY(i, nIn);
				quad(X_IN, y, val, f);
				if (fits && cIn > 0.5)
					lab(`m-in${i}`, X_IN, y, fmt(aIn[i]), { opacity: dimFor('data', val) });
			}
			for (let j = 0; j < nOut; j++) {
				const y = rowY(j, nOut);
				for (let i = 0; i < nIn; i++) {
					const w = W[j * nIn + i];
					const f = matrix * dimFor('param', w);
					const x = X_W0 + i * PITCH;
					quad(x, y, w / 1.5, f);
					if (fits) lab(`m-w${j}.${i}`, x, y, fmt(w), { opacity: dimFor('param', w) });
				}
				const bf = matrix * dimFor('param', b[j]);
				quad(X_B, y, b[j] / 1.5, bf);
				if (fits) lab(`m-b${j}`, X_B, y, fmt(b[j]), { opacity: dimFor('param', b[j]) });

				const val = aOut[j] * cOut;
				const of = matrix * dimFor('data', val) * (0.5 + 0.5 * cOut);
				quad(X_OUT, y, val, of);
				if (fits && cOut > 0.5) lab(`m-out${j}`, X_OUT, y, fmt(aOut[j]), { opacity: dimFor('data', val) });
			}
			quads.count = qi;

			const wMid = X_W0 + ((nIn - 1) / 2) * PITCH;
			head('h-in', X_IN, nIn, 'a in');
			head('h-w', wMid, nOut, 'W');
			head('h-b', X_B, nOut, 'b');
			head('h-out', X_OUT, nOut, 'a out');
			lab('op-times', X_TIMES, PANEL_Y, '×', { tone: 'muted', mono: false });
			lab('op-plus', X_PLUS, PANEL_Y, '+', { tone: 'muted', mono: false });
			lab('op-act', (ARROW.x0 + ARROW.x1) / 2, PANEL_Y, layer === 3 ? 'sigmoid' : 'tanh', {
				tone: 'muted',
				dy: 22
			});
			const a = matrix;
			const col = tone('muted', 0.6 * a);
			const hw = 0.09;
			arrow.set(ln++, { a: [ARROW.x0, PANEL_Y], b: [ARROW.x1, PANEL_Y], width: 2, color: col });
			arrow.set(ln++, { a: [ARROW.x1 - hw, PANEL_Y + hw], b: [ARROW.x1, PANEL_Y], width: 2, color: col });
			arrow.set(ln++, { a: [ARROW.x1 - hw, PANEL_Y - hw], b: [ARROW.x1, PANEL_Y], width: 2, color: col });
			arrow.count = ln;
		}
		quads.upload();
		arrow.upload();
		ctx.labels.set(labels);

		const out = round2(tr.a[last][0]);
		if (out !== lastOut) {
			lastOut = out;
			ctx.emit('out', out);
		}
	}

	function pxPerUnit() {
		const [a] = ctx.view.project(0, 0);
		const [b] = ctx.view.project(1, 0);
		return { ppu: Math.abs(b - a) };
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			diagram = new NetworkDiagram(c, SIZES);
			diagram.rect = NET;
			quads = new Quads(c, { capacity: 64, blend: 'alpha' });
			arrow = new Lines(c, { capacity: 128, blend: 'alpha' });
		},
		setParams(params) {
			p = { ...params };
			if (p.wave !== lastParamWave) {
				lastParamWave = p.wave;
				holdWave = null;
				sendT = null;
			}
		},
		command(id) {
			if (id !== 'send') return;
			holdWave = 0;
			sendT = 0;
		},
		highlight(t) {
			hl = t;
		},
		update(frame) {
			rebuild(frame.dt);
		},
		draw(pass) {
			quads.draw(pass);
			arrow.draw(pass);
			diagram.draw(pass);
		},
		dispose() {
			diagram.destroy();
			quads.destroy();
			arrow.destroy();
		}
	};
	return scene;
};

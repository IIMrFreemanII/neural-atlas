import { tone } from '../../gpu/colors.ts';
import { Lines, bezier } from '../../gpu/primitives/lines.ts';
import { Points } from '../../gpu/primitives/points.ts';
import type { Label, SceneContext, Tone, View } from '../../gpu/types.ts';
import type { WorldRect } from '../fit.ts';

export interface NodeState {
	value: number;
	tone?: Tone;
	emphasis: number;
	name?: string;
	label?: string;
}

export interface EdgeState {
	weight: number;
	signal: number;
	tone?: Tone;
	reverse: boolean;
	emphasis: number;
	label?: string;
}

const CURVE_SEGS = 24;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export class NetworkDiagram {
	readonly sizes: number[];
	rect: WorldRect = { x0: -1.5, x1: 1.5, y0: -1, y1: 1 };

	private ctx: Pick<SceneContext, 'gpu' | 'target' | 'view' | 'reducedMotion'>;
	private radius: number;
	private nodes: NodeState[][];
	private edges: EdgeState[][];
	private lines: Lines;
	private rings: Points;
	private glows: Points;

	constructor(
		ctx: Pick<SceneContext, 'gpu' | 'target' | 'view' | 'reducedMotion'>,
		sizes: number[],
		opts: { nodeRadius?: number } = {}
	) {
		this.ctx = ctx;
		this.sizes = sizes.slice();
		this.radius = opts.nodeRadius ?? 13;
		this.nodes = sizes.map((n) =>
			Array.from({ length: n }, () => ({ value: 0, emphasis: 1 }) as NodeState)
		);
		this.edges = [];
		let edgeCount = 0;
		let nodeCount = sizes[0] ?? 0;
		for (let l = 0; l + 1 < sizes.length; l++) {
			this.edges.push(
				Array.from(
					{ length: sizes[l] * sizes[l + 1] },
					() => ({ weight: 1, signal: 0, reverse: false, emphasis: 1 }) as EdgeState
				)
			);
			edgeCount += sizes[l] * sizes[l + 1];
			nodeCount += sizes[l + 1];
		}
		this.lines = new Lines(ctx, { capacity: Math.max(1, edgeCount * CURVE_SEGS), blend: 'alpha' });
		this.rings = new Points(ctx, { capacity: Math.max(1, nodeCount), blend: 'alpha' });
		this.glows = new Points(ctx, { capacity: Math.max(1, nodeCount), blend: 'additive' });
	}

	nodePos(layer: number, index: number): [number, number] {
		const { x0, x1, y0, y1 } = this.rect;
		const L = this.sizes.length;
		const x = L <= 1 ? (x0 + x1) / 2 : x0 + ((x1 - x0) * layer) / (L - 1);
		const n = this.sizes[layer];
		const y = n <= 1 ? (y0 + y1) / 2 : y1 - ((y1 - y0) * index) / (n - 1);
		return [x, y];
	}

	node(layer: number, index: number): NodeState {
		return this.nodes[layer][index];
	}

	edge(layer: number, from: number, to: number): EdgeState {
		return this.edges[layer][from * this.sizes[layer + 1] + to];
	}

	private nodeTone(s: NodeState): Tone {
		return s.tone ?? (s.value < 0 ? 'neg' : 'data');
	}

	/** World units per CSS pixel along x (the camera is orthographic). */
	private worldPerPx(): number {
		const v = this.ctx.view;
		const [a] = v.project(0, 0);
		const [b] = v.project(1, 0);
		const ppu = Math.abs(b - a);
		return ppu > 1e-6 ? 1 / ppu : 0.01;
	}

	commit(): void {
		const motion = this.ctx.reducedMotion ? 0 : 1;
		const wpp = this.worldPerPx();
		const inset = this.radius * wpp;
		let n = 0;
		for (let l = 0; l + 1 < this.sizes.length; l++) {
			for (let i = 0; i < this.sizes[l]; i++) {
				for (let j = 0; j < this.sizes[l + 1]; j++) {
					const e = this.edge(l, i, j);
					const [ax, ay] = this.nodePos(l, i);
					const [bx, by] = this.nodePos(l + 1, j);
					const sx = ax + inset;
					const ex = bx - inset;
					const mid = (ex - sx) * 0.5;
					let pts = bezier([sx, ay], [sx + mid, ay], [ex - mid, by], [ex, by], CURVE_SEGS);
					if (e.reverse) pts = pts.reverse();
					const t: Tone = e.tone ?? (e.weight < 0 ? 'neg' : 'param');
					const width = clamp(1 + 3 * Math.abs(e.weight), 1, 7);
					const f = e.emphasis;
					const intensity = width < 2 ? 0.9 : 1.2;
					n = this.lines.setPolyline(n, pts, {
						width,
						color: tone(t, intensity * f, 0.7 * f),
						pulseSpeed: 0.3 * motion,
						pulseCount: 2,
						pulseGain: 3 * Math.min(1, Math.max(0, e.signal)) * motion
					});
				}
			}
		}
		this.lines.count = n;

		let k = 0;
		for (let l = 0; l < this.sizes.length; l++) {
			for (let i = 0; i < this.sizes[l]; i++) {
				const s = this.node(l, i);
				const [x, y] = this.nodePos(l, i);
				const t = this.nodeTone(s);
				const f = s.emphasis;
				const kk = Math.min(Math.abs(s.value), 1.5) / 1.5;
				this.rings.set(k, { x, y, radius: this.radius, ring: 2, color: tone(t, 1.3 * f, f) });
				this.glows.set(k, {
					x,
					y,
					radius: this.radius * 3.2,
					softness: 1,
					color: tone(t, (0.15 + 2.1 * kk) * f, (0.1 + 0.6 * kk) * f)
				});
				k++;
			}
		}
		this.rings.count = k;
		this.glows.count = k;
		this.lines.upload();
		this.rings.upload();
		this.glows.upload();
	}

	draw(pass: GPURenderPassEncoder): void {
		this.lines.draw(pass);
		this.glows.draw(pass);
		this.rings.draw(pass);
	}

	labels(view: View, prefix = 'net'): Label[] {
		const out: Label[] = [];
		const r = this.radius;
		for (let l = 0; l < this.sizes.length; l++) {
			for (let i = 0; i < this.sizes[l]; i++) {
				const s = this.node(l, i);
				const [px, py] = view.project(...this.nodePos(l, i));
				if (s.name) {
					out.push({
						id: `${prefix}-n${l}.${i}`,
						x: px,
						y: py - r - 14,
						text: s.name,
						opacity: s.emphasis
					});
				}
				if (s.label) {
					out.push({
						id: `${prefix}-v${l}.${i}`,
						x: px,
						y: py + r + 14,
						text: s.label,
						tone: this.nodeTone(s),
						mono: true,
						opacity: s.emphasis
					});
				}
			}
		}
		for (let l = 0; l + 1 < this.sizes.length; l++) {
			for (let i = 0; i < this.sizes[l]; i++) {
				for (let j = 0; j < this.sizes[l + 1]; j++) {
					const e = this.edge(l, i, j);
					if (!e.label) continue;
					const [ax, ay] = this.nodePos(l, i);
					const [bx, by] = this.nodePos(l + 1, j);
					const [px, py] = view.project((ax + bx) / 2, (ay + by) / 2);
					out.push({
						id: `${prefix}-e${l}.${i}.${j}`,
						x: px,
						y: py - 12,
						text: e.label,
						tone: e.tone ?? (e.weight < 0 ? 'neg' : 'param'),
						mono: true,
						opacity: e.emphasis
					});
				}
			}
		}
		return out;
	}

	destroy(): void {
		this.lines.destroy();
		this.rings.destroy();
		this.glows.destroy();
	}
}

import { paperFill, tone } from '../../gpu/colors.ts';
import { Lines, bezier } from '../../gpu/primitives/lines.ts';
import { Points } from '../../gpu/primitives/points.ts';
import type { Label, SceneContext, Tone, View } from '../../gpu/types.ts';
import type { WorldRect } from '../fit.ts';
import { Pulses, type PulsePath } from './pulses.ts';

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

export interface PulseOptions {
	/** Defaults to 'data' going forward and 'grad' going backward. */
	tone?: Tone;
	/** Seconds one layer of edges takes. Default 0.55. */
	layerDuration?: number;
	/** Only run along these edge layers (0 = input to first hidden layer ...), still in travel order. Default all. */
	edgeLayers?: number[];
	/** Called as the pulse arrives at a layer of nodes (so values can appear as it lands). */
	onLayer?: (nodeLayer: number) => void;
	onDone?: () => void;
}

interface PulseRun {
	elapsed: number;
	layerDuration: number;
	direction: 'forward' | 'backward';
	steps: number[];
	next: number;
	onLayer?: (nodeLayer: number) => void;
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
	private discs: Points;
	private pulses: Pulses;
	private run: PulseRun | null = null;
	/** Forward-direction polyline of each edge, as of the last commit(): [layer][from * n + to]. */
	private edgePts: Array<Array<Array<[number, number]>>> = [];

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
		this.discs = new Points(ctx, { capacity: Math.max(1, nodeCount), blend: 'alpha' });
		this.pulses = new Pulses(ctx as SceneContext, { capacity: Math.max(1, edgeCount) });
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
		const wpp = this.worldPerPx();
		const inset = this.radius * wpp;
		let n = 0;
		this.edgePts = [];
		for (let l = 0; l + 1 < this.sizes.length; l++) {
			const layerPts: Array<Array<[number, number]>> = [];
			this.edgePts.push(layerPts);
			for (let i = 0; i < this.sizes[l]; i++) {
				for (let j = 0; j < this.sizes[l + 1]; j++) {
					const e = this.edge(l, i, j);
					const [ax, ay] = this.nodePos(l, i);
					const [bx, by] = this.nodePos(l + 1, j);
					const sx = ax + inset;
					const ex = bx - inset;
					const mid = (ex - sx) * 0.5;
					const fwd = bezier([sx, ay], [sx + mid, ay], [ex - mid, by], [ex, by], CURVE_SEGS);
					layerPts.push(fwd);
					const pts = e.reverse ? fwd.slice().reverse() : fwd;
					const t: Tone = e.tone ?? (e.weight < 0 ? 'neg' : 'param');
					const width = clamp(1 + 3 * Math.abs(e.weight), 1, 7);
					const f = e.emphasis;
					// Edges stay static ink; moving signal is drawn by pulse().
					n = this.lines.setPolyline(n, pts, {
						width,
						color: tone(t, 0.7 * f),
						pulseSpeed: 0,
						pulseCount: 2,
						pulseGain: 0
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
				// A paper-colored disc hides the edges behind the node; the ring carries tone and value.
				this.discs.set(k, { x, y, radius: this.radius, color: paperFill() });
				this.rings.set(k, {
					x,
					y,
					radius: this.radius,
					ring: 1.5 + 1 * kk,
					color: tone(t, (0.55 + 0.45 * kk) * f)
				});
				k++;
			}
		}
		this.rings.count = k;
		this.discs.count = k;
		this.lines.upload();
		this.rings.upload();
		this.discs.upload();
	}

	/**
	 * Run a discrete pulse layer by layer along the edges (all edges between layer k and k+1 together, then the
	 * next layer). 'forward' goes input to output, 'backward' output to input. Uses the edge curves of the last
	 * commit(). Needs update(dt) every frame. Under reduced motion nothing is drawn and onLayer/onDone fire on the
	 * next update().
	 */
	pulse(direction: 'forward' | 'backward', opts: PulseOptions = {}): void {
		const layerDuration = opts.layerDuration ?? 0.55;
		const t: Tone = opts.tone ?? (direction === 'forward' ? 'data' : 'grad');
		const nLayers = this.sizes.length - 1;
		const steps: number[] = [];
		for (let s = 0; s < nLayers; s++) {
			const l = direction === 'forward' ? s : nLayers - 1 - s;
			if (!opts.edgeLayers || opts.edgeLayers.includes(l)) steps.push(l);
		}
		const paths: PulsePath[] = [];
		steps.forEach((l, s) => {
			for (const pts of this.edgePts[l] ?? []) {
				paths.push({
					points: direction === 'forward' ? pts : pts.slice().reverse(),
					tone: t,
					delay: s * layerDuration
				});
			}
		});
		const run: PulseRun = { elapsed: 0, layerDuration, direction, steps, next: 0, onLayer: opts.onLayer };
		this.run = run;
		this.pulses.fire(paths, layerDuration, () => {
			if (this.run !== run) return;
			this.flushLayers(run, steps.length);
			this.run = null;
			opts.onDone?.();
		});
	}

	get pulsing(): boolean {
		return this.pulses.busy;
	}

	clearPulses(): void {
		this.run = null;
		this.pulses.clear();
	}

	private flushLayers(run: PulseRun, upTo: number): void {
		while (run.next < upTo) {
			const l = run.steps[run.next++];
			run.onLayer?.(run.direction === 'forward' ? l + 1 : l);
		}
	}

	/** Advance pulses. Call once per frame, after commit(). */
	update(dt: number): void {
		const run = this.run;
		if (run) run.elapsed += dt;
		this.pulses.update(dt);
		const cur = this.run;
		if (cur && cur === run) {
			const arrived = Math.min(cur.steps.length, Math.floor(cur.elapsed / cur.layerDuration + 1e-6));
			this.flushLayers(cur, arrived);
		}
	}

	draw(pass: GPURenderPassEncoder): void {
		this.lines.draw(pass);
		this.discs.draw(pass);
		this.rings.draw(pass);
		this.pulses.draw(pass);
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
		this.discs.destroy();
		this.pulses.dispose();
	}
}

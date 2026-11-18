import { paperFill, tone } from '../../gpu/colors.ts';
import { Points } from '../../gpu/primitives/points.ts';
import { easeInOut } from '../../gpu/space.ts';
import type { SceneContext, Tone } from '../../gpu/types.ts';

export type PulsePoint = [number, number] | [number, number, number];

export interface PulsePath {
	points: PulsePoint[];
	tone: Tone;
	/** Seconds to wait before this pulse starts. Default 0. */
	delay?: number;
}

type P3 = [number, number, number];

/** Cumulative arc length at each point (3D; missing z = 0). */
export function cumulativeLengths(points: ReadonlyArray<PulsePoint>): number[] {
	const out = [0];
	for (let i = 1; i < points.length; i++) {
		const a = points[i - 1];
		const b = points[i];
		out.push(out[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1], (b[2] ?? 0) - (a[2] ?? 0)));
	}
	return out;
}

/** The point a fraction f (0..1) of the way along the polyline by arc length. */
export function pointAtFraction(points: ReadonlyArray<PulsePoint>, cum: ReadonlyArray<number>, f: number): P3 {
	if (points.length === 0) return [0, 0, 0];
	const total = cum[cum.length - 1];
	const first = points[0];
	if (points.length === 1 || total <= 0) return [first[0], first[1], first[2] ?? 0];
	const target = Math.max(0, Math.min(1, f)) * total;
	let i = 1;
	while (i < cum.length - 1 && cum[i] < target) i++;
	const seg = cum[i] - cum[i - 1];
	const k = seg > 0 ? (target - cum[i - 1]) / seg : 0;
	const a = points[i - 1];
	const b = points[i];
	return [
		a[0] + (b[0] - a[0]) * k,
		a[1] + (b[1] - a[1]) * k,
		(a[2] ?? 0) + ((b[2] ?? 0) - (a[2] ?? 0)) * k
	];
}

/** Trailing dots sit this far behind the head, as a fraction of the path's length. */
export const TRAIL_LAG = [0.045, 0.09];

export interface PulseDots {
	head: P3;
	trail: [P3, P3];
}

/** Dot positions for one pulse at `elapsed` seconds since fire(); null while waiting or once it has arrived. */
export function pulseDots(
	points: ReadonlyArray<PulsePoint>,
	cum: ReadonlyArray<number>,
	elapsed: number,
	delay: number,
	duration: number
): PulseDots | null {
	const local = elapsed - delay;
	if (local < 0 || local >= duration) return null;
	const f = easeInOut(duration > 0 ? local / duration : 1);
	return {
		head: pointAtFraction(points, cum, f),
		trail: [
			pointAtFraction(points, cum, f - TRAIL_LAG[0]),
			pointAtFraction(points, cum, f - TRAIL_LAG[1])
		]
	};
}

interface Active {
	points: PulsePoint[];
	cum: number[];
	tone: Tone;
	delay: number;
}

/** The timing of a set of pulses, with no GPU in it. */
export class PulseTimeline {
	private paths: Active[] = [];
	private elapsed = 0;
	private duration = 0;
	private onDone: (() => void) | undefined;
	private pending = false;
	/** Capacity (max simultaneous pulses). */
	constructor(
		readonly capacity: number,
		readonly reducedMotion: boolean
	) {}

	get busy(): boolean {
		return this.pending;
	}

	/** Start pulses, replacing any still running (their onDone is dropped). */
	fire(paths: PulsePath[], duration: number, onDone?: () => void): void {
		this.paths = paths.slice(0, this.capacity).map((p) => ({
			points: p.points,
			cum: cumulativeLengths(p.points),
			tone: p.tone,
			delay: p.delay ?? 0
		}));
		this.elapsed = 0;
		this.duration = duration;
		this.onDone = onDone;
		this.pending = true;
	}

	/** Advance time; calls onDone once everything has arrived (on the next update when reduced motion is on). */
	update(dt: number): void {
		if (!this.pending) return;
		this.elapsed += dt;
		const end = this.reducedMotion ? 0 : this.paths.reduce((m, p) => Math.max(m, p.delay + this.duration), 0);
		if (this.reducedMotion || this.elapsed >= end) {
			const cb = this.onDone;
			this.pending = false;
			this.paths = [];
			this.onDone = undefined;
			cb?.();
		}
	}

	/** Visible pulses right now (none under reduced motion). */
	dots(): Array<{ tone: Tone; dots: PulseDots }> {
		if (!this.pending || this.reducedMotion) return [];
		const out: Array<{ tone: Tone; dots: PulseDots }> = [];
		for (const p of this.paths) {
			const d = pulseDots(p.points, p.cum, this.elapsed, p.delay, this.duration);
			if (d) out.push({ tone: p.tone, dots: d });
		}
		return out;
	}

	clear(): void {
		this.paths = [];
		this.onDone = undefined;
		this.pending = false;
	}
}

/**
 * Discrete signal pulses: a solid ink dot with a thin paper ring and two fainter trailing dots, travelling a path
 * by arc length. They only run when fired, never idly. Under reduced motion nothing is drawn and onDone fires on
 * the next update().
 */
export class Pulses {
	private timeline: PulseTimeline;
	private points: Points;
	private radius: number;

	constructor(ctx: SceneContext, opts: { capacity: number; radiusPx?: number }) {
		this.radius = opts.radiusPx ?? 4;
		this.timeline = new PulseTimeline(opts.capacity, ctx.reducedMotion);
		this.points = new Points(ctx, { capacity: opts.capacity * 4 });
	}

	get busy(): boolean {
		return this.timeline.busy;
	}

	fire(paths: PulsePath[], duration: number, onDone?: () => void): void {
		this.timeline.fire(paths, duration, onDone);
	}

	update(dt: number): void {
		this.timeline.update(dt);
		const r = this.radius;
		const paper = paperFill();
		let n = 0;
		for (const { tone: t, dots } of this.timeline.dots()) {
			const put = (p: P3, radius: number, color: [number, number, number, number]) =>
				this.points.set(n++, { x: p[0], y: p[1], z: p[2], radius, color });
			put(dots.trail[1], r * 0.5, tone(t, 0.28));
			put(dots.trail[0], r * 0.7, tone(t, 0.5));
			put(dots.head, r + 1.5, paper);
			put(dots.head, r, tone(t, 1));
		}
		this.points.count = n;
		this.points.upload();
	}

	draw(pass: GPURenderPassEncoder): void {
		this.points.draw(pass);
	}

	clear(): void {
		this.timeline.clear();
		this.points.count = 0;
	}

	dispose(): void {
		this.points.destroy();
	}
}

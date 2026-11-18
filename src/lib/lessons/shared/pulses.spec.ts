import { describe, expect, it, vi } from 'vitest';
import { PulseTimeline, cumulativeLengths, pointAtFraction, pulseDots, type PulsePath } from './pulses.ts';

const path: PulsePath = {
	points: [
		[0, 0],
		[4, 0],
		[4, 4]
	],
	tone: 'data'
};

describe('arc-length helpers', () => {
	it('accumulates 2D and 3D lengths', () => {
		expect(cumulativeLengths(path.points)).toEqual([0, 4, 8]);
		expect(cumulativeLengths([[0, 0, 0], [0, 3, 4]])).toEqual([0, 5]);
	});

	it('places points by arc length, not by index', () => {
		const cum = cumulativeLengths(path.points);
		expect(pointAtFraction(path.points, cum, 0)).toEqual([0, 0, 0]);
		expect(pointAtFraction(path.points, cum, 0.25)).toEqual([2, 0, 0]);
		expect(pointAtFraction(path.points, cum, 0.5)).toEqual([4, 0, 0]);
		expect(pointAtFraction(path.points, cum, 0.75)).toEqual([4, 2, 0]);
		expect(pointAtFraction(path.points, cum, 1)).toEqual([4, 4, 0]);
		expect(pointAtFraction(path.points, cum, 2)).toEqual([4, 4, 0]);
		expect(pointAtFraction(path.points, cum, -1)).toEqual([0, 0, 0]);
	});

	it('interpolates z and survives degenerate paths', () => {
		const p: Array<[number, number, number]> = [[0, 0, 0], [0, 0, 2]];
		expect(pointAtFraction(p, cumulativeLengths(p), 0.5)).toEqual([0, 0, 1]);
		const same: Array<[number, number]> = [[1, 1], [1, 1]];
		expect(pointAtFraction(same, cumulativeLengths(same), 0.5)).toEqual([1, 1, 0]);
	});
});

describe('pulseDots', () => {
	const cum = cumulativeLengths(path.points);

	it('is hidden before its delay and after arriving', () => {
		expect(pulseDots(path.points, cum, 0.4, 0.5, 1)).toBeNull();
		expect(pulseDots(path.points, cum, 1.5, 0.5, 1)).toBeNull();
	});

	it('eases from the start to the end and trails behind the head', () => {
		const start = pulseDots(path.points, cum, 0.5, 0.5, 1)!;
		expect(start.head).toEqual([0, 0, 0]);
		const mid = pulseDots(path.points, cum, 1.0, 0.5, 1)!;
		expect(mid.head[0]).toBeCloseTo(4, 10); // halfway through time = halfway along (8 / 2)
		expect(mid.head[1]).toBeCloseTo(0, 10);
		expect(mid.trail[0][0]).toBeLessThan(mid.head[0]);
		expect(mid.trail[1][0]).toBeLessThan(mid.trail[0][0]);
		// ease in: a quarter of the way in time is well short of a quarter of the way along
		const early = pulseDots(path.points, cum, 0.75, 0.5, 1)!;
		expect(early.head[0]).toBeLessThan(2);
		const late = pulseDots(path.points, cum, 1.49, 0.5, 1)!;
		expect(late.head[1]).toBeGreaterThan(3.9);
	});
});

describe('PulseTimeline', () => {
	it('runs to completion once and calls onDone', () => {
		const done = vi.fn();
		const t = new PulseTimeline(4, false);
		expect(t.busy).toBe(false);
		t.fire([path, { ...path, delay: 0.5 }], 1, done);
		expect(t.busy).toBe(true);
		t.update(0.25);
		expect(t.dots()).toHaveLength(1); // the delayed one has not started
		t.update(0.5);
		expect(t.dots()).toHaveLength(2);
		t.update(0.5);
		expect(done).not.toHaveBeenCalled(); // second path still travelling (ends at 1.5)
		t.update(0.5);
		expect(done).toHaveBeenCalledTimes(1);
		expect(t.busy).toBe(false);
		expect(t.dots()).toEqual([]);
		t.update(1);
		expect(done).toHaveBeenCalledTimes(1);
	});

	it('draws nothing under reduced motion and finishes on the next update', () => {
		const done = vi.fn();
		const t = new PulseTimeline(4, true);
		t.fire([path], 1, done);
		expect(t.dots()).toEqual([]);
		expect(done).not.toHaveBeenCalled();
		t.update(0);
		expect(done).toHaveBeenCalledTimes(1);
		expect(t.busy).toBe(false);
	});

	it('caps at capacity and clear() drops everything silently', () => {
		const done = vi.fn();
		const t = new PulseTimeline(2, false);
		t.fire([path, path, path], 1, done);
		t.update(0.5);
		expect(t.dots()).toHaveLength(2);
		t.clear();
		t.update(5);
		expect(done).not.toHaveBeenCalled();
		expect(t.busy).toBe(false);
	});
});

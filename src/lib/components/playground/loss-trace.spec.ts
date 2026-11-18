import { describe, expect, it } from 'vitest';
import { LossTrace, MAX_POINTS } from './loss-trace.ts';

describe('LossTrace', () => {
	it('keeps samples in order', () => {
		const t = new LossTrace();
		t.add(0, 0.7);
		t.add(5, 0.5);
		t.add(9, 0.3);
		expect(t.points).toEqual([
			[0, 0.7],
			[5, 0.5],
			[9, 0.3]
		]);
	});

	it('clears when steps goes down', () => {
		const t = new LossTrace();
		t.add(0, 0.7);
		t.add(10, 0.4);
		t.add(0, 0.9);
		expect(t.points).toEqual([[0, 0.9]]);
		expect(t.max).toBe(0.9);
	});

	it('keeps a repeated steps value', () => {
		const t = new LossTrace();
		t.add(3, 0.5);
		t.add(3, 0.4);
		expect(t.points.length).toBe(2);
	});

	it('thins past MAX_POINTS and keeps the last point', () => {
		const t = new LossTrace();
		const n = MAX_POINTS * 4;
		for (let i = 0; i < n; i++) {
			t.add(i, 1 / (i + 1));
			expect(t.points.length).toBeLessThanOrEqual(MAX_POINTS);
		}
		expect(t.points[t.points.length - 1]).toEqual([n - 1, 1 / n]);
		for (let i = 1; i < t.points.length; i++) expect(t.points[i][0]).toBeGreaterThan(t.points[i - 1][0]);
	});

	it('ignores non-finite loss', () => {
		const t = new LossTrace();
		t.add(0, 0.5);
		t.add(1, NaN);
		t.add(2, Infinity);
		expect(t.points).toEqual([[0, 0.5]]);
	});

	it('tracks the max and resets it on clear', () => {
		const t = new LossTrace();
		expect(t.max).toBe(0);
		t.add(0, 0.4);
		t.add(1, 0.9);
		t.add(2, 0.2);
		expect(t.max).toBe(0.9);
		t.clear();
		expect(t.max).toBe(0);
		expect(t.points).toEqual([]);
	});
});

import { describe, expect, it } from 'vitest';
import { initCenters, runKMeans } from '../../../ml/kmeans.ts';
import { makeCloud } from './data.ts';
import { elbowCurve } from './elbow.ts';

const MAX_K = 6;

/** The k after which one more center helps least, compared with what the center before it gained. */
function bend(curve: Float64Array): number {
	let at = 0;
	let most = 0;
	for (let k = 2; k < curve.length; k++) {
		const gain = curve[k - 2] - curve[k - 1];
		const next = curve[k - 1] - curve[k];
		if (gain / next > most) {
			most = gain / next;
			at = k;
		}
	}
	return at;
}

describe('elbow chart', () => {
	it('falls with every extra center, on every dataset', () => {
		for (let id = 0; id < 4; id++) {
			const c = elbowCurve(makeCloud(id, 1), MAX_K);
			expect(c.length).toBe(MAX_K);
			for (let k = 1; k < MAX_K; k++) expect(c[k]).toBeLessThan(c[k - 1]);
		}
	});

	it('is deterministic', () => {
		const pts = makeCloud(1, 1);
		expect(elbowCurve(pts, MAX_K)).toEqual(elbowCurve(pts, MAX_K));
	});

	it('bends at 5 for the five blobs, as the lesson says', () => {
		const c = elbowCurve(makeCloud(1, 1), MAX_K);
		expect(bend(c)).toBe(5);
		// The fifth center still helps a lot, the sixth does not.
		expect(c[3] - c[4]).toBeGreaterThan(3 * (c[4] - c[5]));
	});

	it('bends at 3 for the three blobs', () => {
		expect(bend(elbowCurve(makeCloud(0, 1), MAX_K))).toBe(3);
	});

	it('is not worse than a full run on all the points at 5 centers', () => {
		// What the learner reads on the stage after a good run must agree with the chart.
		const pts = makeCloud(1, 1);
		let full = Infinity;
		for (let s = 0; s < 10; s++) {
			full = Math.min(full, runKMeans(pts, initCenters(pts, 5, 100 + s, 'plusplus')).inertia);
		}
		const chart = elbowCurve(pts, MAX_K)[4];
		expect(chart / full).toBeGreaterThan(0.999);
		expect(chart / full).toBeLessThan(1.01);
	});

	it('handles fewer points than its sample', () => {
		const pts = new Float64Array([0, 0, 0.1, 0, 1, 1, 1.1, 1]);
		const c = elbowCurve(pts, 2);
		expect(c[0]).toBeGreaterThan(c[1]);
		expect(c[1]).toBeCloseTo(0.01, 6);
	});
});

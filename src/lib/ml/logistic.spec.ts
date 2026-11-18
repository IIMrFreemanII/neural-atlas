import { describe, expect, it } from 'vitest';
import type { Dataset } from './datasets.ts';
import { logisticLoss, logisticProb, logisticStep, type Logistic } from './logistic.ts';
import { gaussian, mulberry32 } from './rng.ts';

function blobs(n: number): Dataset {
	const r = mulberry32(3);
	const x = new Float64Array(n * 2);
	const y = new Float64Array(n);
	for (let k = 0; k < n; k++) {
		const c = k < n / 2 ? 0 : 1;
		const s = c === 1 ? 1 : -1;
		x[2 * k] = s * 0.5 + 0.15 * gaussian(r);
		x[2 * k + 1] = s * 0.4 + 0.15 * gaussian(r);
		y[k] = c;
	}
	return { n, x, y };
}

describe('logistic', () => {
	it('is 0.5 on the line w.x + b = 0', () => {
		const m: Logistic = { w: [2, -1], b: 0.5 };
		// 2*0.25 - 1*1 + 0.5 = 0
		expect(logisticProb(m, 0.25, 1)).toBeCloseTo(0.5, 12);
		expect(logisticProb(m, 5, -5)).toBeGreaterThan(0.99);
	});

	it('step matches the numerical gradient of the loss', () => {
		const d = blobs(40);
		const m: Logistic = { w: [0.6, -0.8], b: 0.5 };
		const eps = 1e-6;
		const num = (f: (m: Logistic, e: number) => void) => {
			const a: Logistic = { w: [...m.w], b: m.b };
			const b: Logistic = { w: [...m.w], b: m.b };
			f(a, eps);
			f(b, -eps);
			return (logisticLoss(a, d) - logisticLoss(b, d)) / (2 * eps);
		};
		const g0 = num((q, e) => (q.w[0] += e));
		const g1 = num((q, e) => (q.w[1] += e));
		const gb = num((q, e) => (q.b += e));
		const lr = 0.1;
		const before = { w: [...m.w], b: m.b };
		const loss = logisticStep(m, d, lr);
		expect(loss).toBeCloseTo(logisticLoss({ w: [before.w[0], before.w[1]], b: before.b }, d), 12);
		expect((before.w[0] - m.w[0]) / lr).toBeCloseTo(g0, 5);
		expect((before.w[1] - m.w[1]) / lr).toBeCloseTo(g1, 5);
		expect((before.b - m.b) / lr).toBeCloseTo(gb, 5);
	});

	it('learns two separated blobs', () => {
		const d = blobs(100);
		const m: Logistic = { w: [0.6, -0.8], b: 0.5 };
		const first = logisticStep(m, d, 1);
		let last = first;
		for (let i = 0; i < 200; i++) last = logisticStep(m, d, 1);
		expect(last).toBeLessThan(first);
		let ok = 0;
		for (let k = 0; k < d.n; k++) {
			if ((logisticProb(m, d.x[2 * k], d.x[2 * k + 1]) >= 0.5 ? 1 : 0) === d.y[k]) ok++;
		}
		expect(ok / d.n).toBe(1);
	});
});

import { describe, expect, it } from 'vitest';
import type { ActivationId } from '../../../ml/activations.ts';
import { X_MAX, X_MIN, evalFit, fitBasis, target, unit } from './basis.ts';

describe('fitBasis', () => {
	it('linear activation gives a straight line for any n', () => {
		for (let n = 1; n <= 6; n++) {
			const f = fitBasis(0, n);
			const h = 0.5;
			for (let x = X_MIN; x + 2 * h <= X_MAX + 1e-9; x += h) {
				const d2 = evalFit(0, n, f, x) - 2 * evalFit(0, n, f, x + h) + evalFit(0, n, f, x + 2 * h);
				expect(Math.abs(d2)).toBeLessThan(1e-3);
			}
		}
	});
	it('more ReLU units fit better', () => {
		expect(fitBasis(2, 6).error).toBeLessThan(fitBasis(2, 2).error);
	});
	it('error is finite and non-negative for every activation', () => {
		for (const act of [0, 1, 2, 3, 4, 5] as ActivationId[]) {
			for (let n = 1; n <= 6; n++) {
				const e = fitBasis(act, n).error;
				expect(Number.isFinite(e)).toBe(true);
				expect(e).toBeGreaterThanOrEqual(0);
			}
		}
	});
	it('evalFit is bias plus the weighted units', () => {
		const f = fitBasis(3, 4);
		const x = 0.7;
		let s = f.bias;
		for (let k = 0; k < 4; k++) s += f.v[k] * unit(3, 4, k, x);
		expect(evalFit(3, 4, f, x)).toBeCloseTo(s, 12);
	});
	it('target is the documented wave', () => {
		expect(target(0)).toBe(0);
	});
});

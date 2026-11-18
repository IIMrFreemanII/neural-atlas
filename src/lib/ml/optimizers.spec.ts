import { describe, expect, it } from 'vitest';
import { SURFACES, curve, initState, step, type Vec2 } from './optimizers.ts';

describe('surfaces', () => {
	const pts: Vec2[] = [
		[0.7, -0.4],
		[-1.3, 0.9],
		[1.9, 1.1]
	];
	for (const s of SURFACES) {
		it(`${s.name} gradient matches finite differences`, () => {
			const h = 1e-5;
			for (const [x, y] of pts) {
				const [gx, gy] = s.grad(x, y);
				const nx = (s.f(x + h, y) - s.f(x - h, y)) / (2 * h);
				const ny = (s.f(x, y + h) - s.f(x, y - h)) / (2 * h);
				expect(gx).toBeCloseTo(nx, 5);
				expect(gy).toBeCloseTo(ny, 5);
			}
		});
	}
	it('two valleys minima are stationary and the left one is global', () => {
		const s = SURFACES[2];
		for (const [x, y] of s.minima) {
			const g = s.grad(x, y);
			expect(Math.hypot(g[0], g[1])).toBeLessThan(1e-9);
		}
		expect(s.minima[0][0]).toBeCloseTo(-1.07, 1);
		expect(s.minima[1][0]).toBeCloseTo(0.92, 1);
		expect(s.f(...s.minima[0])).toBeLessThan(s.f(...s.minima[1]));
	});
});

describe('optimizers', () => {
	const run1d = (lr: number, n: number) => {
		let st = initState([curve.start, 0]);
		for (let i = 0; i < n; i++) st = step('sgd', st, [curve.grad(st.p[0]), 0], lr, 0);
		return st.p[0];
	};
	it('sgd converges on the curve for lr 0.3', () => {
		expect(Math.abs(run1d(0.3, 60))).toBeLessThan(1e-3);
	});
	it('sgd diverges on the curve for lr 1.05', () => {
		expect(Math.abs(run1d(1.05, 60))).toBeGreaterThan(Math.abs(curve.start));
	});
	it('momentum with beta 0 equals sgd', () => {
		let a = initState([1.5, -0.7]);
		let b = initState([1.5, -0.7]);
		const s = SURFACES[1];
		for (let i = 0; i < 20; i++) {
			a = step('sgd', a, s.grad(...a.p), 0.2, 0.5);
			b = step('momentum', b, s.grad(...b.p), 0.2, 0);
		}
		expect(b.p[0]).toBeCloseTo(a.p[0], 12);
		expect(b.p[1]).toBeCloseTo(a.p[1], 12);
	});
	it('adam first step moves each coordinate by about lr', () => {
		for (const g of [
			[0.001, -50],
			[300, 0.02]
		] as Vec2[]) {
			const st = step('adam', initState([0, 0]), g, 0.1, 0);
			expect(Math.abs(st.p[0])).toBeGreaterThan(0.099);
			expect(Math.abs(st.p[0])).toBeLessThan(0.101);
			expect(Math.abs(st.p[1])).toBeGreaterThan(0.099);
			expect(Math.abs(st.p[1])).toBeLessThan(0.101);
			expect(Math.sign(st.p[0])).toBe(-Math.sign(g[0]));
		}
	});
	it('step does not mutate its input', () => {
		for (const kind of ['sgd', 'momentum', 'adam'] as const) {
			const st = initState([1, 2]);
			const snap = JSON.stringify(st);
			step(kind, st, [0.5, -0.5], 0.1, 0.9);
			expect(JSON.stringify(st)).toBe(snap);
		}
	});
});

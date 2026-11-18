import { describe, expect, it } from 'vitest';
import {
	denoised,
	logDensity,
	noise,
	responsibilities,
	reverseStep,
	sampleData,
	schedule,
	score,
	type Mixture
} from './diffusion2d.ts';
import { gaussian, mulberry32 } from './rng.ts';

const three: Mixture = { centers: new Float64Array([-1, -0.5, 1, -0.5, 0, 1.1]), sd: 0.06 };
const meanOf = (m: Mixture): [number, number] => {
	const k = m.centers.length / 2;
	let x = 0;
	let y = 0;
	for (let c = 0; c < k; c++) {
		x += m.centers[c * 2] / k;
		y += m.centers[c * 2 + 1] / k;
	}
	return [x, y];
};
const probes: Array<[number, number]> = [
	[0.3, 0.2],
	[-0.9, -0.4],
	[1.4, 0.8],
	[0.05, -1.1]
];
const times = [0.1, 0.4, 0.7, 0.95];

describe('schedule', () => {
	it('has the right end points and preserves variance', () => {
		expect(schedule(0).alpha).toBeCloseTo(1, 12);
		expect(schedule(0).sigma).toBeCloseTo(0, 12);
		expect(schedule(0.5).alpha).toBeCloseTo(Math.SQRT1_2, 12);
		expect(schedule(0.5).sigma).toBeCloseTo(Math.SQRT1_2, 12);
		expect(schedule(1).alpha).toBeCloseTo(0, 12);
		expect(schedule(1).sigma).toBeCloseTo(1, 12);
		for (const t of [0, 0.13, 0.5, 0.77, 1]) {
			const { alpha, sigma } = schedule(t);
			expect(alpha * alpha + sigma * sigma).toBeCloseTo(1, 12);
		}
	});
});

describe('noise and sampleData', () => {
	it('is deterministic and mixes data with noise', () => {
		const x0 = sampleData(three, 50, mulberry32(3));
		expect(Array.from(sampleData(three, 50, mulberry32(3)))).toEqual(Array.from(x0));
		const eps = new Float64Array(100).map(() => 1);
		const out = new Float64Array(100);
		noise(x0, eps, 0.5, out);
		expect(out[0]).toBeCloseTo(Math.SQRT1_2 * (x0[0] + 1), 12);
		noise(x0, eps, 0, out);
		expect(out[7]).toBeCloseTo(x0[7], 12);
	});
});

describe('responsibilities', () => {
	it('sums to 1 and picks the nearest center at small t', () => {
		const r = new Float64Array(3);
		for (const [x, y] of probes) {
			for (const t of times) {
				responsibilities(three, x, y, t, r);
				expect(r[0] + r[1] + r[2]).toBeCloseTo(1, 10);
			}
		}
		responsibilities(three, 0.9, -0.4, 0.02, r);
		expect(r[1]).toBeGreaterThan(0.999);
		responsibilities(three, 0.1, 1.0, 0.02, r);
		expect(r[2]).toBeGreaterThan(0.999);
	});
	it('stays finite far from every center', () => {
		const r = new Float64Array(3);
		responsibilities(three, 50, 50, 0.01, r);
		expect(r.every(Number.isFinite)).toBe(true);
		expect(r[0] + r[1] + r[2]).toBeCloseTo(1, 10);
	});
});

describe('score', () => {
	it('matches the finite-difference gradient of the log density', () => {
		const h = 1e-5;
		for (const [x, y] of probes) {
			for (const t of times) {
				const [sx, sy] = score(three, x, y, t);
				const gx = (logDensity(three, x + h, y, t) - logDensity(three, x - h, y, t)) / (2 * h);
				const gy = (logDensity(three, x, y + h, t) - logDensity(three, x, y - h, t)) / (2 * h);
				expect(sx).toBeCloseTo(gx, 5);
				expect(sy).toBeCloseTo(gy, 5);
			}
		}
	});
	it('satisfies Tweedie: score = (alpha * denoised - x) / sigma^2', () => {
		for (const [x, y] of probes) {
			for (const t of times) {
				const { alpha, sigma } = schedule(t);
				const [sx, sy] = score(three, x, y, t);
				const [dx, dy] = denoised(three, x, y, t);
				expect(sx).toBeCloseTo((alpha * dx - x) / (sigma * sigma), 8);
				expect(sy).toBeCloseTo((alpha * dy - y) / (sigma * sigma), 8);
			}
		}
	});
});

describe('denoised', () => {
	it('returns the point itself near t = 0 on a center', () => {
		const [x, y] = denoised(three, 1, -0.5, 1e-3);
		expect(x).toBeCloseTo(1, 3);
		expect(y).toBeCloseTo(-0.5, 3);
	});
	it('returns the mean of the centers at t = 1', () => {
		const [mx, my] = meanOf(three);
		for (const [x, y] of probes) {
			const [dx, dy] = denoised(three, x, y, 1);
			expect(dx).toBeCloseTo(mx, 6);
			expect(dy).toBeCloseTo(my, 6);
		}
	});
});

describe('reverseStep', () => {
	const fromNoise = (n: number, seed: number) => {
		const rand = mulberry32(seed);
		return new Float64Array(n * 2).map(() => gaussian(rand));
	};

	it('rebuilds the mixture from pure noise over 100 steps', () => {
		const n = 4000;
		const pts = fromNoise(n, 5);
		const steps = 100;
		for (let s = 0; s < steps; s++) reverseStep(three, pts, 1 - s / steps, 1 - (s + 1) / steps);
		const counts = [0, 0, 0];
		let near = 0;
		for (let i = 0; i < n; i++) {
			let best = Infinity;
			let bc = 0;
			for (let c = 0; c < 3; c++) {
				const d = Math.hypot(pts[i * 2] - three.centers[c * 2], pts[i * 2 + 1] - three.centers[c * 2 + 1]);
				if (d < best) {
					best = d;
					bc = c;
				}
			}
			if (best <= 3.5 * three.sd) {
				near++;
				counts[bc]++;
			}
		}
		expect(near / n).toBeGreaterThanOrEqual(0.97);
		for (const c of counts) {
			expect(c / n).toBeGreaterThan(0.2);
			expect(c / n).toBeLessThan(0.47);
		}
	});

	it('with one step lands every point on the mean of the centers', () => {
		const pts = fromNoise(200, 9);
		reverseStep(three, pts, 1, 0);
		const [mx, my] = meanOf(three);
		for (let i = 0; i < 200; i++) {
			expect(pts[i * 2]).toBeCloseTo(mx, 6);
			expect(pts[i * 2 + 1]).toBeCloseTo(my, 6);
		}
	});
});

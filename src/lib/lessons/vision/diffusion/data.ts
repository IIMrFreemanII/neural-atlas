import type { Mixture } from '../../../ml/diffusion2d.ts';

export const N = 5000;
export const SD = 0.06;

type Pt = [number, number];

function ring(cx: number, cy: number, r: number, n: number, a0 = 0, a1 = Math.PI * 2, closed = true): Pt[] {
	const out: Pt[] = [];
	const span = closed ? n : n - 1;
	for (let i = 0; i < n; i++) {
		const a = a0 + ((a1 - a0) * i) / span;
		out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
	}
	return out;
}

/** Points every `gap` units along a straight stroke, both ends included. */
function stroke(a: Pt, b: Pt, gap = 0.12): Pt[] {
	const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
	const n = Math.max(1, Math.round(len / gap));
	const out: Pt[] = [];
	for (let i = 0; i <= n; i++) out.push([a[0] + ((b[0] - a[0]) * i) / n, a[1] + ((b[1] - a[1]) * i) / n]);
	return out;
}

/** Packs points into a mixture, shifted so the mean of the centers is the origin. */
function pack(pts: Pt[]): Mixture {
	let mx = 0;
	let my = 0;
	for (const p of pts) {
		mx += p[0] / pts.length;
		my += p[1] / pts.length;
	}
	const centers = new Float64Array(pts.length * 2);
	pts.forEach((p, i) => {
		centers[i * 2] = p[0] - mx;
		centers[i * 2 + 1] = p[1] - my;
	});
	return { centers, sd: SD };
}

function spiral(): Pt[] {
	const n = 70;
	const out: Pt[] = [];
	for (let i = 0; i < n; i++) {
		const s = i / (n - 1);
		const a = s * 4 * Math.PI;
		const r = 0.15 + 1.4 * s;
		out.push([r * Math.cos(a), r * Math.sin(a)]);
	}
	return out;
}

function face(): Pt[] {
	return [
		...ring(0, 0, 1.5, 44),
		...ring(-0.5, 0.45, 0.12, 5),
		...ring(0.5, 0.45, 0.12, 5),
		...ring(0, 0.05, 0.85, 14, Math.PI * 1.2, Math.PI * 1.8, false)
	];
}

function rings(): Pt[] {
	return [...ring(-0.6, 0, 0.9, 36), ...ring(0.6, 0, 0.9, 36)];
}

function letters(): Pt[] {
	const a: Pt[] = [
		...stroke([-1.35, -0.8], [-0.85, 0.8]),
		...stroke([-0.85, 0.8], [-0.35, -0.8]),
		...stroke([-1.15, -0.25], [-0.55, -0.25])
	];
	const i: Pt[] = [
		...stroke([0.9, -0.8], [0.9, 0.8]),
		...stroke([0.55, 0.8], [1.25, 0.8]),
		...stroke([0.55, -0.8], [1.25, -0.8])
	];
	return [...a, ...i];
}

export const PATTERNS: { name: string; mixture: Mixture }[] = [
	{ name: 'Spiral', mixture: pack(spiral()) },
	{ name: 'Face', mixture: pack(face()) },
	{ name: 'Rings', mixture: pack(rings()) },
	{ name: 'Letters', mixture: pack(letters()) }
];

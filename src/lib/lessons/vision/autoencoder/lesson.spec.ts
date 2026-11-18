import { describe, expect, it } from 'vitest';
import { encode } from '../../../ml/autoencoder.ts';
import { SAMPLES } from '../cnn-features/data.ts';
import { mapPoints, nearestLabel } from './data.ts';
import lesson from './lesson.ts';
import { WEIGHTS } from './weights.ts';

const SQUARE = 1;
const CIRCLE = 0;

describe('latent space lesson pages', () => {
	// The scene starts a page afresh (the drawing and where the code comes from) when `scenario` changes.
	it('gives every step and question its own scenario number', () => {
		const pages = [...lesson.steps, ...(lesson.questions ?? [])];
		const numbers = pages.map((p) => ({ ...lesson.defaults, ...p.params }).scenario);
		expect(numbers.every((n) => Number.isInteger(n))).toBe(true);
		expect(new Set(numbers).size).toBe(pages.length);
	});
});

describe('what the narration says about the trained weights', () => {
	const points = mapPoints(WEIGHTS);
	const pinned = lesson.steps.find((s) => s.params?.pin === 1)!;

	it('pins "Drawings nobody made" between the circles and the squares, where no drawing landed', () => {
		const x = pinned.params!.zx;
		const y = pinned.params!.zy;
		const near = (p: { x: number; y: number }) => Math.hypot(p.x - x, p.y - y);
		// No map drawing within 0.1 of the point.
		expect(Math.min(...points.map(near))).toBeGreaterThan(0.1);
		// The two nearest clusters are circles and squares.
		const labels = [...points].sort((a, b) => near(a) - near(b)).slice(0, 10).map((p) => p.label);
		expect(new Set(labels)).toEqual(new Set([CIRCLE, SQUARE]));
	});

	it('starts the "draw a cross" task outside the crosses, from the circle sample', () => {
		const q = lesson.questions!.find((q) => q.kind === 'task')!;
		const sample = q.params!.sample;
		const [x, y] = encode(WEIGHTS, SAMPLES[sample - 1]);
		expect(nearestLabel(points, x, y)).toBe(CIRCLE);
	});
});

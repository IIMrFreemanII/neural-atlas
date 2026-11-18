import { describe, expect, it } from 'vitest';
import lesson from './lesson.ts';

describe('k-means lesson pages', () => {
	// The scene starts a page afresh (data, centers, and the count of "New start" presses) when `scenario`
	// changes. Two pages with the same number would hand their state to each other: a run finished on
	// one would already be finished on the other.
	it('gives every step and question its own scenario number', () => {
		const pages = [...lesson.steps, ...(lesson.questions ?? [])];
		const numbers = pages.map((p) => ({ ...lesson.defaults, ...p.params }).scenario);
		expect(numbers.every((n) => Number.isInteger(n))).toBe(true);
		expect(new Set(numbers).size).toBe(pages.length);
	});
});

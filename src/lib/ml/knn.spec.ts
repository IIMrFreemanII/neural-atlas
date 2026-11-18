import { describe, expect, it } from 'vitest';
import type { Dataset } from './datasets.ts';
import { knnNeighbors, knnProb } from './knn.ts';

const d: Dataset = {
	n: 5,
	x: Float64Array.from([0, 0, 1, 0, 2, 0, 3, 0, 4, 0]),
	y: Float64Array.from([1, 0, 0, 1, 1])
};

describe('knn', () => {
	it('k = 1 returns the label of the nearest point', () => {
		expect(knnProb(d, 1, 0.2, 0)).toBe(1);
		expect(knnProb(d, 1, 1.4, 0.3)).toBe(0);
		expect(knnNeighbors(d, 1, 3.9, 0)).toEqual([4]);
	});
	it('k = n gives the class fraction of the whole set', () => {
		expect(knnProb(d, 5, 10, 10)).toBeCloseTo(3 / 5, 12);
	});
	it('orders neighbors by distance and breaks ties by lower index', () => {
		expect(knnNeighbors(d, 3, 2.1, 0)).toEqual([2, 3, 1]);
		expect(knnNeighbors(d, 2, 0.5, 0)).toEqual([0, 1]);
	});
	it('clamps k', () => {
		expect(knnNeighbors(d, 0, 0, 0)).toEqual([0]);
		expect(knnNeighbors(d, 99, 0, 0)).toEqual([0, 1, 2, 3, 4]);
	});
});

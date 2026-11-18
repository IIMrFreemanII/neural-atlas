import { describe, expect, it } from 'vitest';
import { analogy, nearest } from '../../../ml/embedding.ts';
import { cosine } from '../../../ml/vec.ts';
import {
	ANALOGIES,
	CATEGORIES,
	CENTROIDS_3D,
	EXPLAINED,
	POINTS,
	RELATIONS,
	VECTORS,
	WORDS,
	indexOf,
	project3D
} from './data.ts';

describe('embedding table', () => {
	it('has 74 distinct words with 10 numbers each', () => {
		expect(WORDS).toHaveLength(74);
		expect(new Set(WORDS.map((w) => w.word)).size).toBe(74);
		for (const w of WORDS) expect(w.vec).toHaveLength(10);
		expect(WORDS.every((w) => w.category >= 0 && w.category < CATEGORIES.length)).toBe(true);
	});

	it('solves every listed analogy', () => {
		for (const { a, b, c, expected } of ANALOGIES) {
			const r = nearest(VECTORS, analogy(VECTORS[a], VECTORS[b], VECTORS[c]), 1, [a, b, c]);
			expect(WORDS[r[0].index].word).toBe(WORDS[expected].word);
		}
	});

	it('solves every pair of pairs inside each relation', () => {
		for (const rel of RELATIONS) {
			for (const [a, b] of rel.pairs) {
				for (const [c, d] of rel.pairs) {
					if (a === c) continue;
					const r = nearest(VECTORS, analogy(VECTORS[a], VECTORS[b], VECTORS[c]), 1, [a, b, c]);
					expect(WORDS[r[0].index].word, `${rel.name}: ${WORDS[a].word} ${WORDS[b].word} ${WORDS[c].word}`).toBe(
						WORDS[d].word
					);
				}
			}
		}
	});

	it('puts only animals among the 5 nearest neighbours of cat', () => {
		const cat = indexOf('cat');
		const r = nearest(VECTORS, VECTORS[cat], 5, [cat]);
		for (const x of r) expect(WORDS[x.index].category).toBe(1);
	});
});

describe('3D picture', () => {
	it('keeps the farthest point at distance 1 and explains most of the variance', () => {
		const far = Math.max(...POINTS.map((p) => Math.hypot(p[0], p[1], p[2])));
		expect(far).toBeCloseTo(1, 10);
		expect(EXPLAINED).toBeGreaterThan(0.6);
	});

	it('places every word nearer to its own category centroid than to any other', () => {
		const dist = (p: number[], q: number[]) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
		WORDS.forEach((w, i) => {
			const own = dist(POINTS[i], CENTROIDS_3D[w.category]);
			CATEGORIES.forEach((_, k) => {
				if (k !== w.category) expect(own, w.word).toBeLessThan(dist(POINTS[i], CENTROIDS_3D[k]));
			});
		});
	});

	it('draws the arrows of each relation long and nearly parallel', () => {
		for (const rel of RELATIONS) {
			const arrows = rel.pairs.map(([a, b]) => POINTS[b].map((x, i) => x - POINTS[a][i]));
			for (const v of arrows) expect(Math.hypot(...v), rel.name).toBeGreaterThanOrEqual(0.12);
			for (let i = 0; i < arrows.length; i++) {
				for (let j = i + 1; j < arrows.length; j++) {
					expect(cosine(arrows[i], arrows[j]), rel.name).toBeGreaterThanOrEqual(0.9);
				}
			}
		}
	});

	it('projects a word vector onto its own point', () => {
		const i = indexOf('queen');
		const p = project3D(VECTORS[i]);
		for (let k = 0; k < 3; k++) expect(p[k]).toBeCloseTo(POINTS[i][k], 9);
	});
});

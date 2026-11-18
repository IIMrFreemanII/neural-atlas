import { describe, expect, it } from 'vitest';
import { nearest } from '../../../ml/embedding.ts';
import { cosine } from '../../../ml/vec.ts';
import {
	ANALOGIES,
	CATEGORIES,
	CENTROIDS_3D,
	D,
	EXPLAINED,
	POINTS,
	RELATIONS,
	VECTORS,
	WORDS,
	WORD_LIST,
	indexOf,
	project3D,
	solveAnalogy
} from './data.ts';
import { GLOVE, GLOVE_DIM } from './vectors.ts';

const word = (i: number) => WORDS[i].word;
const nearestWord = (w: string) => word(nearest(VECTORS, VECTORS[indexOf(w)], 1, [indexOf(w)])[0].index);

describe('embedding table', () => {
	it('has 74 distinct words with the published 100-number GloVe vector each', () => {
		expect(D).toBe(100);
		expect(GLOVE_DIM).toBe(100);
		expect(WORDS).toHaveLength(74);
		expect(WORD_LIST).toEqual(WORDS.map((w) => w.word));
		expect(new Set(WORDS.map((w) => w.word)).size).toBe(74);
		for (const w of WORDS) {
			expect(w.vec).toHaveLength(100);
			expect(w.vec).toEqual(GLOVE[w.word]);
			expect(w.vec.every(Number.isFinite)).toBe(true);
		}
		expect(WORDS.every((w) => w.category >= 0 && w.category < CATEGORIES.length)).toBe(true);
	});

	it('uses the vectors raw, not scaled to unit length', () => {
		const lens = VECTORS.map((v) => Math.hypot(...v));
		expect(Math.max(...lens) - Math.min(...lens)).toBeGreaterThan(1);
	});
});

describe('analogies', () => {
	it('lands the four working analogies on their expected word', () => {
		for (const { a, b, c, expected } of ANALOGIES.slice(0, 4)) {
			expect(word(solveAnalogy(a, b, c).index), `${word(a)} ${word(b)} ${word(c)}`).toBe(word(expected));
		}
	});

	it('lands man : boy :: king on queen, not on prince', () => {
		const { a, b, c, expected } = ANALOGIES[4];
		expect([word(a), word(b), word(c), word(expected)]).toEqual(['man', 'boy', 'king', 'prince']);
		const r = solveAnalogy(a, b, c);
		expect(word(r.index)).toBe('queen');
		expect(r.index).not.toBe(expected);
	});

	it('solves most pairs of pairs inside the relations that work', () => {
		const solved = (name: string) => {
			const rel = RELATIONS.find((r) => r.name === name)!;
			let ok = 0;
			let all = 0;
			for (const [a, b] of rel.pairs) {
				for (const [c, d] of rel.pairs) {
					if (a === c) continue;
					all++;
					if (solveAnalogy(a, b, c).index === d) ok++;
				}
			}
			return [ok, all];
		};
		expect(solved('Man to woman')[0]).toBeGreaterThanOrEqual(53);
		expect(solved('Country to capital')).toEqual([30, 30]);
		expect(solved('Present to past')).toEqual([30, 30]);
		expect(solved('Big to bigger')[0]).toBeGreaterThanOrEqual(29);
	});
});

describe('neighbours', () => {
	it('finds the expected nearest word of cat, girl and paris', () => {
		expect(nearestWord('cat')).toBe('dog');
		expect(nearestWord('girl')).toBe('boy');
		expect(nearestWord('paris')).toBe('france');
	});

	it('keeps the nearest neighbour in the same category for at least 70 of 74 words', () => {
		const same = WORDS.filter((w, i) => WORDS[nearest(VECTORS, VECTORS[i], 1, [i])[0].index].category === w.category);
		expect(same.length).toBeGreaterThanOrEqual(70);
	});
});

describe('3D picture', () => {
	it('keeps the farthest point at distance 1 and about a third of the variance', () => {
		const far = Math.max(...POINTS.map((p) => Math.hypot(p[0], p[1], p[2])));
		expect(far).toBeCloseTo(1, 10);
		expect(EXPLAINED).toBeGreaterThan(0.25);
		expect(EXPLAINED).toBeLessThan(0.45);
	});

	it('puts at least 65 of 74 words nearer to their own category centroid than to any other', () => {
		const dist = (p: number[], q: number[]) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
		const own = WORDS.filter((w, i) => {
			const d = dist(POINTS[i], CENTROIDS_3D[w.category]);
			return CATEGORIES.every((_, k) => k === w.category || d < dist(POINTS[i], CENTROIDS_3D[k]));
		});
		expect(own.length).toBeGreaterThanOrEqual(65);
	});

	it('projects a word vector onto its own point', () => {
		for (let i = 0; i < WORDS.length; i++) {
			const p = project3D(VECTORS[i]);
			for (let k = 0; k < 3; k++) expect(p[k]).toBeCloseTo(POINTS[i][k], 9);
		}
	});
});

describe('directions', () => {
	const arrows = (pairs: [number, number][], m: number[][]) => pairs.map(([a, b]) => m[b].map((x, i) => x - m[a][i]));
	const meanOf = (vs: number[][]) => vs[0].map((_, k) => vs.reduce((s, v) => s + v[k], 0) / vs.length);

	it('has a fifth relation, big to bigger', () => {
		expect(RELATIONS.map((r) => r.name)).toEqual([
			'Man to woman',
			'Country to capital',
			'Present to past',
			'Adult to young',
			'Big to bigger'
		]);
	});

	it('gives every arrow of a relation a clear shared direction in all 100 numbers', () => {
		for (const rel of RELATIONS) {
			const vs = arrows(rel.pairs, VECTORS);
			const mean = meanOf(vs);
			for (const v of vs) expect(cosine(v, mean), rel.name).toBeGreaterThan(0.5);
		}
	});

	it('draws big to bigger as long, nearly parallel arrows in the 3D picture', () => {
		const vs = arrows(RELATIONS[4].pairs, POINTS);
		for (const v of vs) expect(Math.hypot(...v)).toBeGreaterThan(0.2);
		for (let i = 0; i < vs.length; i++) for (let j = i + 1; j < vs.length; j++) expect(cosine(vs[i], vs[j])).toBeGreaterThan(0.8);
	});
});

import type { Tone } from '../../../gpu/types.ts';
import { pca3, nearest, analogy } from '../../../ml/embedding.ts';
import type { Mat, Vec } from '../../../ml/vec.ts';
import { GLOVE, GLOVE_DIM } from './vectors.ts';
import { SPEC, WORD_LIST } from './words.ts';

export { WORD_LIST };

/**
 * Real GloVe word vectors (Pennington, Socher and Manning, 2014), "glove-wiki-gigaword-100": 100 numbers per
 * word, learned from Wikipedia and news text by counting which words appear near which. This file keeps the
 * 74 words the lesson shows, with the published numbers unchanged (see vectors.ts). Nothing here is tuned: the
 * vectors are used raw, and the clusters, directions and analogy answers are whatever the data gives.
 */
export const D = GLOVE_DIM;

export const CATEGORIES: { name: string; tone: Tone }[] = [
	{ name: 'People', tone: 'param' },
	{ name: 'Animals', tone: 'ok' },
	{ name: 'Food', tone: 'grad' },
	{ name: 'Places', tone: 'data' },
	{ name: 'Actions', tone: 'neg' },
	{ name: 'Sizes', tone: 'text' }
];

export const WORDS: { word: string; category: number; vec: Vec }[] = SPEC.flatMap((groups, category) =>
	groups.flatMap((g) => g.split('/')).map((word) => ({ word, category, vec: GLOVE[word] }))
);
export const VECTORS: Mat = WORDS.map((w) => w.vec);

export function indexOf(word: string): number {
	return WORDS.findIndex((w) => w.word === word);
}

const pca = pca3(VECTORS);
let farthest = 0;
for (const p of pca.points) farthest = Math.max(farthest, Math.hypot(p[0], p[1], p[2]));
const SCALE = farthest > 0 ? 1 / farthest : 1;

/** The 3D picture: PCA of all vectors, scaled so the farthest point is at distance 1. */
export const POINTS: Mat = pca.points.map((p) => p.map((x) => x * SCALE));
/** Share of the total variance the three picture directions keep. */
export const EXPLAINED = pca.explained[0] + pca.explained[1] + pca.explained[2];
export const PCA_EXPLAINED = pca.explained;

/** Put any 100-number vector (for example the result of an analogy) at its place in the 3D picture. */
export function project3D(v: Vec): [number, number, number] {
	const c = v.map((x, i) => x - pca.mean[i]);
	const r = pca.components.map((comp) => comp.reduce((s, x, i) => s + x * c[i], 0) * SCALE);
	return [r[0], r[1], r[2]];
}

/** Mean position of each category in the 3D picture. */
export const CENTROIDS_3D: Mat = CATEGORIES.map((_, k) => {
	const members = WORDS.map((w, i) => (w.category === k ? POINTS[i] : null)).filter((p) => p !== null);
	return [0, 1, 2].map((a) => members.reduce((s, p) => s + p[a], 0) / members.length);
});

const pair = (a: string, b: string): [number, number] => [indexOf(a), indexOf(b)];

export const RELATIONS: { name: string; pairs: [number, number][] }[] = [
	{
		name: 'Man to woman',
		pairs: [
			pair('man', 'woman'),
			pair('boy', 'girl'),
			pair('king', 'queen'),
			pair('prince', 'princess'),
			pair('father', 'mother'),
			pair('uncle', 'aunt'),
			pair('brother', 'sister'),
			pair('actor', 'actress')
		]
	},
	{
		name: 'Country to capital',
		pairs: [
			pair('france', 'paris'),
			pair('italy', 'rome'),
			pair('japan', 'tokyo'),
			pair('spain', 'madrid'),
			pair('egypt', 'cairo'),
			pair('germany', 'berlin')
		]
	},
	{
		name: 'Present to past',
		pairs: [
			pair('walk', 'walked'),
			pair('run', 'ran'),
			pair('jump', 'jumped'),
			pair('eat', 'ate'),
			pair('sing', 'sang'),
			pair('swim', 'swam')
		]
	},
	{
		name: 'Adult to young',
		pairs: [
			pair('man', 'boy'),
			pair('woman', 'girl'),
			pair('king', 'prince'),
			pair('queen', 'princess'),
			pair('cat', 'kitten'),
			pair('dog', 'puppy')
		]
	},
	{
		name: 'Big to bigger',
		pairs: [
			pair('big', 'bigger'),
			pair('small', 'smaller'),
			pair('fast', 'faster'),
			pair('slow', 'slower'),
			pair('cold', 'colder'),
			pair('hot', 'hotter')
		]
	}
];

/**
 * "a is to b as c is to expected". `expected` is the word a person would answer. For the last entry the real
 * vectors land on a different word (queen, not prince), so the lesson shows the miss.
 */
export const ANALOGIES: { a: number; b: number; c: number; expected: number }[] = [
	{ a: indexOf('man'), b: indexOf('woman'), c: indexOf('king'), expected: indexOf('queen') },
	{ a: indexOf('france'), b: indexOf('paris'), c: indexOf('japan'), expected: indexOf('tokyo') },
	{ a: indexOf('walk'), b: indexOf('walked'), c: indexOf('swim'), expected: indexOf('swam') },
	{ a: indexOf('cat'), b: indexOf('kitten'), c: indexOf('dog'), expected: indexOf('puppy') },
	{ a: indexOf('man'), b: indexOf('boy'), c: indexOf('king'), expected: indexOf('prince') }
];

/** The word an analogy lands next to, skipping its three input words. */
export function solveAnalogy(a: number, b: number, c: number): { index: number; similarity: number } {
	return nearest(VECTORS, analogy(VECTORS[a], VECTORS[b], VECTORS[c]), 1, [a, b, c])[0];
}

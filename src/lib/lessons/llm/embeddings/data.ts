import type { Tone } from '../../../gpu/types.ts';
import { pca3, nearest, analogy } from '../../../ml/embedding.ts';
import { gaussian, mulberry32 } from '../../../ml/rng.ts';
import type { Mat, Vec } from '../../../ml/vec.ts';

/**
 * A hand-built embedding table. Nothing here was learned from text: each vector is a category centroid,
 * plus a base vector shared by words that are "the same thing", plus a few fixed attribute directions
 * (female, royal, young, capital, past, more), plus a little noise.
 */
export const D = 10;
export const SEED = 7;
const CENTROID = 1.3;
const BASE_SD = 0.32;
const ATTR_LEN = 0.75;
const ATTR_SPLIT: [number, number] = [0.75, 0.65];
const ATTR_COEF = 0.5;
const NOISE_SD = 0.04;

export const CATEGORIES: { name: string; tone: Tone }[] = [
	{ name: 'People', tone: 'param' },
	{ name: 'Animals', tone: 'ok' },
	{ name: 'Food', tone: 'grad' },
	{ name: 'Places', tone: 'data' },
	{ name: 'Actions', tone: 'neg' },
	{ name: 'Sizes', tone: 'text' }
];

/** Words by category. A group of words separated by '/' shares one base vector. */
const SPEC: string[][] = [
	[
		'man/woman/boy/girl',
		'king/queen/prince/princess',
		'father/mother',
		'uncle/aunt',
		'brother/sister',
		'actor/actress'
	],
	['cat/kitten', 'dog/puppy', 'lion', 'tiger', 'horse', 'cow', 'wolf', 'mouse', 'bird', 'fish'],
	['apple', 'bread', 'cheese', 'rice', 'pizza', 'soup', 'cake', 'banana', 'coffee', 'tea'],
	['france/paris', 'italy/rome', 'japan/tokyo', 'spain/madrid', 'egypt/cairo', 'germany/berlin'],
	['walk/walked', 'run/ran', 'jump/jumped', 'eat/ate', 'sing/sang', 'swim/swam'],
	['big/bigger', 'small/smaller', 'fast/faster', 'slow/slower', 'cold/colder', 'hot/hotter']
];

const FEMALE = ['woman', 'girl', 'queen', 'princess', 'mother', 'aunt', 'sister', 'actress'];
const MALE = ['man', 'boy', 'king', 'prince', 'father', 'uncle', 'brother', 'actor'];
const ROYAL = ['king', 'queen', 'prince', 'princess'];
const YOUNG = ['boy', 'girl', 'prince', 'princess', 'kitten', 'puppy'];
const CAPITAL = ['paris', 'rome', 'tokyo', 'madrid', 'cairo', 'berlin'];
const PAST = ['walked', 'ran', 'jumped', 'ate', 'sang', 'swam'];
const MORE = ['bigger', 'smaller', 'faster', 'slower', 'colder', 'hotter'];

/** Category order in the octahedron: +/-1.3 along dimension 0, 1, 2. */
const VERTICES: [number, number][] = [
	[0, 1],
	[0, -1],
	[1, 1],
	[1, -1],
	[2, 1],
	[2, -1]
];

function unit(rand: () => number, from: number, to: number): Vec {
	const v = new Array<number>(D).fill(0);
	let s = 0;
	for (let i = from; i < to; i++) {
		v[i] = gaussian(rand);
		s += v[i] * v[i];
	}
	const n = Math.sqrt(s);
	return v.map((x) => x / n);
}

function build(seed: number): { word: string; category: number; vec: Vec }[] {
	const rand = mulberry32(seed);
	const centroids: Vec[] = VERTICES.map(([dim, sign]) => {
		const c = new Array<number>(D).fill(0);
		c[dim] = sign * CENTROID;
		return c;
	});
	const attr = (): Vec => {
		const a = unit(rand, 0, 3);
		const b = unit(rand, 3, D);
		const mix = a.map((x, i) => ATTR_SPLIT[0] * x + ATTR_SPLIT[1] * b[i]);
		const len = Math.hypot(...mix);
		return mix.map((x) => (ATTR_LEN * x) / len);
	};
	const dirs = { female: attr(), royal: attr(), young: attr(), capital: attr(), past: attr(), more: attr() };

	const out: { word: string; category: number; vec: Vec }[] = [];
	SPEC.forEach((groups, category) => {
		for (const group of groups) {
			const base = Array.from({ length: D }, () => BASE_SD * gaussian(rand));
			for (const word of group.split('/')) {
				const vec = centroids[category].map((c, i) => c + base[i]);
				const add = (dir: Vec, k: number) => dir.forEach((x, i) => (vec[i] += k * x));
				if (FEMALE.includes(word)) add(dirs.female, ATTR_COEF);
				if (MALE.includes(word)) add(dirs.female, -ATTR_COEF);
				if (ROYAL.includes(word)) add(dirs.royal, 1);
				if (YOUNG.includes(word)) add(dirs.young, 1);
				if (CAPITAL.includes(word)) add(dirs.capital, 1);
				if (PAST.includes(word)) add(dirs.past, 1);
				if (MORE.includes(word)) add(dirs.more, 1);
				for (let i = 0; i < D; i++) vec[i] += NOISE_SD * gaussian(rand);
				out.push({ word, category, vec });
			}
		}
	});
	return out;
}

export const WORDS: { word: string; category: number; vec: Vec }[] = build(SEED);
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

/** Put any 10-number vector (for example the result of an analogy) at its place in the 3D picture. */
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
	}
];

/** "a is to b as c is to expected". */
export const ANALOGIES: { a: number; b: number; c: number; expected: number }[] = [
	{ a: indexOf('man'), b: indexOf('woman'), c: indexOf('king'), expected: indexOf('queen') },
	{ a: indexOf('france'), b: indexOf('paris'), c: indexOf('japan'), expected: indexOf('tokyo') },
	{ a: indexOf('walk'), b: indexOf('walked'), c: indexOf('swim'), expected: indexOf('swam') },
	{ a: indexOf('cat'), b: indexOf('kitten'), c: indexOf('dog'), expected: indexOf('puppy') }
];

/** The word an analogy lands next to, skipping its three input words. */
export function solveAnalogy(a: number, b: number, c: number): { index: number; similarity: number } {
	return nearest(VECTORS, analogy(VECTORS[a], VECTORS[b], VECTORS[c]), 1, [a, b, c])[0];
}

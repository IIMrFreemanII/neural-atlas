import { attention } from '../../../ml/attention.ts';
import { mulberry32 } from '../../../ml/rng.ts';
import { matVec, type Mat } from '../../../ml/vec.ts';

/** Names of the six meaning features of a token vector. */
export const FEATURES = ['animal', 'thing', 'pronoun', 'verb', 'fits a thing', 'fits an animal'];

/** Strength of the per-word signature added to the meaning features. */
const SIGNATURE = 0.15;
/** Seed of the per-word signature, tuned so the spec holds. */
export const SIGNATURE_SEED = 360;
/** Query gains of the three heads. */
export const G0 = 3.5;
export const G1 = 5;
export const G2 = 18;
/** Position frequencies. */
const FA = 0.7;
const FB = 1.9;

const MEANING: Record<string, number[]> = {
	the: [0, 0, 0, 0, 0, 0],
	dog: [1, 0, 0, 0, 0, 0],
	lost: [0, 0, 0, 1, 0, 0],
	ball: [0, 1, 0, 0, 0, 0],
	when: [0, 0, 0, 0, 0, 0],
	it: [0, 0, 1, 0, 0, 0],
	was: [0, 0, 0, 0.6, 0, 0],
	wet: [0, 0, 0, 0, 1, 0],
	tired: [0, 0, 0, 0, 0, 1]
};

const WORDS = Object.keys(MEANING);

/** One fixed vector of plus/minus 1 values per distinct word. Same word, same signature. */
function signatures(seed: number): Record<string, number[]> {
	const rnd = mulberry32(seed);
	const out: Record<string, number[]> = {};
	for (const w of WORDS) out[w] = Array.from({ length: 6 }, () => (rnd() < 0.5 ? -1 : 1));
	return out;
}

export interface Sentence {
	tokens: string[];
	/** One row per token, 10 numbers: 6 meaning features then 4 position features. */
	x: Mat;
}

function makeSentence(text: string, sig: Record<string, number[]>): Sentence {
	const tokens = text.split(' ');
	const x = tokens.map((w, p) => [
		...MEANING[w].map((m, i) => m + SIGNATURE * sig[w][i]),
		Math.sin(FA * p),
		Math.cos(FA * p),
		Math.sin(FB * p),
		Math.cos(FB * p)
	]);
	return { tokens, x };
}

/** Builds both sentences and the three heads. Exposed so the constants can be tuned. */
export function buildModel(seed: number, g0: number, g1: number, g2: number): { sentences: Sentence[]; heads: Head[] } {
	const sig = signatures(seed);
	return {
		sentences: [
			makeSentence('the dog lost the ball when it was wet', sig),
			makeSentence('the dog lost the ball when it was tired', sig)
		],
		heads: [head0(g0), head1(g1), head2(g2)]
	};
}

export interface Head {
	name: string;
	/** 4 x 10 */
	wq: Mat;
	/** 4 x 10 */
	wk: Mat;
	/** 6 x 10 */
	wv: Mat;
}

const D = 10;
const zeros = (r: number, c: number): Mat => Array.from({ length: r }, () => new Array<number>(c).fill(0));
const wvCopy = (): Mat => {
	const m = zeros(6, D);
	for (let i = 0; i < 6; i++) m[i][i] = 1;
	return m;
};

function head0(G0: number): Head {
	const wq = zeros(4, D);
	wq[0][5] = G0;
	wq[0][2] = 0.5 * G0;
	wq[1][4] = G0;
	wq[1][2] = 0.5 * G0;
	const wk = zeros(4, D);
	wk[0][0] = 1;
	wk[1][1] = 1;
	return { name: 'Describes what?', wq, wk, wv: wvCopy() };
}

function head1(G1: number): Head {
	const wq = zeros(4, D);
	// Rotate the position features back by one position: sin(a(p-1)), cos(a(p-1)) as a linear map of sin(ap), cos(ap).
	const rot = (row: number, col: number, f: number) => {
		wq[row][col] = G1 * Math.cos(f);
		wq[row][col + 1] = -G1 * Math.sin(f);
		wq[row + 1][col] = G1 * Math.sin(f);
		wq[row + 1][col + 1] = G1 * Math.cos(f);
	};
	rot(0, 6, FA);
	rot(2, 8, FB);
	const wk = zeros(4, D);
	for (let i = 0; i < 4; i++) wk[i][6 + i] = 1;
	return { name: 'Previous word', wq, wk, wv: wvCopy() };
}

function head2(G2: number): Head {
	const wq = zeros(4, D);
	const wk = zeros(4, D);
	for (let i = 0; i < 4; i++) {
		wq[i][i] = G2;
		wk[i][i] = 1;
	}
	return { name: 'Same word', wq, wk, wv: wvCopy() };
}


const MODEL = buildModel(SIGNATURE_SEED, G0, G1, G2);
export const SENTENCES: Sentence[] = MODEL.sentences;
export const HEADS: Head[] = MODEL.heads;

export interface HeadRun {
	q: Mat;
	k: Mat;
	v: Mat;
	scores: Mat;
	weights: Mat;
	out: Mat;
}

/** Runs attention() with scale = sharp (the gains are already inside the weights). */
export function runHead(s: Sentence, h: Head, opts: { causal: boolean; sharp: number }): HeadRun {
	const q = s.x.map((x) => matVec(h.wq, x));
	const k = s.x.map((x) => matVec(h.wk, x));
	const v = s.x.map((x) => matVec(h.wv, x));
	const r = attention(q, k, v, { causal: opts.causal, scale: opts.sharp });
	return { q, k, v, ...r };
}

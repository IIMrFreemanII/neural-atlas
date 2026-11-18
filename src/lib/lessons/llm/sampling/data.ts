import { nextLogits } from '../../../ml/gpt.ts';
import { END, encode } from '../../../ml/wordtok.ts';
import type { Lm } from '../shared/lm.ts';

/** Starts the learner can choose. Every word is in the model's vocabulary. */
/** The second start is one the model is nearly sure about (the next word is almost always 'to'); the other two leave it many choices. */
export const PROMPTS = ['once upon a time there was a', 'one day , lily went', 'the little girl wanted to'];

/** Token ids of a prompt. */
export function promptIds(lm: Lm, index: number): number[] {
	const i = Math.min(PROMPTS.length - 1, Math.max(0, Math.round(index)));
	return encode(lm.vocab, PROMPTS[i]);
}

/**
 * The model's scores for the next token after `ids`, as a plain array of length vocab.
 * The score of '<unk>' is set to -Infinity, so an unknown word is never offered or picked.
 */
export function nextScores(lm: Lm, ids: readonly number[]): number[] {
	const scores = Array.from(nextLogits(lm.model, ids));
	scores[0] = -Infinity;
	return scores;
}

/** How a token is written on screen: '<end>' reads 'end of story', everything else is its text. */
export function tokenText(lm: Lm, id: number): string {
	return id === END ? 'end of story' : lm.vocab.words[id];
}

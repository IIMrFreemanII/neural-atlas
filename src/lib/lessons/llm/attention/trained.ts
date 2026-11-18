import { trace } from '../../../ml/gpt.ts';
import { encode } from '../../../ml/wordtok.ts';
import type { Lm } from '../shared/lm.ts';

/** Sentences shown in the trained-model step. Every word is in the model's vocabulary. */
export const TRAINED_TEXTS = [
	'the dog lost the ball when it was wet',
	'tom lost his ball . he was very sad',
	'the little girl saw a big dog and she was scared'
];

export interface TrainedSentence {
	tokens: string[];
	/** weights[block][head][i][j]: how much token i looks at token j. Rows sum to 1; 0 for j > i. */
	weights: number[][][][];
}

/** Runs the trained model on each sentence and keeps the attention weights of every block and head. */
export function buildTrained(lm: Lm): TrainedSentence[] {
	return TRAINED_TEXTS.map((text) => {
		const ids = encode(lm.vocab, text);
		const t = trace(lm.model, ids);
		return {
			tokens: ids.map((id) => lm.vocab.words[id]),
			weights: t.layers.map((layer) => layer.heads.map((m) => m.map((row) => row.slice())))
		};
	});
}

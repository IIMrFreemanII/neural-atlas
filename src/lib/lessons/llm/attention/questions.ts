import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';
import { HEADS, SENTENCES, runHead } from './model.ts';

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'The last word is "wet", and its thickest arc goes to "ball". Now you switch the last word to "tired". Where does its thickest arc go?\n\nDecide first, then check by switching the sentence.',
		params: { scenario: 2, sentence: 0, head: 0, causal: 0, sharp: 1, arcs: 1, matrix: 0, mech: 0, token: 8 },
		controls: ['sentence'],
		options: [
			{
				text: 'To "ball" again, because "ball" is still where it was.',
				why: 'The arcs follow what the words ask for and offer, not where they stand. "wet" asks for a thing, and "tired" asks for something else.'
			},
			{
				text: 'To "was", the word just before it.',
				why: 'Looking at the previous word is the job of another head. This head matches what a word asks for with what another offers, and "was" offers a verb.'
			},
			{
				text: 'To "it", because "it" is a pronoun that points at something.',
				why: '"it" offers being a pronoun, not being an animal. "tired" gives most of its attention to the word that offers what it asks for.'
			},
			{
				text: 'To "dog", because a tired thing is an animal.',
				correct: true,
				why: '"tired" makes a query that asks for an animal, and "dog" is the token whose key offers being an animal. The two line up, so "dog" gets about 80% of the attention and "ball" gets 1%.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'Look from the last word, "tired". Using only **Sharpness**, make its attention less decisive: "dog" must get **less than 40%** of it and still have the thickest arc.',
		params: { scenario: 4, sentence: 1, head: 0, causal: 0, sharp: 1, arcs: 1, matrix: 0, mech: 0, token: 8 },
		controls: ['sharp'],
		goal: ({ params: p }) => {
			const run = runHead(SENTENCES[Math.round(p.sentence)], HEADS[Math.round(p.head)], {
				causal: p.causal >= 0.5,
				sharp: p.sharp
			});
			const row = run.weights[SENTENCES[Math.round(p.sentence)].tokens.length - 1];
			return shown(row[1] * 100, 0) < 40 && row.every((w) => w <= row[1]);
		},
		hint: 'Sharpness multiplies every score before softmax turns them into shares. Think about what happens to the shares when the scores are pulled closer together.',
		solved:
			'A smaller factor shrinks the gaps between the scores, so softmax shares the 100% more evenly. At a sharpness of 0.40, "dog" gets 34%. It is still the best match, so it keeps the largest share.',
		example: { params: { sharp: 0.4 } }
	},
	{
		kind: 'choice',
		prompt:
			'Look from the first word, "the". It spreads its attention over all nine words. You turn **Hide later words** on. What does its attention become?\n\nDecide first, then check with the toggle.',
		params: { scenario: 8, sentence: 0, head: 0, causal: 0, sharp: 1, arcs: 1, matrix: 1, mech: 0, token: 0 },
		controls: ['causal'],
		options: [
			{
				text: 'It goes blank, because no word comes before it.',
				why: 'The shares still add up to 100%. A token may look at itself, so it is never left with nothing.'
			},
			{
				text: 'It moves to "dog", the closest word after it.',
				why: 'Later words are exactly the ones that get hidden. "dog" comes after "the", so it is out of sight.'
			},
			{
				text: 'All of it, 100%, lands on "the" itself.',
				correct: true,
				why: 'Only "the" is left in view, the mask sets the hidden words to 0, and the shares of what remains must still add up to 100%.'
			},
			{
				text: 'It stays the same, because the first word comes before everything else.',
				why: 'Every other word comes after it, so all eight of them are hidden. Its attention to them goes to 0 and the rest is shared out again.'
			}
		]
	}
];

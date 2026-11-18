import type { LessonDef } from '../../types.ts';
import { createAttentionScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'attention',
	track: 'llm',
	title: 'Self-attention',
	summary: 'See how every token looks at the other tokens, and how that lets a model work out what a word refers to.',
	defaults: { sentence: 0, head: 0, causal: 0, sharp: 1, arcs: 1, mech: 0, matrix: 0, token: 8, scenario: 1 },
	discrete: ['arcs', 'mech', 'matrix', 'token', 'scenario'],
	controls: [
		{
			kind: 'select',
			key: 'sentence',
			label: 'Last word',
			options: [
				{ value: 0, label: 'wet' },
				{ value: 1, label: 'tired' }
			]
		},
		{
			kind: 'select',
			key: 'head',
			label: 'Head',
			options: [
				{ value: 0, label: 'Describes what?' },
				{ value: 1, label: 'Previous word' },
				{ value: 2, label: 'Same word' }
			]
		},
		{ kind: 'toggle', key: 'causal', label: 'Hide later words' },
		{ kind: 'slider', key: 'sharp', label: 'Sharpness', min: 0.25, max: 3, step: 0.05, tone: 'data', digits: 2 }
	],
	readouts: [
		{ key: 'from', label: 'Looking from', tone: 'ok' },
		{ key: 'to', label: 'Looks most at', tone: 'data' }
	],
	legend: [
		{ tone: 'ok', label: 'Query' },
		{ tone: 'param', label: 'Key' },
		{ tone: 'data', label: 'Value, attention' }
	],
	steps: [
		{
			title: 'A word alone is not enough',
			body: 'Read the sentence: the dog lost the ball when it was wet. You know at once what is wet. The model has only the vector for "wet", and that vector is the same in every sentence.\n\nTo work out what "wet" describes, the tokens have to **exchange information**.',
			params: { scenario: 1, arcs: 0, token: 8 },
			controls: []
		},
		{
			title: 'Each token looks at the others',
			body: 'The arcs show how much "wet" looks at each word. The shares add up to 100%, and the thickest arc goes to the word it cares about most. Click another word to look from it, then switch the last word to "tired".\n\nThe numbers here were set by hand so the pattern is easy to read. A real model learns them from text.',
			params: { scenario: 2, arcs: 1, token: 8 },
			controls: ['sentence']
		},
		{
			title: 'Queries and keys',
			body: 'Every token makes a [[ok:query]], what it is looking for, and a [[param:key]], what it offers. Each is a short list of numbers.\n\n"wet" asks for a thing, and "ball" offers being a thing. When a query and a key line up, that token is worth looking at.',
			params: { scenario: 3, mech: 1, token: 8 },
			controls: ['sentence']
		},
		{
			title: 'Match, then share out 100%',
			body: 'The score is how well the [[ok:query]] matches each [[param:key]]. Softmax turns the scores into shares that add up to 100%, and the best match takes most of it.\n\nDrag **Sharpness** to make the choice more decisive or more even.',
			params: { scenario: 4, mech: 3, token: 8 },
			controls: ['sharp', 'sentence'],
			deepDive: {
				note: 'The score of token i for token j is the dot product of query i and key j. The shares are a softmax over j. Here the Sharpness slider is the factor s in front of the scores; real models use 1 / sqrt(d).',
				math: [
					{ tex: 'e_{ij} = s \\, q_i \\cdot k_j', caption: 'score' },
					{ tex: '\\alpha_{ij} = \\frac{e^{e_{ij}}}{\\sum_{m} e^{e_{im}}}', caption: 'share (softmax)' }
				]
			}
		},
		{
			title: 'Gather the values',
			body: 'Each token also offers a [[data:value]], the information it passes on. The output is the average of all values, weighted by the shares.\n\nSo the new vector for "wet" now carries "thing" from "ball". Hover the small cells to see which feature each one is.',
			params: { scenario: 5, mech: 4, token: 8 },
			controls: ['sentence'],
			deepDive: {
				note: 'The output for token i is the sum of the values, weighted by the shares. Stacking all tokens as rows of matrices gives the whole layer in one line.',
				math: [
					{ tex: 'o_i = \\sum_{j} \\alpha_{ij} \\, v_j', caption: 'output for one token' },
					{ tex: '\\mathrm{Attention}(Q, K, V) = \\mathrm{softmax}\\!\\left(\\frac{Q K^{\\top}}{\\sqrt{d}}\\right) V', caption: 'all tokens at once' }
				]
			}
		},
		{
			title: 'Every token does this at once',
			body: 'Here is the whole thing as a grid. Each row is a token looking, and each column is a token being looked at. Hover a row to follow it.\n\n"it" splits between "dog" and "ball". Words with nothing to ask spread their attention out evenly.',
			params: { scenario: 6, matrix: 1, token: 6 },
			controls: ['sentence']
		},
		{
			title: 'Different heads, different jobs',
			body: 'A layer runs several **heads** side by side, each with its own weights. Switch **Head** and watch the grid change: one head looks at the previous word, another at repeats of the same word.\n\nEach head asks a different question about the same tokens.',
			params: { scenario: 7, matrix: 1, head: 1, token: 4 },
			controls: ['head', 'sentence'],
			deepDive: {
				note: 'The toy uses whole words as tokens so the grid is easy to read. Real models use subword tokens. The three heads here were designed by hand. In a trained model the heads are found by training, and most of them do jobs that are harder to name.'
			}
		},
		{
			title: 'No peeking ahead',
			body: 'A model that predicts the next token must not see it. So each token may only look at itself and earlier tokens, and the upper triangle goes dark.\n\nTurn **Hide later words** on and off and see how the rows change.',
			params: { scenario: 8, matrix: 1, head: 0, causal: 1, token: 8 },
			controls: ['causal', 'head']
		}
	],
	createScene: createAttentionScene
};

export default lesson;

import type { LessonDef } from '../../types.ts';
import { createTransformerBlockScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'transformer-block',
	track: 'llm',
	title: 'Inside a transformer block',
	summary: 'Follow one block of a small trained transformer step by step: normalize, attention, add, feed-forward, add. Then stack four of them.',
	defaults: { focus: 1, depth: 1, zoom: 0, column: 4, scenario: 1 },
	discrete: ['focus', 'column', 'scenario'],
	controls: [
		{ kind: 'action', id: 'play', label: 'Send the tokens through', icon: 'play' },
		{ kind: 'slider', key: 'depth', label: 'Blocks', min: 1, max: 4, step: 1, tone: 'param', digits: 0 }
	],
	readouts: [
		{ key: 'inspect', label: 'Inspecting' }
	],
	legend: [
		{ tone: 'data', label: 'Vectors' },
		{ tone: 'param', label: 'Layers' },
		{ tone: 'ok', label: 'Update' }
	],
	steps: [
		{
			title: 'One lane per token',
			body: 'Each token of "she loved to play with" gets its own lane. Its [[data:vector]] of 64 numbers travels up the lane, drawn as an 8 by 8 tile of cells: cyan for positive numbers, rose for negative, brighter for larger. This running vector is called the residual stream, and a block reads it and writes small changes back.\n\nThese are the real weights of a small transformer trained on simple children\'s stories. Click a lane to inspect it, then press **Send the tokens through**.',
			params: { focus: 1, zoom: 0, scenario: 1 },
			controls: ['play']
		},
		{
			title: 'Attention moves information between lanes',
			body: '[[param:Self-attention]] is the only place where tokens talk to each other. Each token gathers from itself and the tokens before it, and the [[data:arcs]] show how much from each. This is the attention from the previous lesson. This block has 4 heads, and the arcs show their average.\n\nClick different lanes. The first token can only look at itself.',
			params: { focus: 2, zoom: 0, scenario: 2 },
			controls: ['play']
		},
		{
			title: 'Add, do not replace',
			body: 'A layer does not overwrite its lane. Its output is added to what was already there: before + [[ok:update]] = after.\n\nThe bypass line carries the old vector around the layer. Old information survives unless an update writes over it.',
			params: { focus: 3, zoom: 0, scenario: 3 },
			controls: [],
			deepDive: {
				note: 'Every layer in a block is wrapped the same way: normalize, run the layer, add the result back to the stream.',
				math: [{ tex: 'x \\leftarrow x + \\mathrm{Attention}(\\mathrm{LayerNorm}(x))', caption: 'The attention half of the block.' }]
			}
		},
		{
			title: 'The feed-forward layer thinks about each token alone',
			body: 'The [[param:feed-forward layer]] stretches each lane\'s vector to 4 times as many numbers (256), passes them through a nonlinearity, and squeezes them back to 64. The result is an [[ok:update]] to add.\n\nNo information moves between lanes here. Every token is handled on its own.',
			params: { focus: 4, zoom: 0, scenario: 4 },
			controls: [],
			deepDive: {
				math: [
					{ tex: 'x \\leftarrow x + W_2\\,\\mathrm{GELU}\\big(W_1\\,\\mathrm{LayerNorm}(x) + b_1\\big) + b_2', caption: 'The feed-forward half of the block.' },
					{ tex: '\\mathrm{GELU}(z) \\approx \\tfrac{z}{2}\\Big(1 + \\tanh\\big(\\sqrt{2/\\pi}\\,(z + 0.044715\\,z^3)\\big)\\Big)', caption: 'A smooth version of ReLU.' }
				]
			}
		},
		{
			title: 'Normalize first',
			body: 'Before each layer, the vector is rescaled to mean 0 and spread 1. That keeps the numbers in a workable range, however deep the stack gets.\n\nCompare the **before** and **after** bars for the lane you picked, each drawn to fill its box. The ups and downs keep their order. Only the center and the scale change, and the spread numbers show by how much.',
			params: { focus: 5, zoom: 0, scenario: 5 },
			controls: [],
			deepDive: {
				math: [
					{
						tex: '\\mathrm{LayerNorm}(x)_i = \\dfrac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}, \\quad \\mu = \\tfrac{1}{d}\\sum_i x_i, \\quad \\sigma^2 = \\tfrac{1}{d}\\sum_i (x_i - \\mu)^2',
						caption: 'In this model, layer norm has no learned gain or bias, and a token is a whole word.'
					}
				]
			}
		},
		{
			title: 'Stack the same block several times',
			body: 'A model repeats this same structure, with different [[param:weights]] in every block. Each block reads the [[data:stream]] left by the one below and refines it. This model stacks 4 blocks. GPT-2 small stacks 12, and large models stack dozens.\n\nDrag **Blocks** from 1 to 4 and watch the top vectors change.',
			params: { focus: 6, zoom: 1, depth: 2, scenario: 6 },
			controls: ['depth', 'play']
		},
		{
			title: 'Back to words',
			body: 'At the top, the vector of the last lane is compared against the vector of every word the model knows (2,048 here). The best matches are the likely [[data:next tokens]]. The bars show the model\'s own probabilities for its 8 likeliest words, and the text under them gives the share of all other words.\n\nDrag **Blocks** down to 1. With fewer blocks you read the stream early, which the model was not trained for, and its first guess is "his", a word that often follows "play with" in its stories. Only the full stack settles on "her", the word that fits "she". The next lesson is about choosing from these probabilities.',
			params: { focus: 7, zoom: 1, depth: 4, scenario: 7 },
			controls: ['depth', 'play'],
			deepDive: {
				note: 'The last lane\'s final vector is normalized, multiplied by one matrix that has a row per vocabulary word, and turned into probabilities with a softmax.',
				math: [{ tex: 'p = \\mathrm{softmax}\\big(W_U\\,\\mathrm{LayerNorm}(h_{\\text{last}})\\big)', caption: 'W_U has one row per token in the vocabulary.' }]
			}
		}
	],
	createScene: createTransformerBlockScene
};

export default lesson;

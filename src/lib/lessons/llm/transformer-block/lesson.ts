import type { LessonDef } from '../../types.ts';
import { createTransformerBlockScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'transformer-block',
	track: 'llm',
	title: 'Inside a transformer block',
	summary: 'Follow one block step by step: normalize, attention, add, feed-forward, add. Then stack many of them.',
	defaults: { focus: 1, depth: 1, zoom: 0, column: 4, scenario: 1 },
	discrete: ['focus', 'column', 'scenario'],
	controls: [
		{ kind: 'action', id: 'play', label: 'Send the tokens through', icon: 'play' },
		{ kind: 'slider', key: 'depth', label: 'Blocks', min: 1, max: 12, step: 1, tone: 'param', digits: 0 }
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
			body: 'Each token of "the cat sat on the" gets its own lane. Its [[data:vector]] of 8 numbers travels up the lane. This running vector is called the residual stream, and a block reads it and writes small changes back.\n\nThe weights here are random, so the numbers are real arithmetic but carry no meaning. A real model learns them from text. Click a lane to inspect it, then press **Send the tokens through**.',
			params: { focus: 1, zoom: 0, scenario: 1 },
			controls: ['play']
		},
		{
			title: 'Attention moves information between lanes',
			body: '[[param:Self-attention]] is the only place where tokens talk to each other. Each token gathers from itself and the tokens before it, and the [[data:arcs]] show how much from each. This is the attention from the previous lesson, with a single head.\n\nClick different lanes. The first token can only look at itself.',
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
			body: 'The [[param:feed-forward layer]] stretches each lane\'s vector to 4 times as many numbers, passes them through a nonlinearity, and squeezes them back to 8. The result is an [[ok:update]] to add.\n\nNo information moves between lanes here. Every token is handled on its own.',
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
			body: 'Before each layer, the vector is rescaled to mean 0 and spread 1. That keeps the numbers in a workable range, however deep the stack gets.\n\nCompare the **before** and **after** bars for the lane you picked. The ups and downs keep their order. Only the center and the scale change.',
			params: { focus: 5, zoom: 0, scenario: 5 },
			controls: [],
			deepDive: {
				math: [
					{
						tex: '\\mathrm{LayerNorm}(x)_i = \\dfrac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}, \\quad \\mu = \\tfrac{1}{d}\\sum_i x_i, \\quad \\sigma^2 = \\tfrac{1}{d}\\sum_i (x_i - \\mu)^2',
						caption: 'This toy version has no learned gain or bias.'
					}
				]
			}
		},
		{
			title: 'Stack the same block many times',
			body: 'A model repeats this same structure, with different [[param:weights]] in every block. Each block reads the [[data:stream]] left by the one below and refines it. GPT-2 small stacks 12 blocks, and large models stack dozens.\n\nDrag **Blocks** and watch the top vectors change.',
			params: { focus: 6, zoom: 1, depth: 4, scenario: 6 },
			controls: ['depth', 'play']
		},
		{
			title: 'Back to words',
			body: 'At the top, the vector of the last lane is compared against every word the model knows. That gives a probability for each possible [[data:next token]].\n\nIn this toy the output weights were set by hand so the answer is sensible, and it gets surer as the stack gets taller. The next lesson is about choosing from these probabilities.',
			params: { focus: 7, zoom: 1, depth: 12, scenario: 7 },
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

import type { LessonDef } from '../../types.ts';
import { createForwardPassScene } from './scene.ts';

const FORWARD_CODE = `type Layer = { W: number[][]; b: number[]; f: (z: number) => number };

function forward(layers: Layer[], x: number[]): number[] {
  let a = x;
  for (const { W, b, f } of layers) {
    // one matrix multiply per layer: every neuron at once
    a = W.map((row, j) => f(row.reduce((s, w, i) => s + w * a[i], b[j])));
  }
  return a;
}`;

const lesson: LessonDef = {
	id: 'forward-pass',
	track: 'fundamentals',
	title: 'Forward pass',
	summary:
		'Follow numbers through a small network, layer by layer, until they come out as a prediction.',
	defaults: { x1: 0.8, x2: -0.5, wave: 3, focus: -1, matrix: 0, layer: 1 },
	discrete: ['focus'],
	controls: [
		{ kind: 'action', id: 'send', label: 'Send signal', icon: 'play' },
		{ kind: 'slider', key: 'x1', label: 'Input x₁', min: -1, max: 1, step: 0.05, tone: 'data' },
		{ kind: 'slider', key: 'x2', label: 'Input x₂', min: -1, max: 1, step: 0.05, tone: 'data' },
		{
			kind: 'select',
			key: 'layer',
			label: 'Layer',
			options: [
				{ value: 1, label: '1' },
				{ value: 2, label: '2' },
				{ value: 3, label: '3' }
			]
		}
	],
	readouts: [{ key: 'out', label: 'Output ŷ', tone: 'data', digits: 2 }],
	legend: [
		{ tone: 'data', label: 'Activation' },
		{ tone: 'param', label: 'Weight' },
		{ tone: 'neg', label: 'Negative value' }
	],
	steps: [
		{
			title: 'A network is layers of neurons',
			body: `This network has an input layer on the left, two hidden layers in the middle and one output neuron on the right. The [[data:inputs]] are just two numbers.

Every neuron connects to every neuron in the next layer. Each connection has a [[param:weight]], and the weights are what the network knows.`,
			params: { wave: 0 },
			controls: []
		},
		{
			title: 'Signals move left to right',
			body: `A layer can only compute after the layer before it has finished. So the [[data:values]] sweep from the inputs through the hidden layers to the output.

This single sweep is the **forward pass**. Press the button to watch it happen again.`,
			params: { wave: 3 },
			controls: ['send']
		},
		{
			title: 'Each neuron uses the same recipe',
			body: `Take one neuron of the first hidden layer. It multiplies each input by its [[param:weight]], adds them up, adds a bias, and squeezes the sum through the activation.

That is exactly the recipe from the neuron lesson. Move the inputs and watch the numbers change.`,
			params: { wave: 1, focus: 1 },
			controls: ['x1', 'x2']
		},
		{
			title: 'A layer is one matrix multiply',
			body: `All the [[param:weights]] into a layer fit in a grid: one row per neuron, one column per input. Multiplying the grid by the input values gives every neuron's sum at once.

Add the biases, apply the activation, and the whole layer is done. Doing this in one step is why GPUs are so good at neural networks.`,
			params: { wave: 1, matrix: 1, layer: 1 },
			controls: ['x1', 'x2'],
			deepDive: {
				math: [
					{
						tex: '\\mathbf{a}^{(1)} = f\\left(W^{(1)}\\mathbf{x} + \\mathbf{b}^{(1)}\\right)',
						caption: 'One layer: weights times inputs, plus biases, through the activation f.'
					}
				],
				code: { lang: 'ts', source: FORWARD_CODE }
			}
		},
		{
			title: 'Layers feed layers',
			body: `The output of one layer is the input of the next. Switch the layer and see that the [[data:activations]] coming out of one grid are the ones going into the next.

Deeper layers combine the simple features found earlier into richer ones.`,
			params: { wave: 3, matrix: 1, layer: 2 },
			controls: ['layer', 'send']
		},
		{
			title: 'The output is a prediction',
			body: `The last neuron uses a sigmoid, which squeezes its sum into a number between 0 and 1. You can read that number as a score or a probability.

These weights are random, so the prediction means nothing yet. [[param:Training]] will change that by adjusting the weights.`,
			params: { wave: 3 },
			controls: ['x1', 'x2', 'send']
		},
		{
			title: 'Explore the pass',
			body: `Change the inputs and watch which paths carry the strongest signal. Some [[neg:negative]] values appear along the way, and the output shifts with them.

Switch between layers to see each weight grid and how it turns its inputs into the next layer.`,
			params: { wave: 3, matrix: 1, layer: 1 }
		}
	],
	createScene: createForwardPassScene
};

export default lesson;

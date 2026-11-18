import type { LessonDef } from '../../types.ts';
import { questions } from './questions.ts';
import { createNeuronScene } from './scene.ts';

const NEURON_CODE = `function neuron(x: number[], w: number[], b: number): number {
  let z = b;
  for (let i = 0; i < x.length; i++) z += w[i] * x[i];
  return z;
}`;

const ACTIVATE_CODE = `function activate(id: 0 | 1 | 2, z: number): number {
  if (id === 1) return 1 / (1 + Math.exp(-z)); // sigmoid
  if (id === 2) return Math.max(0, z);          // ReLU
  return z;                                     // linear
}`;

const ALL = ['x1', 'x2', 'w1', 'w2', 'b', 'act'];
const FULL = { showX2: 1, showBias: 1, showAct: 1 };

const lesson: LessonDef = {
	id: 'neuron',
	track: 'fundamentals',
	title: 'The neuron',
	summary:
		'A neuron weighs its inputs, adds a bias and passes the sum through an activation to produce an output.',
	defaults: { x1: 0.8, x2: -0.5, w1: 1, w2: 0.6, b: 0, act: 0, showX2: 0, showBias: 0, showAct: 0 },
	controls: [
		{ kind: 'slider', key: 'x1', label: 'Input x₁', min: -1, max: 1, step: 0.05, tone: 'data' },
		{ kind: 'slider', key: 'x2', label: 'Input x₂', min: -1, max: 1, step: 0.05, tone: 'data' },
		{ kind: 'slider', key: 'w1', label: 'Weight w₁', min: -2, max: 2, step: 0.05, tone: 'param' },
		{ kind: 'slider', key: 'w2', label: 'Weight w₂', min: -2, max: 2, step: 0.05, tone: 'param' },
		{ kind: 'slider', key: 'b', label: 'Bias b', min: -2, max: 2, step: 0.05, tone: 'param' },
		{
			kind: 'select',
			key: 'act',
			label: 'Activation',
			options: [
				{ value: 0, label: 'Linear' },
				{ value: 1, label: 'Sigmoid' },
				{ value: 2, label: 'ReLU' }
			]
		}
	],
	readouts: [
		{ key: 'z', label: 'Sum z', tone: 'data', digits: 2 },
		{ key: 'out', label: 'Output', tone: 'data', digits: 2 }
	],
	legend: [
		{ tone: 'data', label: 'Signal value' },
		{ tone: 'param', label: 'Weight and bias' },
		{ tone: 'neg', label: 'Negative value' }
	],
	steps: [
		{
			title: 'A neuron is a tiny function',
			body: 'A [[data:neuron]] is a tiny function: one number goes in and one number comes out. This one simply passes its input along.\n\nDrag the input and watch the output follow it. Every big network in this atlas is built from many copies of this small idea.',
			params: {},
			controls: ['x1']
		},
		{
			title: 'Weights scale the signal',
			body: 'The input travels along a connection that carries a [[param:weight]]. The neuron multiplies the input by it, so a bigger weight strengthens the signal and a weight near zero ignores the input.\n\nA [[neg:negative]] weight flips the sign of the signal. Pink marks negative values, cyan marks positive ones.',
			params: { w1: 1.5 },
			controls: ['x1', 'w1']
		},
		{
			title: 'Many inputs, one sum',
			body: 'Real neurons take many inputs. Each [[data:input]] has its own [[param:weight]], and the neuron adds all the weighted inputs into a single number called **z**.\n\nChange one input or one weight and watch how the sum moves. A larger weight makes that input matter more.',
			params: { showX2: 1 },
			controls: ['x1', 'x2', 'w1', 'w2'],
			deepDive: {
				math: [
					{ tex: 'z = w_1 x_1 + w_2 x_2', caption: 'Two inputs, two weights.' },
					{ tex: 'z = \\sum_i w_i x_i', caption: 'Any number of inputs.' }
				],
				code: { lang: 'ts', source: NEURON_CODE }
			}
		},
		{
			title: 'Bias shifts the baseline',
			body: 'The [[param:bias]] is a number added to the sum no matter what the inputs are. It shifts the baseline, which sets how easily the neuron responds.\n\nA positive bias starts the sum above zero. A [[neg:negative]] bias means the inputs have to work harder to lift it.',
			params: { showX2: 1, showBias: 1, b: 0.5 },
			controls: ['x1', 'x2', 'w1', 'w2', 'b'],
			deepDive: {
				math: [{ tex: 'z = \\sum_i w_i x_i + b', caption: 'The full weighted sum.' }]
			}
		},
		{
			title: 'An activation decides what passes',
			body: 'The sum z then goes through an **activation function**. ReLU passes a positive sum unchanged and outputs zero for a [[neg:negative]] one. Move the inputs until z crosses zero and watch the output switch off.\n\nWithout a non-linear activation, stacked neurons collapse into one linear function.',
			params: { ...FULL, act: 2, b: 0 },
			controls: ALL,
			deepDive: {
				math: [{ tex: '\\mathrm{ReLU}(z) = \\max(0, z)', caption: 'Keep positive values, drop negative ones.' }]
			}
		},
		{
			title: 'Sigmoid squashes to a confidence',
			body: 'Sigmoid maps any sum into the range 0 to 1. Large positive sums give values near 1, large negative sums give values near 0, and a sum of zero gives exactly 0.5.\n\nRead the [[data:output]] as how strongly the neuron fires for these inputs.',
			params: { ...FULL, act: 1 },
			controls: ALL,
			deepDive: {
				math: [{ tex: '\\sigma(z) = \\frac{1}{1 + e^{-z}}', caption: 'The sigmoid function.' }],
				code: { lang: 'ts', source: ACTIVATE_CODE }
			}
		},
		{
			title: 'Make it fire',
			body: 'The inputs are now fixed at 1 and [[neg:−1]]. Find [[param:weights]] and a [[param:bias]] that push the output above 0.9, then settings that push it below 0.1.\n\nThat search for good numbers is what learning means. A network is many of these units wired together, each one feeding the next.',
			params: { ...FULL, act: 1, x1: 1, x2: -1, w1: 0, w2: 0, b: 0 },
			controls: ['w1', 'w2', 'b', 'act']
		}
	],
	questions,
	createScene: createNeuronScene
};

export default lesson;

import type { LessonDef } from '../../types.ts';
import { questions } from './questions.ts';
import { createBackpropScene } from './scene.ts';

const BACKWARD_CODE = `// A 2-2-1 network: tanh hidden layer, sigmoid output, loss L = 0.5 * (yHat - y)^2.
function backward(x: number[], W1: number[][], W2: number[], yHat: number, y: number, a1: number[]) {
  // 1. Blame at the output: dL/dz = (yHat - y) * sigmoid'(z) = (yHat - y) * yHat * (1 - yHat)
  const dOut = (yHat - y) * yHat * (1 - yHat);

  // 2. Blame at each hidden neuron: pass dOut back through its weight, times tanh'(z) = 1 - a^2
  const dHid = a1.map((a, j) => W2[j] * dOut * (1 - a * a));

  // 3. Gradient of every weight = blame at the destination * signal at the source
  const gW2 = a1.map((a) => dOut * a);
  const gW1 = dHid.map((d) => x.map((xi) => d * xi));

  return { gW1, gW2 };
}`;

const lesson: LessonDef = {
	id: 'backpropagation',
	track: 'fundamentals',
	title: 'Backpropagation',
	summary: 'Trace the blame for a wrong answer backward through a network, one weight at a time.',
	defaults: { x1: 0.8, x2: -0.4, target: 1, lr: 0.5, phase: 0 },
	discrete: ['phase'],
	controls: [
		{ kind: 'action', id: 'update', label: 'Update weights', icon: 'step' },
		{ kind: 'action', id: 'run', label: 'Train', icon: 'play' },
		{ kind: 'action', id: 'reset', label: 'Reset', icon: 'reset' },
		{ kind: 'slider', key: 'lr', label: 'Learning rate', min: 0.05, max: 2, step: 0.05, tone: 'grad', digits: 2 },
		{ kind: 'slider', key: 'x1', label: 'Input x₁', min: -1, max: 1, step: 0.05, tone: 'data' },
		{ kind: 'slider', key: 'x2', label: 'Input x₂', min: -1, max: 1, step: 0.05, tone: 'data' },
		{ kind: 'toggle', key: 'target', label: 'Target is 1' }
	],
	readouts: [
		{ key: 'pred', label: 'Prediction ŷ', tone: 'data', digits: 3 },
		{ key: 'target', label: 'Target y', tone: 'ok', digits: 0 },
		{ key: 'loss', label: 'Loss', tone: 'grad', digits: 4 },
		{ key: 'updates', label: 'Updates', digits: 0 }
	],
	legend: [
		{ tone: 'data', label: 'Activation' },
		{ tone: 'param', label: 'Weight' },
		{ tone: 'grad', label: 'Blame (gradient)' },
		{ tone: 'ok', label: 'Target' }
	],
	steps: [
		{
			title: 'First, a guess',
			body: 'The network turns two inputs into one number. Each connection multiplies a [[data:value]] by its [[param:weight]], each neuron adds up what arrives, and the result flows forward to the output.\n\nThat number, ŷ, is the network’s guess. With its current weights it is probably wrong. Drag the inputs and watch the guess change.',
			params: { phase: 0 },
			controls: ['x1', 'x2']
		},
		{
			title: 'Then, a score',
			body: 'Now compare the guess with the [[ok:target]], the answer we wanted. The gap between them is the error, and the [[grad:loss]] turns that gap into one number: zero means perfect, bigger means worse.\n\nTry flipping the target and see the loss jump.',
			params: { phase: 1 },
			controls: ['x1', 'x2', 'target'],
			deepDive: {
				math: [{ tex: 'L = \\tfrac{1}{2}(\\hat{y} - y)^2', caption: 'Squared error: the loss grows quickly as the guess moves away from the target.' }]
			}
		},
		{
			title: 'Blame starts at the output',
			body: 'To improve, we need to know who is responsible. Ask: if the output neuron’s sum nudged up a little, how would the [[grad:loss]] change? That number is [[grad:δ]] (delta), the output’s share of the blame.\n\nIts sign says which way to push, and its size says how much it matters.',
			params: { phase: 2 },
			controls: ['target'],
			deepDive: {
				math: [
					{
						tex: '\\delta^{\\text{out}} = (\\hat{y} - y)\\,\\sigma\'(z^{\\text{out}})',
						caption: 'The error, scaled by how sensitive the output neuron is to its sum z.'
					}
				]
			}
		},
		{
			title: 'Blame flows backward',
			body: 'The hidden neurons never touched the target, yet they share the blame. Each one receives it through the [[param:weight]] that connected it to the output: a strong connection passes on more.\n\nReusing the output’s [[grad:δ]] instead of starting over is the chain rule at work.',
			params: { phase: 3 },
			controls: [],
			deepDive: {
				math: [
					{
						tex: '\\delta_j = w_j^{\\text{out}}\\,\\delta^{\\text{out}}\\,\\tanh\'(z_j)',
						caption: 'Blame arriving through the weight, scaled by the hidden neuron’s own sensitivity.'
					}
				]
			}
		},
		{
			title: 'Every weight gets a gradient',
			body: 'A weight’s [[grad:gradient]] answers: if this weight grew a little, how would the loss change? It equals the blame at the weight’s destination times the [[data:signal]] at its source.\n\nOne backward sweep gives the gradient of every weight at once, which is why backpropagation is fast. Thicker edges mean bigger gradients.',
			params: { phase: 4 },
			controls: ['x1', 'x2'],
			deepDive: {
				math: [{ tex: '\\frac{\\partial L}{\\partial w_{ji}} = \\delta_j\\,a_i', caption: 'Blame at the destination j, times the activation at the source i.' }],
				code: { lang: 'ts', source: BACKWARD_CODE }
			}
		},
		{
			title: 'Nudge every weight downhill',
			body: 'The gradient points uphill on the loss, so we move each [[param:weight]] a small step the other way. The [[grad:learning rate]] η sets how big that step is.\n\nPress **Update weights** and watch the loss drop. The edges change width as the weights change.',
			params: { phase: 5 },
			controls: ['update', 'reset', 'lr'],
			deepDive: {
				math: [{ tex: 'w \\leftarrow w - \\eta\\,\\frac{\\partial L}{\\partial w}', caption: 'Gradient descent, applied to every weight in the network.' }]
			}
		},
		{
			title: 'Repeat, and the network learns',
			body: 'Forward pass, loss, backward pass, update, then again and again: that loop is [[grad:training]]. Each lap makes the [[data:prediction]] a little closer to the [[ok:target]].\n\nPress **Train** and watch the loss fall. Then flip the target and train again: the same network learns the opposite answer.',
			params: { phase: 5 }
		}
	],
	questions,
	createScene: createBackpropScene
};

export default lesson;

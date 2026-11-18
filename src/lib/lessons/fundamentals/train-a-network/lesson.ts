import type { LessonDef } from '../../types.ts';
import { createTrainScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'train-a-network',
	track: 'fundamentals',
	title: 'Train a network',
	summary: 'Watch a small network learn to separate two groups of points, one step at a time.',
	defaults: { dataset: 0, hidden: 4, layers: 1, act: 0, lr: 0.3, noise: 0.05, scenario: 1 },
	discrete: ['scenario'],
	controls: [
		{ kind: 'action', id: 'run', label: 'Train', icon: 'play' },
		{ kind: 'action', id: 'step', label: 'Step', icon: 'step' },
		{ kind: 'action', id: 'reset', label: 'Reset', icon: 'reset' },
		{ kind: 'slider', key: 'lr', label: 'Learning rate', min: 0.01, max: 1.5, step: 0.01, tone: 'grad', digits: 2 },
		{
			kind: 'select',
			key: 'dataset',
			label: 'Data',
			options: [
				{ value: 0, label: 'Circles' },
				{ value: 1, label: 'Moons' },
				{ value: 2, label: 'XOR' },
				{ value: 3, label: 'Spiral' }
			]
		},
		{
			kind: 'select',
			key: 'hidden',
			label: 'Units per layer',
			options: [
				{ value: 2, label: '2' },
				{ value: 4, label: '4' },
				{ value: 8, label: '8' }
			]
		},
		{
			kind: 'select',
			key: 'layers',
			label: 'Hidden layers',
			options: [
				{ value: 1, label: '1' },
				{ value: 2, label: '2' }
			]
		},
		{
			kind: 'select',
			key: 'act',
			label: 'Activation',
			options: [
				{ value: 0, label: 'Tanh' },
				{ value: 1, label: 'ReLU' }
			]
		},
		{ kind: 'slider', key: 'noise', label: 'Noise', min: 0, max: 0.25, step: 0.01, tone: 'data', digits: 2 }
	],
	readouts: [
		{ key: 'loss', label: 'Loss', tone: 'grad', digits: 3 },
		{ key: 'acc', label: 'Accuracy', tone: 'ok' },
		{ key: 'steps', label: 'Steps', digits: 0 }
	],
	legend: [
		{ tone: 'data', label: 'Class A' },
		{ tone: 'neg', label: 'Class B' },
		{ tone: 'param', label: 'Weights' },
		{ tone: 'grad', label: 'Loss' }
	],
	steps: [
		{
			title: 'The task: tell two groups apart',
			body: 'Each dot is one example. Its position gives two inputs, and its color is its class: [[data:class A]] or [[neg:class B]]. The goal is to predict the class from the position alone.\n\nThe background shows what the network predicts at every point. Its [[param:weights]] are random, so the picture means nothing yet.',
			params: { scenario: 1 },
			controls: ['step', 'reset']
		},
		{
			title: 'Training bends the boundary',
			body: 'Each step runs a forward pass on every dot, measures the [[grad:loss]], sends the error backward, and nudges the [[param:weights]] a little to make it smaller.\n\nPress **Train** and watch the white boundary wrap around the inner group while the loss curve falls.',
			params: { scenario: 2 },
			controls: ['run', 'step', 'reset']
		},
		{
			title: 'The learning rate still matters',
			body: 'The [[grad:learning rate]] sets how far each nudge goes. With a tiny rate the loss curve barely falls, and training crawls.\n\nRaise it and learning speeds up. Push it too high and the loss starts to jump around instead of settling.',
			params: { scenario: 3, lr: 0.03 },
			controls: ['run', 'step', 'reset', 'lr']
		},
		{
			title: 'Too few units cannot bend enough',
			body: 'Each hidden unit adds one bend to the boundary. With only 2 units the network cannot close a ring around the inner group, so some dots stay wrong, marked with an [[grad:orange ring]].\n\nSwitch to 4 or 8 units and train again.',
			params: { scenario: 4, hidden: 2 },
			controls: ['run', 'reset', 'lr', 'hidden']
		},
		{
			title: 'Harder patterns need more capacity',
			body: 'A spiral cannot be separated with just a few bends. A second hidden layer lets the network combine bends into curls.\n\nIt also takes many more steps, so this step uses ReLU and a higher [[grad:learning rate]] to get there sooner. Keep training until the arms separate.',
			params: { scenario: 5, dataset: 3, hidden: 8, layers: 2, act: 1, lr: 1 },
			controls: ['run', 'reset', 'lr', 'dataset', 'hidden', 'layers', 'act'],
			deepDive: {
				note: 'This is the same loop as the previous lessons: forward pass, loss, backward pass, update. Only the network and the data are bigger.',
				code: {
					lang: 'ts',
					source:
						'for (let step = 0; step < steps; step++) {\n  const trace = forward(net, x);\n  const grads = backward(net, trace, y, \'bce\');\n  applyGrads(net, grads, lr);\n}'
				}
			}
		},
		{
			title: 'Your turn',
			body: 'Try every dataset. Add some noise and see how the boundary copes. Change the activation and compare how the shape of the boundary changes.\n\nFor each dataset, find the **smallest network** that still solves it.',
			params: { scenario: 6 }
		}
	],
	createScene: createTrainScene
};

export default lesson;

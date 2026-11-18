import type { LessonDef } from '../../types.ts';
import { questions } from './questions.ts';
import { createGradientDescentScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'gradient-descent',
	track: 'fundamentals',
	title: 'Gradient descent',
	summary: 'Roll downhill on the loss, one slope-guided step at a time.',
	defaults: { lr: 0.2, beta: 0.9, map: 0, surface: 0, race: 0, scenario: 1 },
	discrete: ['scenario'],
	controls: [
		{ kind: 'action', id: 'run', label: 'Run', icon: 'play' },
		{ kind: 'action', id: 'step', label: 'Step', icon: 'step' },
		{ kind: 'action', id: 'reset', label: 'Reset', icon: 'reset' },
		{ kind: 'slider', key: 'lr', label: 'Learning rate', min: 0.01, max: 1.1, step: 0.01, tone: 'grad', digits: 2 },
		{ kind: 'slider', key: 'beta', label: 'Momentum β', min: 0, max: 0.98, step: 0.01, tone: 'data', digits: 2 },
		{
			kind: 'select',
			key: 'surface',
			label: 'Surface',
			options: [
				{ value: 0, label: 'Bowl' },
				{ value: 1, label: 'Ravine' },
				{ value: 2, label: 'Two valleys' }
			]
		},
		{ kind: 'toggle', key: 'race', label: 'Compare optimizers' }
	],
	readouts: [
		{ key: 'loss', label: 'Loss', tone: 'grad', digits: 3 },
		{ key: 'steps', label: 'Steps', digits: 0 }
	],
	legend: [
		{ tone: 'grad', label: 'Plain gradient descent' },
		{ tone: 'data', label: 'Momentum' },
		{ tone: 'ok', label: 'Adam' },
		{ tone: 'param', label: 'Loss landscape' }
	],
	steps: [
		{
			title: 'Learning is rolling downhill',
			body: 'This curve shows the [[param:loss]] for every value of one weight. A high point means a bad guess, a low point means a good one.\n\nTraining means finding the lowest point. Press **Step** and watch the [[grad:ball]] move.',
			params: { scenario: 1 },
			controls: ['step', 'reset'],
			legend: ['Plain gradient descent', 'Loss landscape']
		},
		{
			title: 'The slope points the way',
			body: 'The [[grad:gradient]] is the slope of the curve at the current weight. It tells us which way is uphill, so we step the opposite way.\n\nA steep slope gives a big step and a flat one gives a small step, so the [[data:steps]] shrink near the bottom.',
			params: { scenario: 2 },
			controls: ['step', 'reset'],
			legend: ['Plain gradient descent', 'Loss landscape'],
			deepDive: {
				math: [
					{
						tex: 'w \\leftarrow w - \\eta \\, \\frac{dL}{dw}',
						caption: 'The update rule. The Greek letter eta (η) is the learning rate.'
					}
				],
				code: {
					lang: 'ts',
					source:
						'for (let i = 0; i < steps; i++) {\n  w = w - lr * grad(w);\n}'
				}
			}
		},
		{
			title: 'The learning rate sets the stride',
			body: 'The [[grad:learning rate]] scales every step. With a small one the ball crawls and needs many steps to arrive.\n\nPress **Run**, then try raising the learning rate to see the ball move faster.',
			params: { scenario: 3, lr: 0.08 },
			controls: ['run', 'step', 'reset', 'lr'],
			legend: ['Plain gradient descent', 'Loss landscape']
		},
		{
			title: 'Too big a stride overshoots',
			body: 'With a large [[grad:learning rate]] the ball jumps across the valley and lands on the other side, bouncing back and forth.\n\nAbove 1.0 on this curve each jump lands higher than the last, and training **diverges**. Try it.',
			params: { scenario: 4, lr: 0.93 },
			controls: ['run', 'step', 'reset', 'lr'],
			legend: ['Plain gradient descent', 'Loss landscape'],
			deepDive: {
				note: 'For L = w² the gradient is 2w, so every update multiplies the weight by a constant. That only shrinks the weight when the constant is between −1 and 1.',
				math: [
					{
						tex: 'w \\leftarrow w - \\eta \\cdot 2w = (1 - 2\\eta)\\,w',
						caption: 'It converges only when |1 − 2η| < 1, which means 0 < η < 1.'
					}
				]
			}
		},
		{
			title: 'Real models have many weights',
			body: 'Now there are two weights. The map shows the [[param:loss as contour lines]], like a topographic map: the closer the lines, the steeper the ground.\n\nClick anywhere on the map to choose a starting point, then run.',
			params: { scenario: 5, map: 1, surface: 0, lr: 0.1 },
			controls: ['run', 'step', 'reset', 'lr'],
			legend: ['Plain gradient descent', 'Loss landscape']
		},
		{
			title: 'Ravines cause zigzags',
			body: 'In a narrow valley the [[grad:gradient]] points mostly across it, toward the nearest wall. Plain descent bounces between the walls and crawls along the floor.\n\nA smaller learning rate stops the bouncing but makes the crawl even slower.',
			params: { scenario: 6, map: 1, surface: 1, lr: 0.42 },
			controls: ['run', 'step', 'reset', 'lr', 'surface'],
			legend: ['Plain gradient descent', 'Loss landscape']
		},
		{
			title: 'Momentum and Adam',
			body: '[[data:Momentum]] keeps a running velocity, so zigzags cancel out while progress along the valley builds. [[ok:Adam]] also scales each weight\'s step by how large its recent gradients were.\n\nTry **Two valleys** and different starting points: optimizers can settle in different minima.',
			params: { scenario: 7, map: 1, surface: 1, race: 1, lr: 0.1 },
			deepDive: {
				math: [
					{
						tex: 'v \\leftarrow \\beta v + g,\\quad w \\leftarrow w - \\eta v',
						caption: 'Momentum: v is a running velocity and β controls how much of it is kept.'
					},
					{
						tex: 'w \\leftarrow w - \\eta \\, \\frac{\\hat m}{\\sqrt{\\hat v} + \\epsilon}',
						caption: 'Adam: m̂ is a smoothed gradient and v̂ a smoothed squared gradient.'
					}
				],
				code: {
					lang: 'ts',
					source:
						'v = beta * v + g;\nw = w - lr * v;'
				}
			}
		}
	],
	questions,
	createScene: createGradientDescentScene
};

export default lesson;

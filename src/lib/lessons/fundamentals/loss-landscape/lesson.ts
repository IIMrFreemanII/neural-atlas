import type { LessonDef } from '../../types.ts';
import { bestLine } from './model.ts';
import { questions } from './questions.ts';
import { createLossLandscapeScene } from './scene.ts';

const snap = (v: number) => Number((Math.round(v / 0.02) * 0.02).toFixed(2));
const BEST = bestLine();
const BEST_W = snap(BEST.w);
const BEST_B = snap(BEST.b);

const lesson: LessonDef = {
	id: 'loss-landscape',
	track: 'fundamentals',
	title: 'Loss landscape',
	summary: 'Turn every possible setting of a model into a height and see the terrain training has to cross.',
	defaults: { w: 0.2, b: 0.6, model: 0, resid: 0, surface: 0, pose: 0 },
	discrete: ['pose'],
	controls: [
		{ kind: 'slider', key: 'w', label: 'Weight w', min: -2, max: 2, step: 0.02, tone: 'param', digits: 2 },
		{ kind: 'slider', key: 'b', label: 'Bias b', min: -2, max: 2, step: 0.02, tone: 'param', digits: 2 },
		{
			kind: 'select',
			key: 'model',
			label: 'Model',
			options: [
				{ value: 0, label: 'Line' },
				{ value: 1, label: 'Tanh neuron' }
			]
		}
	],
	readouts: [{ key: 'loss', label: 'Loss', tone: 'grad', digits: 3 }],
	legend: [
		{ tone: 'data', label: 'Data points' },
		{ tone: 'param', label: 'Model and its settings' },
		{ tone: 'grad', label: 'Error and loss' },
		{ tone: 'ok', label: 'Best fit' }
	],
	steps: [
		{
			title: 'A model makes guesses',
			body: 'The [[data:dots]] are our data: examples where we know the right answer. The [[param:line]] is the model. Its **weight** sets the tilt and its **bias** sets the height.\n\nDrag the sliders and try to make the line pass through the dots.',
			params: {},
			grid: false,
			controls: ['w', 'b'],
			legend: ['Data points', 'Model and its settings']
		},
		{
			title: 'Error is the gap for each point',
			body: 'For every dot, the model makes a guess. The amber [[grad:segment]] is the gap between that guess and the true value. This gap is the **error** for one point.\n\nA good setting leaves every segment short.',
			params: { resid: 1 },
			grid: false,
			controls: ['w', 'b'],
			legend: ['Data points', 'Model and its settings', 'Error and loss']
		},
		{
			title: 'Loss squeezes all errors into one number',
			body: 'To judge a setting with one number, square each gap and take the average. This is the mean squared error, called the [[grad:loss]]. Lower is better.\n\nWatch the **Loss** readout while you move the sliders. Squaring makes big misses count much more than small ones.',
			params: { resid: 1 },
			grid: false,
			controls: ['w', 'b'],
			legend: ['Data points', 'Model and its settings', 'Error and loss'],
			deepDive: {
				math: [
					{
						tex: 'L(w,b) = \\frac{1}{n}\\sum_{i=1}^{n}\\left(w x_i + b - y_i\\right)^2',
						caption: 'Mean squared error of the line y = wx + b over n data points.'
					}
				],
				code: {
					lang: 'ts',
					source:
						'function loss(w: number, b: number, points: { x: number; y: number }[]) {\n  let sum = 0;\n  for (const { x, y } of points) {\n    const gap = w * x + b - y;\n    sum += gap * gap;\n  }\n  return sum / points.length;\n}'
				}
			}
		},
		{
			title: 'Every setting has a loss',
			body: 'Try every pair of [[param:weight and bias]] and record the [[grad:loss]] of each. The result is a surface: each point on the floor is one setting, and its height is how bad that setting is.\n\nThe glowing [[grad:ball]] is the current setting. Drag the surface to rotate it.',
			params: { resid: 1, surface: 1, pose: 1 },
			grid: false,
			controls: ['w', 'b'],
			deepDive: {
				note: 'The height shows the square root of the loss, so the valley floor stays readable instead of being squashed flat next to the steep walls. The Loss readout always shows the true value.'
			}
		},
		{
			title: 'The lowest point is the best model',
			body: 'At the bottom of the [[ok:bowl]] the line fits the dots best. Move the sliders away from it and the ball climbs the walls.\n\nFor a line with squared error the surface is always a single smooth bowl, so there is exactly one best setting.',
			params: { resid: 1, surface: 1, pose: 2, w: BEST_W, b: BEST_B },
			grid: false,
			controls: ['w', 'b']
		},
		{
			title: 'Real landscapes are not bowls',
			body: 'Now the model is one tanh neuron, which bends the line into an S shape. Its surface is no longer a bowl: there are wide [[param:flat plateaus]] where moving barely changes the [[grad:loss]], and a curved valley.\n\nNetworks with millions of weights have landscapes with far more structure.',
			params: { resid: 1, surface: 1, pose: 3, model: 1, w: -1.5, b: 1.2 },
			grid: false
		},
		{
			title: 'Training is finding the way down',
			body: 'Nobody can try every setting when a network has millions of weights. The goal is still the same: reach a low point of the [[grad:loss]].\n\nPlay with this landscape and notice how the [[param:slope]] under the ball changes. The next lesson shows how to follow that slope downhill.',
			params: { resid: 1, surface: 1, pose: 4, model: 1, w: 0.4, b: 0.3 },
			grid: false
		}
	],
	questions,
	createScene: createLossLandscapeScene
};

export default lesson;

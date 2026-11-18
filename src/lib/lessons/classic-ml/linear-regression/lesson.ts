import type { LessonDef } from '../../types.ts';
import { questions } from './questions.ts';
import { createLinearRegressionScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'linear-regression',
	track: 'classic-ml',
	title: 'Linear regression',
	summary: 'Fit a line to noisy points by minimizing squared error.',
	defaults: { w: 0.2, b: -0.3, show: 3, best: 0, noise: 0.18, scenario: 1 },
	discrete: ['show', 'scenario'],
	controls: [
		{ kind: 'action', id: 'fit', label: 'Fit', icon: 'play' },
		{ kind: 'action', id: 'shuffle', label: 'New data', icon: 'shuffle' },
		{ kind: 'slider', key: 'w', label: 'Slope', min: -2, max: 2, step: 0.01, tone: 'param', digits: 2 },
		{ kind: 'slider', key: 'b', label: 'Intercept', min: -1, max: 1, step: 0.01, tone: 'param', digits: 2 },
		{ kind: 'toggle', key: 'best', label: 'Show best fit' },
		{ kind: 'slider', key: 'noise', label: 'Noise', min: 0, max: 0.5, step: 0.01, tone: 'data', digits: 2 }
	],
	readouts: [
		{ key: 'mse', label: 'Squared error', tone: 'grad', digits: 3 },
		{ key: 'best', label: 'Best possible', tone: 'ok', digits: 3 }
	],
	legend: [
		{ tone: 'data', label: 'Data points' },
		{ tone: 'param', label: 'Model line' },
		{ tone: 'grad', label: 'Error' },
		{ tone: 'ok', label: 'Best fit' }
	],
	steps: [
		{
			title: 'A trend hidden in noisy points',
			body: 'Each [[data:dot]] is one measurement. The horizontal position is what you know, and the vertical position is what you want to predict.\n\nThey rise together, but not exactly. Press **New data** to see another batch with the same hidden trend.',
			params: { show: 0 },
			controls: ['shuffle']
		},
		{
			title: 'A line is the model',
			body: 'A straight [[param:line]] is the simplest model. It has two parameters: the **slope** tilts it and the **intercept** lifts it.\n\nMove both sliders and try to run the line through the middle of the cloud.',
			params: { show: 1 },
			controls: ['w', 'b']
		},
		{
			title: 'Measure the misses',
			body: 'Each [[grad:segment]] is how far the [[param:line]] is from one point. Some are above and some are below, so simply adding them up would let them cancel out.\n\nAdjust the sliders and watch the segments shrink and grow.',
			params: { show: 2 },
			controls: ['w', 'b']
		},
		{
			title: 'Square the misses',
			body: 'Each miss becomes a [[grad:square]]. A square is never negative, and a miss twice as long costs **four times** as much.\n\nThe average area is the mean squared error, shown by the bar on the right. Move the line to make the squares smaller.',
			params: { show: 3 },
			controls: ['w', 'b'],
			deepDive: {
				note: 'Mean squared error averages the squared miss over all n points. Fitting the line means finding the slope and intercept that make it smallest.',
				math: [
					{
						tex: '\\mathrm{MSE}(w, b) = \\frac{1}{n} \\sum_{i=1}^{n} \\left( w x_i + b - y_i \\right)^2',
						caption: 'Each term is the area of one square.'
					}
				]
			}
		},
		{
			title: 'Slide to the best line',
			body: 'Press **Fit**. Each step nudges the slope and the intercept in the direction that shrinks the squares, and the sliders follow along.\n\nThe bar sinks until it reaches the [[ok:green tick]], the lowest error any line can reach. The [[ok:green line]] is that best fit.',
			params: { show: 3, best: 1 },
			controls: ['fit', 'w', 'b', 'shuffle'],
			deepDive: {
				note: 'Fit runs gradient descent. The gradient says how the error changes when slope or intercept grows, and each step moves a little the other way. For a line there is also an exact answer, which is the green tick.',
				math: [
					{
						tex: '\\frac{\\partial\\,\\mathrm{MSE}}{\\partial w} = \\frac{2}{n} \\sum_i (\\hat y_i - y_i)\\, x_i \\qquad \\frac{\\partial\\,\\mathrm{MSE}}{\\partial b} = \\frac{2}{n} \\sum_i (\\hat y_i - y_i)',
						caption: 'The two gradient components.'
					},
					{
						tex: 'w^{*} = \\frac{\\sum_i (x_i - \\bar x)(y_i - \\bar y)}{\\sum_i (x_i - \\bar x)^2} \\qquad b^{*} = \\bar y - w^{*} \\bar x',
						caption: 'The closed-form least squares solution.'
					}
				],
				code: {
					lang: 'ts',
					source:
						'for (let step = 0; step < 100; step++) {\n  const { dw, db } = gradient(x, y, w, b);\n  w -= 0.3 * dw;\n  b -= 0.3 * db;\n}'
				}
			}
		},
		{
			title: 'One outlier pulls hard',
			body: 'One [[data:point]] sits far below the rest. Because misses are squared, its square is huge, so the [[ok:best line]] tilts toward it.\n\nDrag that point back up to the others and watch the green line recover. Then press **Fit**.',
			params: { show: 3, best: 1, scenario: 2, w: 0.8, b: 0.1 },
			controls: ['fit', 'best']
		},
		{
			title: 'Your turn',
			body: 'Drag any [[data:point]] and see how far the [[ok:best line]] moves. Raise the noise and the squares grow, even for the best line.\n\nPress **Fit** to slide the line into place again.',
			params: { show: 3, best: 1, scenario: 3 }
		}
	],
	questions,
	createScene: createLinearRegressionScene
};

export default lesson;

import type { LessonDef } from '../../types.ts';
import { createOverfittingScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'overfitting',
	track: 'classic-ml',
	title: 'Overfitting and underfitting',
	summary: 'Too simple a model misses the pattern, too flexible a model memorizes noise.',
	defaults: { degree: 4, n: 14, noise: 0.15, reg: 0, fit: 1, truthLine: 0, test: 0, chart: 0, ghosts: 0, scenario: 1 },
	discrete: ['fit', 'truthLine', 'test', 'chart', 'ghosts', 'scenario'],
	controls: [
		{ kind: 'action', id: 'resample', label: 'New sample', icon: 'shuffle' },
		{ kind: 'slider', key: 'degree', label: 'Degree', min: 1, max: 12, step: 1, tone: 'param', digits: 0 },
		{ kind: 'slider', key: 'n', label: 'Training points', min: 8, max: 60, step: 1, tone: 'data', digits: 0 },
		{ kind: 'slider', key: 'noise', label: 'Noise', min: 0, max: 0.4, step: 0.01, tone: 'data', digits: 2 },
		{ kind: 'slider', key: 'reg', label: 'Penalty', min: 0, max: 1, step: 0.01, tone: 'param', digits: 2 }
	],
	readouts: [
		{ key: 'train', label: 'Training error', tone: 'data', digits: 3 },
		{ key: 'test', label: 'Test error', tone: 'ok', digits: 3 }
	],
	legend: [
		{ tone: 'data', label: 'Training points' },
		{ tone: 'ok', label: 'Test points' },
		{ tone: 'param', label: 'Model' },
		{ tone: 'muted', label: 'True pattern' }
	],
	steps: [
		{
			title: 'A pattern hidden in noise',
			body: 'The faint [[muted:curve]] is the real pattern. Each [[data:dot]] is a noisy measurement of it, a little above or below the truth.\n\nA model never sees the curve. It only sees the dots, and has to guess the pattern from them.',
			params: { fit: 0, truthLine: 1 },
			controls: []
		},
		{
			title: 'Too simple: underfitting',
			body: 'Here the [[param:model]] is a straight line. A line cannot bend, so it misses the shape of the [[data:data]] however you place it.\n\nThis is **underfitting**: the model is too simple for the pattern, so even its error on the dots stays high.',
			params: { degree: 1 },
			controls: []
		},
		{
			title: 'Flexible enough',
			body: 'Now the [[param:curve]] has a few bends. It follows the overall shape of the [[data:dots]] without chasing any single one.\n\nTry the **Degree** slider a step or two either way and see how the curve changes.',
			params: { degree: 4 },
			controls: ['degree']
		},
		{
			title: 'Too flexible: overfitting',
			body: 'Push the degree up to 10. The [[param:curve]] now bends to chase individual [[data:dots]], and the [[data:training error]] keeps shrinking.\n\nBut between the dots it swings, and sometimes leaves the plot. It is memorizing the noise instead of learning the pattern. That is **overfitting**.',
			params: { degree: 10 },
			controls: ['degree'],
			deepDive: {
				note: 'The degree is the highest power of x the model may use. A higher degree allows more bends, so more freedom to chase the data.',
				math: [
					{ tex: '\\hat{y}(x) = c_0 + c_1 x + c_2 x^2 + \\dots + c_d x^d', caption: 'a polynomial of degree d' },
					{ tex: '\\hat{y}(x) = \\sum_{j=0}^{d} c_j\\,T_j(x)', caption: 'the same family, written with Chebyshev polynomials T_j, which the code uses for numerical stability' }
				]
			}
		},
		{
			title: 'Test it on points it has never seen',
			body: 'The green [[ok:rings]] are fresh measurements that were not used for fitting. The wiggly [[param:curve]] misses many of them, because its wiggles were shaped by the old dots.\n\nLower the **Degree** and watch the test error in the top corner fall.',
			params: { degree: 10, test: 1, truthLine: 1 },
			controls: ['degree']
		},
		{
			title: 'The sweet spot',
			body: 'The chart shows both errors at every degree. The [[data:training error]] only ever falls. The [[ok:test error]] falls, then climbs again.\n\nThe best model sits at the bottom of the test curve. Slide the **Degree** and watch the [[param:marker]] move along it, then push it to 12 and see the test error shoot up.',
			params: { degree: 10, test: 1, chart: 1 },
			controls: ['degree'],
			deepDive: {
				note: 'Both are mean squared errors. The only difference is which points they average over.',
				math: [
					{ tex: 'E_{\\text{train}} = \\frac{1}{n}\\sum_{i=1}^{n}\\bigl(y_i - \\hat{y}(x_i)\\bigr)^2', caption: 'over the n training points' },
					{ tex: 'E_{\\text{test}} = \\frac{1}{m}\\sum_{k=1}^{m}\\bigl(y_k - \\hat{y}(x_k)\\bigr)^2', caption: 'over m held-out points the fit never saw' }
				]
			}
		},
		{
			title: 'A new sample, a new curve',
			body: 'The faint [[param:curves]] are fits to 20 other samples of the same pattern. At high degree they disagree wildly: a different sample gives a very different curve. That is high **variance**.\n\nSet the degree to 1 and they agree, but all miss the [[muted:pattern]]. That is high **bias**. Press **New sample** to draw fresh dots.',
			params: { degree: 10, ghosts: 1, truthLine: 1 },
			controls: ['degree', 'resample'],
			deepDive: {
				note: 'Bias is the error from a model too rigid to match the pattern, even on average. Variance is the error from a model so flexible that its answer depends on which sample it saw. Raising the degree trades bias for variance.'
			}
		},
		{
			title: 'Two cures: more data or a penalty',
			body: 'Raise **Training points** and the wiggles calm down: more dots leave less room to memorize noise. Or keep the data and raise **Penalty**, which charges the model for large coefficients and so keeps the [[param:curve]] smooth.\n\nWatch the [[ok:test error]] as you try each.',
			params: { degree: 10, test: 1, chart: 1, ghosts: 1 },
			deepDive: {
				note: 'The penalty is called ridge regression. The coefficient c_0 is left out so the curve can still sit at any height.',
				math: [{ tex: '\\min_{c}\; \\sum_{i=1}^{n}\\bigl(y_i - \\hat{y}(x_i)\\bigr)^2 + \\lambda \\sum_{j\\ge 1} c_j^2', caption: 'squared error plus a charge for big coefficients' }],
				code: {
					lang: 'ts',
					source:
						"import { fitPoly, evalPoly } from '../ml/polyfit.ts';\n\nconst model = fitPoly(x, y, 10, 0.01); // degree 10, lambda 0.01\nconst yHat = evalPoly(model, 0.5);"
				}
			}
		}
	],
	createScene: createOverfittingScene
};

export default lesson;

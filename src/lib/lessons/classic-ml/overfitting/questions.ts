import { fitPoly, polyMse } from '../../../ml/polyfit.ts';
import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';
import { makeSample } from './data.ts';

/**
 * Test error as the scene computes it for a question's starting sample. Every question below uses its own
 * `scenario`, which makes the scene start from sample seed 1; the test sample is always seed 9001.
 */
function testError(degree: number, n: number, noise: number, reg: number): number {
	const lambda = reg <= 0 ? 0 : Math.pow(10, -5 + 5 * reg);
	const train = makeSample(n, noise, 1);
	const test = makeSample(60, noise, 9001);
	return polyMse(fitPoly(train.x, train.y, degree, lambda), test.x, test.y);
}

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'The curve now has degree 4, fitted to 14 training points. You raise the **Degree** to 10. What happens to the [[data:training error]] and to the [[ok:test error]]?\n\nDecide first, then check with the slider.',
		params: { degree: 4, test: 1, scenario: 21 },
		controls: ['degree'],
		options: [
			{
				text: 'Both fall, because a more flexible curve fits better.',
				why: 'A flexible curve fits the training dots better, so that error falls. The test dots were not used for fitting, and the extra bends are shaped by the old dots, so they do not help there.'
			},
			{
				text: 'The training error falls and the test error rises.',
				correct: true,
				why: 'The extra bends chase the noise in the 14 training dots: the training error falls from 0.012 to 0.001. The same bends miss the new dots, and the test error rises from 0.019 to 0.517.'
			},
			{
				text: 'Both rise, because a higher degree makes the curve less steady.',
				why: 'A higher degree can only fit the training dots as well or better, because it can still draw every curve of the lower degree. The training error cannot rise, so only the test error does.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'The curve has degree 10 and only a few training points, so it overfits. The degree stays at 10. Bring the [[ok:test error]] below 0.04 with the other controls.',
		params: { degree: 10, test: 1, scenario: 22 },
		controls: ['n', 'reg'],
		goal: ({ params: p }) => shown(testError(p.degree, Math.round(p.n), p.noise, p.reg), 3) < 0.04,
		hint: 'The curve overfits because it has too much freedom for so few dots. Take away some of that freedom, or give it more dots to be held in check by.',
		solved:
			'Both controls work against the memorizing. More dots leave less room to chase the noise of any single one, and the penalty charges the curve for the large coefficients that wild bends need.',
		example: { params: { reg: 0.7 } }
	},
	{
		kind: 'choice',
		prompt:
			'The [[param:model]] is a straight line fitted to 14 training points, and its [[ok:test error]] is about 0.05. You raise **Training points** to 60. What happens to the test error?\n\nDecide first, then check with the slider.',
		params: { degree: 1, test: 1, scenario: 23 },
		controls: ['n'],
		options: [
			{
				text: 'It stays about the same, because a line cannot bend to fit the pattern.',
				correct: true,
				why: 'More dots only pin down the same straight line more firmly. The pattern is curved, so the line keeps missing it: the test error is 0.049 with 14 points and 0.047 with 60. More data cures overfitting and not underfitting.'
			},
			{
				text: 'It falls a lot, because more dots always teach the model more.',
				why: 'More dots help only if the model can use them. A line has two numbers to set and no bends, so extra dots cannot teach it a curve. A higher degree would have to do that.'
			},
			{
				text: 'It rises, because more dots bring more noise to fit.',
				why: 'A line cannot chase noise, so more noisy dots do not hurt it. With more dots the line is fixed more firmly, and its test error stays about where it was.'
			}
		]
	}
];

import { shown } from '../../questions.ts';
import { curve } from '../../../ml/optimizers.ts';
import type { Question } from '../../types.ts';

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'The learning rate is 0.2. The first step moves the ball from w = 1.8 to w = 1.08. Compare the second step with the first one. Is it longer, shorter or the same?\n\nDecide first, then check with **Step**.',
		params: { scenario: 2 },
		controls: ['step', 'reset'],
		legend: ['Plain gradient descent', 'Loss landscape'],
		options: [
			{
				text: 'The same length, because the learning rate has not changed.',
				why: 'The learning rate is only one factor. A step is the learning rate times the [[grad:gradient]], and the slope is flatter at the new spot.'
			},
			{
				text: 'Longer, because the ball picks up speed on the way down.',
				why: 'Plain gradient descent has no speed. It reads the slope where the ball is and nothing else. A rule that keeps a speed is momentum, and it is a separate rule.'
			},
			{
				text: 'Shorter, because the slope is flatter where the ball is now.',
				correct: true,
				why: 'A step is the learning rate times the slope. The slope is 3.60 at w = 1.8 and 2.16 at w = 1.08, so the second step is 0.2 × 2.16 = 0.43 long, against 0.72 for the first.'
			},
			{
				text: 'Shorter, because the learning rate shrinks after every step.',
				why: 'The learning rate is a setting and stays at 0.2 until you change it. The steps shrink because the slope shrinks as the ball nears the bottom.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'The ball starts at w = 1.8. Set the [[grad:learning rate]] so that the first step carries it across to the other side of the valley, and to a spot lower on the curve than where it started.',
		params: { scenario: 4, lr: 0.2 },
		controls: ['lr', 'step', 'reset'],
		legend: ['Plain gradient descent', 'Loss landscape'],
		goal: ({ params: p }) => {
			const w = curve.start;
			const next = w - p.lr * curve.grad(w);
			return shown(next) <= -0.05 && shown(curve.f(next)) <= shown(curve.f(w)) - 0.05;
		},
		hint: 'Compare two distances: how far the ball is from the bottom, and how far one step takes it. The arrow under the curve shows where the next step lands.',
		solved:
			'At w = 1.8 the slope is 3.6, so the first step is 3.6 times the learning rate. A rate above 0.5 makes the step longer than the 1.8 to the bottom, so the ball crosses over. A rate below 1.0 still lands it lower than it started. At 1.0 it lands at the same height on the other side, and above 1.0 it lands higher.',
		example: { params: { lr: 0.7 } }
	},
	{
		kind: 'choice',
		prompt:
			'On the ravine, plain descent bounces between the walls and crawls along the floor, even though the minimum is straight ahead along the floor. Why does it bounce?\n\nDecide first, then check with **Run** and the slider.',
		params: { scenario: 6, map: 1, surface: 1, lr: 0.42 },
		controls: ['run', 'step', 'reset', 'lr'],
		legend: ['Plain gradient descent', 'Loss landscape'],
		options: [
			{
				text: 'Plain descent adds a little random noise to every step.',
				why: 'Every step is fixed by the slope at the current point. From the same start and the same learning rate, the path is the same every time.'
			},
			{
				text: 'The gradient points mostly across the valley, because the ground is much steeper across than along.',
				correct: true,
				why: 'From the start, the gradient is −0.44 along the floor and 4.00 across it. With a rate of 0.42 the first step moves 1.68 across the valley but only 0.18 along it, so the ball overshoots the floor and keeps crossing it.'
			},
			{
				text: 'The learning rate is too big, and lowering it fixes the problem.',
				why: 'A smaller rate stops the bounce, but the crawl along the floor gets even slower. The cause is the direction of the gradient, which mostly points across the valley.'
			},
			{
				text: 'The gradient points away from the minimum, so each step goes uphill.',
				why: 'The gradient points uphill, and a step goes the opposite way, so every step does go downhill. It goes mostly across the valley instead of along it.'
			}
		]
	}
];

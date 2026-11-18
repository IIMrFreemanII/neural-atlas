import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'Two lines are tested on the same two points. Line A misses both points by 0.2. Line B hits one point exactly and misses the other by 0.4. In both cases the misses add up to a total miss of 0.4. Which line has the larger [[grad:squared error]]?',
		params: { show: 3 },
		controls: [],
		options: [
			{
				text: 'Neither. Both miss by 0.4 in total, so the errors are equal.',
				why: 'The total is the same, but the error adds up squares, not misses. Line A gives 0.2 × 0.2 + 0.2 × 0.2 = 0.08 and line B gives 0.4 × 0.4 = 0.16.'
			},
			{
				text: 'Line A, because it misses two points instead of one.',
				why: 'A miss of 0.2 makes a small square of 0.04. Two of them add up to 0.08, which is still less than the 0.16 of one miss of 0.4.'
			},
			{
				text: 'Line B, because one big miss costs more than two small ones.',
				correct: true,
				why: 'A square grows faster than the miss: line A gives 0.04 + 0.04 = 0.08 and line B gives 0.16. A miss twice as long costs four times as much, so one large miss weighs more than two small ones.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'Use the [[param:slope]] and the [[param:intercept]] to bring the [[grad:squared error]] to within 0.01 of the [[ok:best possible]] error.',
		params: { show: 3, best: 0 },
		controls: ['w', 'b'],
		goal: ({ readouts }) => shown(readouts.mse, 3) <= shown(readouts.best, 3) + 0.01,
		hint: 'Look at the squares. Tilt the line until it follows the rise of the cloud, then lift it until the misses are about as often above as below.',
		solved:
			'The best line is the one with the smallest squares, and you reached almost the same error by eye. No line can go below the green tick, because the points do not lie on any line.'
	},
	{
		kind: 'choice',
		prompt:
			'The [[ok:green line]] is the best fit for all the points, including the one far below the rest. How does that single point change the green line?',
		params: { show: 3, best: 1, scenario: 2, w: 0.8, b: 0.1 },
		controls: ['best', 'fit'],
		options: [
			{
				text: 'It tilts the line toward that point, because its miss gets squared and becomes huge.',
				correct: true,
				why: 'The line is chosen to make the sum of the squares small. A far miss turns into a very large square, so the best line gives up some fit on the other points to get closer to this one.'
			},
			{
				text: 'It barely changes the line, because it is only one point of many.',
				why: 'Each point has one square, but the squares are not equal. A point far from the line has a square much larger than the others, so it pulls hard. Drag the point up to the others and the green line moves.'
			},
			{
				text: 'It makes the line pass exactly through that point.',
				why: 'The line has to suit all the points at once, and a line through the low point alone would miss the others by a lot. The best line only leans toward it.'
			}
		]
	}
];

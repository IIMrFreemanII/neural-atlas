import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';

// Every question has its own `scenario` number (steps use 1 to 8), so the scene starts it afresh.
export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'Every point is colored by its nearest [[text:center]]. You press **Move centers** and then **Assign**. What happens to the [[grad:spread]] shown?\n\nDecide first, then check with the two buttons.',
		params: { scenario: 9, prep: 2, spokes: 1 },
		controls: ['move', 'assign'],
		options: [
			{
				text: 'It stays the same, because moving a center does not move any point.',
				why: 'The spread is the total squared distance from every point to its center, so moving a center changes it. Moving a center to the average of its points makes those distances smaller.'
			},
			{
				text: 'It goes down, or stays the same once nothing changes any more.',
				correct: true,
				why: 'The average is the spot with the smallest total squared distance to a group, so a move cannot raise the spread. Assigning each point to its nearest center cannot raise it either. This is why the run always ends.'
			},
			{
				text: 'It can go up, because some points switch to a different center.',
				why: 'A point only switches to a center that is closer to it than its old one, so its distance can only get shorter. The switches lower the spread further.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'Five blobs, five [[text:centers]]. Get the [[grad:spread]] below 900.',
		params: { scenario: 10, dataset: 1, k: 5, init: 0, seed: 1 },
		controls: ['run', 'reseed', 'init'],
		goal: ({ readouts }) => shown(readouts.inertia, 1) < 900,
		hint: 'Every round lowers the spread, but a run only goes downhill from where it started. If it stops high, more rounds will not help. Think about where the centers begin.',
		solved:
			'With a good start every blob gets its own center, and the spread settles at about 877. A start that puts two centers in one blob leaves another blob without one, and the rounds cannot move a center across that gap.'
	},
	{
		kind: 'choice',
		prompt:
			'K-means with k = 2 runs on the moons, two curved bands that wind around each other. It cuts across both bands instead of giving each band one color. Why?',
		params: { scenario: 11, dataset: 2, k: 2, init: 1, live: 1, autorun: 1 },
		controls: ['run', 'k', 'reseed', 'init'],
		options: [
			{
				text: 'The border between two centers is a straight line, and no straight line splits the two bands.',
				correct: true,
				why: 'A point goes to its nearest center, so the border between two centers is a straight line halfway between them. The bands curve around each other, so any such line cuts through both.'
			},
			{
				text: 'Two centers are too few. With more centers each band would get its own color.',
				why: 'More centers give more pieces, but every border is still a straight cut halfway between two centers. The bands are cut into pieces and not followed.'
			},
			{
				text: 'The start was unlucky. A New start would find the two bands.',
				why: 'A new start moves where the line falls, but with two centers the border is always straight, so it cannot follow a curved band. Try New start and see.'
			}
		]
	}
];

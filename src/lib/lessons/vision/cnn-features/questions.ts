import type { Question } from '../../types.ts';

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'Some cells of a [[data:feature map]] are 0 after ReLU. What does a 0 tell you about the spot of the drawing that the cell looks at?\n\nHover a few cells to read their values.',
		params: { show: 1, scenario: 2, sample: 3 },
		controls: ['sample'],
		legend: ['Input', 'Maps', 'Kernel'],
		readouts: [],
		options: [
			{
				text: 'The drawing is empty at that spot.',
				why: 'A spot on a stroke can give 0 as well. If the stroke is not what this map\'s kernel looks for, the answer there is negative and ReLU sets it to 0. Another kernel can still respond to the same stroke.'
			},
			{
				text: 'The map\'s [[param:kernel]] has weight 0 at that spot.',
				why: 'A kernel does not change as it slides. The same nine weights are used at every position, and only the patch under them differs.'
			},
			{
				text: 'The map\'s pattern is not present at that spot.',
				correct: true,
				why: 'A feature map shows where its kernel\'s pattern appears. A patch that does not match the pattern gives an answer of zero or less, and ReLU turns every negative answer into 0.'
			},
			{
				text: 'The network has decided that this spot does not matter.',
				why: 'The maps are computed before any decision. Only the later layers weigh the maps, and a 0 here only says that this one pattern is absent.'
			}
		]
	},
	{
		kind: 'choice',
		prompt:
			'You press **Clear** and draw a squiggle that looks like none of the four shapes. What does **It sees** show?\n\nDecide first, then check by drawing.',
		params: { show: 4, scenario: 5, sample: 1 },
		controls: ['sample', 'clear'],
		options: [
			{
				text: 'That it does not know, because it never saw a squiggle.',
				why: 'The network has no output for "unknown". It has four scores, one per shape, and they are turned into four probabilities that add up to 100%. One of them is always the largest.'
			},
			{
				text: 'A dash until the drawing looks like one of the shapes.',
				why: 'The dash only appears while the drawing is empty or nearly empty. As soon as there are strokes, the network answers.'
			},
			{
				text: 'The name of one of the four shapes, with a percentage.',
				correct: true,
				why: 'The network can only choose among its four shapes, so it names the one with the highest probability, even for a scribble. The percentage compares the four shapes with each other, not how much the drawing looks like a real shape.'
			},
			{
				text: 'A shape name, but always with a low percentage.',
				why: 'The percentage only says how the four scores compare, not whether the drawing is a real shape. Nothing in the network forces it to be low for a squiggle.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'The drawing is a square now. Press **Clear** and draw a triangle with the pointer, until the network answers **triangle**.',
		params: { show: 4, scenario: 6, sample: 2 },
		controls: ['clear'],
		goal: ({ readouts }) => typeof readouts.guess === 'string' && readouts.guess.startsWith('triangle'),
		hint: 'The maps respond to strokes, and the answer is built from the maps. A triangle has a point on top, two slanted sides and a flat base.',
		solved:
			'Your strokes made the first-layer maps respond, the second layer combined them, and the answer bars followed. The network was trained on drawings of four shapes, so it reads yours by the same strokes.'
	}
];

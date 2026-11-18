import type { Question } from '../../types.ts';

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'The [[data:rebuilt]] drawing is smoother than the original: the wobbles are gone. Why?\n\nSwitch between the shapes and compare the two drawings.',
		params: { show: 1, sample: 3, scenario: 8 },
		controls: ['sample'],
		options: [
			{
				text: 'The decoder blurs every drawing on purpose, to hide the noise.',
				why: 'Nothing in the decoder is a blur filter. It never sees the original at all: it draws a whole new picture from the two numbers it gets.'
			},
			{
				text: 'Two numbers cannot describe every wobble, so the decoder draws the typical drawing for that code.',
				correct: true,
				why: 'The code is all the decoder gets. Training made the two numbers keep what matters most for redrawing, mainly which shape it is. Details that differ from drawing to drawing do not fit, so the decoder draws what drawings with that code usually look like.'
			},
			{
				text: 'The encoder cleans up the drawing before it squeezes it.',
				why: 'The encoder\'s output is just two numbers, not a picture. There is no cleaned-up drawing anywhere inside the network.'
			},
			{
				text: 'The network was trained on smooth drawings only.',
				why: 'It was trained on wobbly drawings made by the same generator as these. It saw thousands of different wobbles, which is exactly why it could not keep any one of them.'
			}
		]
	},
	{
		kind: 'task',
		prompt: 'The point is in the circles now. Move it so that the [[param:decoder]] draws a **cross**.',
		params: { show: 3, sample: 1, scenario: 9 },
		controls: ['zx', 'zy'],
		goal: ({ readouts }) => readouts.nearest === 'cross',
		hint: 'Find the cluster of crosses on the map, then drag the point into it, or move it there with the sliders.',
		solved:
			'Inside the crosses, the decoder draws a cross. The two numbers are everything the decoder knows, so the place on the map decides the drawing.'
	},
	{
		kind: 'choice',
		prompt:
			'What does the [[param:decoder]] draw for a point halfway between the circles and the squares, where no training drawing landed?\n\nDecide first, then move the point there and look.',
		params: { show: 3, sample: 1, scenario: 10 },
		controls: ['zx', 'zy'],
		options: [
			{
				text: 'Nothing: the decoder only works for codes it saw in training.',
				why: 'The decoder is a function of two numbers. It gives a drawing for every point of the map, whether or not a training drawing ever landed there.'
			},
			{
				text: 'A circle or a square, picked at random.',
				why: 'Nothing in the decoder is random. The same point always gives the same drawing.'
			},
			{
				text: 'Something partly circle and partly square, unlike any training drawing.',
				correct: true,
				why: 'Nearby codes give similar drawings, so between two clusters the drawing changes gradually from one shape to the other. Picking a point no drawing came from is how a model makes something new.'
			},
			{
				text: 'An empty image, because no drawing landed there.',
				why: 'The decoder does not look up stored drawings. It computes a picture from the two numbers, and between the clusters that picture is a blend, not a blank.'
			}
		]
	}
];

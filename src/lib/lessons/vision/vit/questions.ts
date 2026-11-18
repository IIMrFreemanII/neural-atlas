import type { Question } from '../../types.ts';

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'Why does every patch get a [[param:position vector]]?\n\nTurn **Position vectors** off, press **Shuffle patches** and watch the answer, then try it with the vectors on.',
		params: { show: 3, scenario: 9, sample: 4 },
		controls: ['positions', 'shuffle', 'restore'],
		legend: ['Ink', 'Class token'],
		readouts: ['answer'],
		options: [
			{
				text: 'To make patches with more ink count more.',
				why: 'How much ink a patch has is already in its own 32 numbers. The position vector says nothing about ink: it is the same for a patch whatever the drawing shows.'
			},
			{
				text: 'Attention alone cannot tell where a patch came from, so without it a shuffled image looks the same.',
				correct: true,
				why: 'Attention compares tokens without knowing their order. With no position vectors the model sees a bag of patches, so moving patches around changes nothing. The position vector is how the place in the image gets into the token.'
			},
			{
				text: 'So that patches only look at their neighbours.',
				why: 'Position vectors do not limit what a patch can look at. Every patch can attend to every other, near or far, and the trained model does not mostly look at neighbours.'
			},
			{
				text: 'To tell the model which shape to expect.',
				why: 'The position vectors are the same for every drawing. They carry where a patch is, not what the drawing is.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'Find the block and head where the [[param:class token]] looks most at the strokes of the drawing.\n\nStart with **Block 1, Head 1**, and compare how much [[grad:attention]] falls on the ink in each setting.',
		params: { show: 5, scenario: 10, layer: 1, head: 1, focus: 0 },
		controls: ['layer', 'head'],
		goal: ({ params }) => params.layer === 2 && params.head === 1,
		example: { params: { layer: 2, head: 1 } },
		hint: 'Try all four settings. Look for the one where almost all of the amber lands on patches that have ink, and little on the empty ones.',
		solved:
			'In block 2, head 1 the class token puts most of its attention on the patches with ink. Those are the patches that tell the shapes apart, so this head gathers the information that decides the answer.'
	},
	{
		kind: 'choice',
		prompt:
			'A patch with no ink at all still becomes a token. What does its vector carry?\n\nLook at the strips of the empty patches in the corners.',
		params: { show: 3, scenario: 11, sample: 1 },
		controls: ['sample'],
		legend: ['Ink', 'Class token'],
		readouts: ['answer'],
		options: [
			{
				text: 'Nothing: it is all zeros.',
				why: 'The vector is not zero. The embedding adds a bias to every patch, and the position vector is added on top, so even an empty patch has 32 numbers that are not zero.'
			},
			{
				text: 'The shape the model expects.',
				why: 'Nothing about the shape is known at that point. The vector is made from one patch and its place, before any patch has looked at another.'
			},
			{
				text: 'Where it is in the image, and nothing about ink.',
				correct: true,
				why: 'With no ink the 16 pixels are all 0, so only the bias is left of the embedding. The position vector adds where the patch sits. That is why empty patches in different places still have different tokens.'
			},
			{
				text: 'A copy of the class token.',
				why: 'The class token starts the same for every image and belongs to no patch. An empty patch has its own token, made from its place.'
			}
		]
	}
];

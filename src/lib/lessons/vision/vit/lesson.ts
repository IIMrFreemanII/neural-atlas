import type { LessonDef } from '../../types.ts';
import { questions } from './questions.ts';
import { createVitScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'vit',
	track: 'vision',
	title: 'Vision transformer',
	summary: 'Cut a drawing into patches and watch them attend to each other, like words in a sentence.',
	defaults: { sample: 4, show: 1, layer: 2, head: 1, positions: 1, focus: 0, scenario: 1 },
	discrete: ['sample', 'show', 'layer', 'head', 'positions', 'focus', 'scenario'],
	controls: [
		{
			kind: 'select',
			key: 'sample',
			label: 'Drawing',
			options: [
				{ value: 1, label: 'Circle' },
				{ value: 2, label: 'Square' },
				{ value: 3, label: 'Triangle' },
				{ value: 4, label: 'Cross' }
			]
		},
		{
			kind: 'select',
			key: 'layer',
			label: 'Block',
			options: [
				{ value: 1, label: 'Block 1' },
				{ value: 2, label: 'Block 2' }
			]
		},
		{
			kind: 'select',
			key: 'head',
			label: 'Head',
			options: [
				{ value: 1, label: 'Head 1' },
				{ value: 2, label: 'Head 2' }
			]
		},
		{ kind: 'toggle', key: 'positions', label: 'Position vectors' },
		{ kind: 'action', id: 'shuffle', label: 'Shuffle patches', icon: 'shuffle' },
		{ kind: 'action', id: 'restore', label: 'Put back', icon: 'reset' },
		{ kind: 'action', id: 'clear', label: 'Clear', icon: 'reset' }
	],
	readouts: [{ key: 'answer', label: 'Answer', tone: 'ok' }],
	legend: [
		{ tone: 'data', label: 'Ink' },
		{ tone: 'param', label: 'Class token' },
		{ tone: 'grad', label: 'Attention' }
	],
	steps: [
		{
			title: 'Cut the image into patches',
			body: 'A transformer reads a sequence of tokens. A vision transformer makes one from an image by cutting it into a grid of [[data:patches]]: here 16 patches of 4 x 4 pixels.\n\nThe patches are read in order, row by row, and each one becomes a token in a row. A seventeenth token, the [[param:class token]], goes first. You will see what it is for in a moment.',
			params: { show: 1, scenario: 1 },
			controls: ['sample']
		},
		{
			title: 'Each patch becomes a vector',
			body: 'A token is a list of numbers, so each patch is turned into one. Its 16 pixels are weighed into 32 numbers, the same way words become embedding vectors in the language lessons.\n\nEach strip is one patch vector: [[data:cyan]] cells are positive and pink cells are negative. Switch **Drawing** and watch the strips change with the pixels.',
			params: { show: 2, scenario: 2 },
			controls: ['sample'],
			deepDive: {
				note: 'The same matrix E and bias b are used for every patch. The class token is a vector that is learned in training and is the same for every image.',
				math: [{ tex: 'e_p = E\\, x_p + b', caption: 'Patch p: 16 pixels x_p become 32 numbers e_p' }]
			}
		},
		{
			title: 'Where is each patch?',
			body: 'Attention compares tokens without knowing their order, so each patch also gets a learned [[param:position vector]], added to its own. Without it, the model could not tell the top of the image from the bottom.\n\nTurn **Position vectors** off. The cross is now taken for another shape, and **Shuffle patches** no longer changes the answer at all: without positions the model sees a bag of patches. Press **Put back** to undo the shuffle.',
			params: { show: 3, scenario: 3, sample: 4 },
			controls: ['positions', 'shuffle', 'restore'],
			deepDive: {
				note: 'Run without its position vectors, the same trained model gets about a third of 1000 new drawings right, close to guessing among four shapes. With them it gets about 100% right.',
				math: [{ tex: 'z_p = e_p + \\text{pos}_p', caption: 'The token that enters block 1' }]
			}
		},
		{
			title: 'Patches look at each other',
			body: 'In each block, every token looks at every other token and decides how much each one matters. The [[grad:amber]] shows where the outlined patch looks, and how strongly.\n\nClick any patch to see where it looks. Any patch can look at any other, near or far, from the very first block. Change **Block** and **Head** to compare.',
			params: { show: 4, scenario: 4, focus: 6 },
			controls: ['sample', 'layer', 'head'],
			deepDive: {
				note: 'Each block has 2 heads, and the model has 2 blocks. Every head has its own query, key and value weights, so each looks for a different link between patches. This is the same attention as in the language lessons, with patches in place of words.',
				math: [
					{ tex: '\\alpha_{ij} = \\frac{e^{\\, q_i \\cdot k_j / \\sqrt{d}}}{\\sum_{m} e^{\\, q_i \\cdot k_m / \\sqrt{d}}}', caption: 'How much token i looks at token j (softmax over all 17 tokens)' },
					{ tex: 'o_i = \\sum_{j} \\alpha_{ij} \\, v_j', caption: 'The new vector for token i' }
				]
			}
		},
		{
			title: 'A summary token',
			body: 'The [[param:class token]] belongs to no patch. It starts the same for every image and gathers from the patches by attention.\n\nIn block 2, head 1 it looks mostly at the patches with ink, the ones that tell the shapes apart. Try the other heads and the other block, and compare where the [[grad:attention]] falls.',
			params: { show: 5, scenario: 5, layer: 2, head: 1, focus: 0 },
			controls: ['sample', 'layer', 'head']
		},
		{
			title: 'From summary to answer',
			body: 'After both blocks, only the [[param:class token]] is used. A small layer turns its 32 numbers into four scores, and softmax turns the scores into [[ok:probabilities]].\n\nIt says about 92% for every shape it recognizes. Training used label smoothing, which keeps the model from claiming 100%.',
			params: { show: 6, scenario: 6 },
			controls: ['sample'],
			deepDive: {
				note: 'Label smoothing trains toward 92.5% for the right shape and 2.5% for each of the others, instead of 100% and 0%. The scores become less extreme, and so does the confidence.',
				math: [
					{ tex: 'z = W_h \\, \\mathrm{LN}(c) + b_h', caption: 'Four scores from the final class token c' },
					{ tex: 'p_k = \\frac{e^{z_k}}{\\sum_{j} e^{z_j}}', caption: 'Softmax' }
				]
			}
		},
		{
			title: 'Draw your own',
			body: 'Press **Clear** and draw on the left with the pointer. Everything updates as you draw: the tokens, the attention and the answer.\n\nThe model only knows four shapes, so even a scribble is forced into one of them, with the same confidence.',
			params: { show: 7, scenario: 7 },
			controls: ['clear', 'layer', 'head']
		},
		{
			title: 'Patches versus kernels',
			body: 'A CNN\'s first layer sees a 3 x 3 neighbourhood and widens step by step. Here every patch can reach every other at once.\n\nReal vision transformers cut a 224 x 224 photo into 196 patches of 16 x 16 pixels and learn from millions of images. This one has about 18 thousand weights, trained by this repo on generated drawings.',
			params: { show: 6, scenario: 8 },
			controls: ['sample'],
			deepDive: {
				note: 'The model is trained by scripts/train-vit.py in this repo, on drawings made by the same generator as the ones you see here.'
			}
		}
	],
	questions,
	createScene: createVitScene
};

export default lesson;

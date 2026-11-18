import type { LessonDef } from '../../types.ts';
import { questions } from './questions.ts';
import { createAutoencoderScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'autoencoder',
	track: 'vision',
	title: 'Latent space',
	summary: 'Squeeze drawings down to two numbers, then walk the map of those numbers and watch new drawings appear.',
	defaults: { sample: 1, show: 3, zx: 0, zy: 0, pin: 0, scenario: 1 },
	discrete: ['show', 'zx', 'zy', 'pin', 'scenario'],
	controls: [
		{
			kind: 'select',
			key: 'sample',
			label: 'Shape',
			options: [
				{ value: 1, label: 'Circle' },
				{ value: 2, label: 'Square' },
				{ value: 3, label: 'Triangle' },
				{ value: 4, label: 'Cross' }
			]
		},
		{ kind: 'action', id: 'clear', label: 'Clear', icon: 'reset' },
		{ kind: 'slider', key: 'zx', label: 'Code x', min: -1, max: 1, step: 0.01, tone: 'data', digits: 2 },
		{ kind: 'slider', key: 'zy', label: 'Code y', min: -1, max: 1, step: 0.01, tone: 'data', digits: 2 }
	],
	readouts: [{ key: 'nearest', label: 'Nearest drawings', tone: 'data' }],
	legend: [
		{ tone: 'text', label: 'Drawing' },
		{ tone: 'param', label: 'Encoder, decoder' },
		{ tone: 'data', label: 'Code, rebuilt' }
	],
	steps: [
		{
			title: 'Squeeze, then rebuild',
			body: 'An **autoencoder** is two small networks back to back. The [[param:encoder]] squeezes a drawing of 256 pixels down to just two numbers, its [[data:code]]. The [[param:decoder]] gets only those two numbers and draws the picture again.\n\nNobody told the network what the two numbers should mean. It was trained on thousands of drawings with one goal: the rebuilt drawing should match the original.',
			params: { show: 1, scenario: 1 },
			controls: ['sample'],
			readouts: [],
			deepDive: {
				note: 'Both halves are small fully connected networks: 256 pixels, 32 hidden units, 2 code numbers, 32 hidden units, 256 pixels. The hidden layers and the code use tanh, so each code number is between -1 and 1. The output uses a sigmoid, so each pixel is between 0 and 1.',
				math: [
					{ tex: 'z = f_{\\text{enc}}(x) \\in (-1, 1)^2', caption: 'Encoder: 256 pixels to a code of 2 numbers' },
					{ tex: '\\hat{x} = f_{\\text{dec}}(z)', caption: 'Decoder: the code back to 256 pixels' },
					{
						tex: 'L = -\\frac{1}{256} \\sum_{i=1}^{256} \\big( x_i \\log \\hat{x}_i + (1 - x_i) \\log (1 - \\hat{x}_i) \\big)',
						caption: 'Training lowers the gap between each pixel and its rebuilt value'
					}
				],
				code: {
					lang: 'ts',
					source: 'const code = encode(weights, drawing); // [x, y]\nconst rebuilt = decode(weights, code); // 256 pixels'
				}
			}
		},
		{
			title: 'What survives the squeeze',
			body: 'Switch between the shapes. The [[data:rebuilt]] drawing keeps the kind of shape, but the wobbles and the exact strokes are gone.\n\nTwo numbers cannot hold everything about 256 pixels. Training kept what helps most to redraw a picture, and that is mainly which shape it is.',
			params: { show: 1, scenario: 2 },
			controls: ['sample'],
			readouts: []
		},
		{
			title: 'The code is a place',
			body: 'Two numbers are a position on a plane: the first goes across, the second goes up. So every drawing\'s [[data:code]] is a point on this map.\n\nSwitch shapes and watch the point jump. Each shape lands somewhere else.',
			params: { show: 2, scenario: 3 },
			controls: ['sample'],
			readouts: []
		},
		{
			title: 'A map of all drawings',
			body: 'Here are the codes of 800 different drawings, colored by shape. The encoder was never told the shape names, yet drawings of the same shape landed together.\n\nThis plane is the network\'s **latent space**. Latent means hidden: it is a space inside the network, not something it was shown. Press **Clear** and draw your own shape on the left to see where it lands.',
			params: { show: 3, scenario: 4 },
			controls: ['sample', 'clear']
		},
		{
			title: 'Walk the map',
			body: 'Now leave the encoder out. Drag the point across the map, or use the sliders, and the [[param:decoder]] draws whatever belongs at that place.\n\nInside a cluster you get a clean shape. Watch how the drawing changes as the point crosses from one cluster to the next.',
			params: { show: 3, scenario: 5 },
			controls: ['zx', 'zy']
		},
		{
			title: 'Drawings nobody made',
			body: 'This point lies between two clusters, where no training drawing landed. The decoder still draws something: a blend, part one shape and part the other.\n\nThat is the idea behind generating new images: pick a point no example came from, and let the decoder draw it.',
			params: { show: 3, pin: 1, zx: 0.32, zy: -0.21, scenario: 6 },
			controls: ['zx', 'zy']
		},
		{
			title: 'Every point is a drawing',
			body: 'Here the decoder has drawn a picture at every place of a grid over the map. The shapes turn into each other gradually, because nearby codes give similar drawings.\n\nThe next lesson, diffusion, generates images another way: it starts from pure noise and removes it step by step. Large image generators combine both ideas and run diffusion inside a latent space, with many more than two numbers.',
			params: { show: 4, scenario: 7 },
			controls: ['zx', 'zy']
		}
	],
	questions,
	createScene: createAutoencoderScene
};

export default lesson;

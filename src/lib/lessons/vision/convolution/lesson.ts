import type { LessonDef } from '../../types.ts';
import { questions } from './questions.ts';
import { createConvolutionScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'convolution',
	track: 'vision',
	title: 'Convolution',
	summary: 'Slide a small grid of weights over an image and watch it pick out edges, blur and detail.',
	defaults: { image: 0, kernel: 3, scan: 1, stride: 1, pad: 0, calc: 1, scenario: 1 },
	discrete: ['scan', 'stride', 'calc', 'scenario'],
	controls: [
		{
			kind: 'select',
			key: 'image',
			label: 'Image',
			options: [
				{ value: 0, label: 'Shapes' },
				{ value: 1, label: 'Digit' },
				{ value: 2, label: 'Gradient' },
				{ value: 3, label: 'Blank' }
			]
		},
		{
			kind: 'select',
			key: 'kernel',
			label: 'Kernel',
			options: [
				{ value: 0, label: 'Same' },
				{ value: 1, label: 'Blur' },
				{ value: 2, label: 'Sharpen' },
				{ value: 3, label: 'Vertical' },
				{ value: 4, label: 'Horizontal' },
				{ value: 5, label: 'Outline' }
			]
		},
		{ kind: 'action', id: 'play', label: 'Slide the kernel', icon: 'play' },
		{ kind: 'action', id: 'clear', label: 'Clear image', icon: 'reset' },
		{ kind: 'slider', key: 'stride', label: 'Stride', min: 1, max: 2, step: 1, digits: 0 },
		{ kind: 'toggle', key: 'pad', label: 'Pad the edges' }
	],
	readouts: [
		{ key: 'size', label: 'Output' },
		{ key: 'sum', label: 'This position', digits: 1 }
	],
	legend: [
		{ tone: 'text', label: 'Image' },
		{ tone: 'param', label: 'Kernel' },
		{ tone: 'data', label: 'Plus' },
		{ tone: 'neg', label: 'Minus' }
	],
	steps: [
		{
			title: 'An image is a grid of numbers',
			body: 'Each pixel of the [[text:image]] on the left is one number, from 0 (black) to 1 (white). Hover a pixel to read its value.\n\nThis picture is tiny on purpose: real photos have millions of pixels and three color channels, while this one has 576 pixels and one channel.',
			params: { scenario: 1, kernel: 0, calc: 0, scan: 1 },
			controls: ['image']
		},
		{
			title: 'A kernel looks at nine pixels at once',
			body: 'The [[param:kernel]] is a 3 x 3 grid of weights. Lay it on a patch of the image, multiply each pixel by the weight on top of it, and add the nine products. That sum is one number of the output.\n\nHover the image to move the kernel and watch the arithmetic below follow it.',
			params: { scenario: 2, kernel: 3, calc: 1, scan: 0 },
			controls: []
		},
		{
			title: 'Slide it everywhere',
			body: 'Press **Slide the kernel**. The same nine weights visit every position, one after another, and each sum fills one cell of the output map.\n\nNothing about the kernel changes on the way. Only the patch under it does.',
			params: { scenario: 3, kernel: 3, calc: 1, scan: 0 },
			controls: ['play'],
			deepDive: {
				note: 'Deep learning calls this operation convolution, although strictly it is cross-correlation: the kernel is not flipped. S is the stride and p the padding.',
				math: [
					{
						tex: '\\text{out}(r,c)=\\sum_{i=0}^{2}\\sum_{j=0}^{2} x(rS+i-p,\; cS+j-p)\\,K(i,j)',
						caption: 'Pixels outside the image count as zero.'
					}
				],
				code: {
					lang: 'ts',
					source:
						'for (let r = 0; r < outH; r++)\n  for (let c = 0; c < outW; c++) {\n    let sum = 0;\n    for (let i = 0; i < 3; i++)\n      for (let j = 0; j < 3; j++) sum += x(r + i, c + j) * K[i][j];\n    out[r][c] = sum;\n  }'
				}
			}
		},
		{
			title: 'Different weights find different things',
			body: 'The vertical-edge kernel answers [[data:positive]] where the image gets brighter to the right, [[neg:negative]] where it gets darker, and zero on flat areas.\n\nTry the other kernels and images. Blur averages, sharpen exaggerates, outline lights up every border.',
			params: { scenario: 4, kernel: 3, calc: 1, scan: 1 },
			controls: ['kernel', 'image']
		},
		{
			title: 'Make your own',
			body: 'Click a weight in the [[param:kernel]] to change it and watch the output react. Each click steps the weight up, and it wraps around after 2.\n\nDrag on the image to draw your own picture. Choose Blank or **Clear image** first for a clean page.',
			params: { scenario: 5, image: 0, kernel: 3, calc: 1, scan: 1 },
			controls: ['kernel', 'image', 'clear']
		},
		{
			title: 'Stride and padding set the output size',
			body: 'With no padding the output is 2 pixels smaller in each direction, because the kernel cannot hang over the edge. **Pad the edges** adds a border of zeros so the output keeps the image size.\n\nA **Stride** of 2 visits every other position and halves the output.',
			params: { scenario: 6, kernel: 3, calc: 1, scan: 1 },
			controls: ['stride', 'pad', 'play'],
			deepDive: {
				note: 'n is the image width, k the kernel width, p the padding and S the stride. Here n = 24 and k = 3.',
				math: [
					{
						tex: '\\text{out}=\\left\\lfloor \\frac{n+2p-k}{S} \\right\\rfloor + 1',
						caption: 'For example 24 with no padding and stride 1 gives 22.'
					}
				]
			}
		},
		{
			title: 'The weights can be learned',
			body: 'Nobody picks kernels by hand in a neural network. It starts from random weights, and training adjusts them, exactly like the weights of a neuron.\n\nThe next lesson stacks many learned kernels, so each one can specialize in its own pattern.',
			params: { scenario: 7, kernel: 5, calc: 1, scan: 1 },
			controls: ['kernel', 'image']
		}
	],
	questions,
	createScene: createConvolutionScene
};

export default lesson;

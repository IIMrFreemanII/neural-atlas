import type { LessonDef } from '../../types.ts';
import { questions } from './questions.ts';
import { createCnnFeaturesScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'cnn-features',
	track: 'vision',
	title: 'CNN feature maps',
	summary: 'Watch a small trained network turn a drawing into feature maps, and the maps into an answer.',
	defaults: { sample: 1, show: 4, scenario: 1 },
	discrete: ['show', 'scenario'],
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
		{ kind: 'action', id: 'clear', label: 'Clear', icon: 'reset' }
	],
	readouts: [{ key: 'guess', label: 'It sees', tone: 'ok' }],
	legend: [
		{ tone: 'text', label: 'Input' },
		{ tone: 'data', label: 'Maps' },
		{ tone: 'param', label: 'Kernel' },
		{ tone: 'ok', label: 'Answer' }
	],
	steps: [
		{
			title: 'Many kernels, many maps',
			body: 'The last lesson slid one kernel over an image. A convolution layer uses several at once, here 4. Each one produces its own **feature map**: a picture of where its pattern appears in the drawing.\n\nThis network is tiny and was really trained, on thousands of generated drawings of four shapes. Its [[param:kernels]] were learned, not designed.',
			params: { show: 1, scenario: 1 },
			controls: ['sample']
		},
		{
			title: 'Keep only what fired',
			body: 'After each kernel, negative answers are set to zero (ReLU, as in the activation lesson). A [[data:map]] then shows only where its pattern **is** present.\n\nSwitch between shapes and watch which map responds to which strokes: some favor one direction, others another.',
			params: { show: 1, scenario: 2 },
			controls: ['sample']
		},
		{
			title: 'Pooling shrinks the map',
			body: 'Each 2 x 2 block of a map keeps only its largest value. The [[data:pooled maps]] are 4 times smaller, and a small shift of the drawing changes them very little.\n\nSwitch shapes and compare each pooled map with the map it came from.',
			params: { show: 2, scenario: 3 },
			controls: ['sample'],
			deepDive: {
				note: 'Max pooling slides a 2 x 2 window with stride 2 and keeps the largest value in each window.',
				math: [{ tex: 'p_{i,j} = \\max_{0 \\le a, b < 2} \; m_{2i+a,\\, 2j+b}', caption: 'Max pooling' }]
			}
		},
		{
			title: 'The second layer looks at maps, not pixels',
			body: 'Each second-layer kernel reads all four pooled maps at once, so it can respond to combinations: two strokes meeting, a curve, a crossing.\n\nHover any cell of any map. The [[ok:green square]] on the drawing is everything that cell can see. It grows with depth: 3 x 3, then 4 x 4, then 8 x 8 pixels.',
			params: { show: 3, scenario: 4 },
			controls: ['sample']
		},
		{
			title: 'From maps to an answer',
			body: 'Each second-layer map reports only its strongest response, marked with a ring. Those 8 numbers are weighed into four scores and turned into [[ok:probabilities]], as in the sampling lesson.\n\nThe longest bar is the network\'s answer.',
			params: { show: 4, scenario: 5 },
			controls: ['sample'],
			deepDive: {
				note: 'Global max pooling keeps one number per map. A dense layer weighs those 8 numbers into 4 scores, and softmax turns the scores into probabilities.',
				math: [
					{ tex: 'g_k = \\max_{i,j} \; a^{(2)}_{k,i,j}', caption: 'Global max pooling' },
					{ tex: 'z_c = \\sum_{k=1}^{8} W_{c,k}\\, g_k + b_c', caption: 'Dense layer' },
					{ tex: 'p_c = \\frac{e^{z_c}}{\\sum_{d} e^{z_d}}', caption: 'Softmax' }
				],
				code: {
					lang: 'ts',
					source: 'const t = forward(weights, image);\nconst answer = t.probs.indexOf(Math.max(...t.probs));'
				}
			}
		},
		{
			title: 'Draw your own',
			body: 'Press **Clear** and draw on the left with the pointer. Watch which maps light up as you add strokes.\n\nThe network only knows four shapes, so anything you draw is forced into one of them, even a scribble.',
			params: { show: 4, scenario: 6 },
			controls: ['clear', 'sample']
		},
		{
			title: 'Bigger networks, same idea',
			body: 'Real image networks stack many more layers, with hundreds of kernels in each, and learn from millions of photos.\n\nTheir early layers also find edges and color blobs. Later layers respond to textures, then to parts, then to whole objects.',
			params: { show: 4, scenario: 7 },
			controls: ['sample', 'clear']
		}
	],
	questions,
	createScene: createCnnFeaturesScene
};

export default lesson;

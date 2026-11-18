import type { LessonDef } from '../../types.ts';
import { questions } from './questions.ts';
import { createDiffusionScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'diffusion',
	track: 'vision',
	title: 'Diffusion',
	summary: 'Turn a pattern into noise, then follow the denoising direction backward to generate it from nothing.',
	defaults: { pattern: 0, t: 0, steps: 40, field: 0, scenario: 1 },
	discrete: ['steps', 'scenario'],
	controls: [
		{ kind: 'slider', key: 't', label: 'Noise level', min: 0, max: 1, step: 0.01, tone: 'grad', digits: 2 },
		{ kind: 'action', id: 'noise', label: 'Add noise', icon: 'play' },
		{ kind: 'action', id: 'generate', label: 'Generate', icon: 'play' },
		{ kind: 'action', id: 'again', label: 'New noise', icon: 'shuffle' },
		{ kind: 'slider', key: 'steps', label: 'Steps', min: 1, max: 100, step: 1, tone: 'param', digits: 0 },
		{
			kind: 'select',
			key: 'pattern',
			label: 'Pattern',
			options: [
				{ value: 0, label: 'Spiral' },
				{ value: 1, label: 'Face' },
				{ value: 2, label: 'Rings' },
				{ value: 3, label: 'Letters' }
			]
		},
		{ kind: 'toggle', key: 'field', label: 'Show directions' }
	],
	readouts: [
		{ key: 'level', label: 'Noise', tone: 'grad' },
		{ key: 'step', label: 'Step' }
	],
	legend: [
		{ tone: 'data', label: 'Samples' },
		{ tone: 'grad', label: 'Noise' },
		{ tone: 'ok', label: 'Direction' }
	],
	steps: [
		{
			title: 'The data is a pattern',
			body: 'Each [[data:dot]] stands for one training example. A real image is a point with millions of coordinates, one per pixel value. Here every example has just two, so the whole dataset fits on screen.\n\nThe pattern is what a model should learn to produce. Pick another one to see.',
			params: { t: 0, scenario: 1 },
			controls: ['pattern']
		},
		{
			title: 'Add noise, a little at a time',
			body: 'Press **Add noise** or drag the **Noise level** slider. Each dot is nudged by a random amount and pulled toward the center, so the pattern blurs.\n\nAt the end only a shapeless [[grad:blob]] is left, and it looks the same whatever the pattern was.',
			params: { t: 0, scenario: 2 },
			controls: ['noise', 't'],
			deepDive: {
				note: 'Every dot keeps its own random noise, so the slider moves it along a straight path. The two weights always satisfy alpha squared plus sigma squared equals one.',
				math: [
					{
						tex: 'x_t = \\alpha_t\\, x_0 + \\sigma_t\\, \\varepsilon,\\qquad \\alpha_t=\\cos\\tfrac{\\pi t}{2},\\ \\ \\sigma_t=\\sin\\tfrac{\\pi t}{2},\\ \\ \\varepsilon\\sim\\mathcal{N}(0, I)',
						caption: 'The noised dot at level t.'
					}
				]
			}
		},
		{
			title: 'Destroying is easy. Can it be undone?',
			body: 'Drag the slider back from the blob. From a pure [[grad:blob]] you cannot tell where a dot came from.\n\nBut at any noise level there is a best guess of which way "less noisy" lies, and that guess is the whole trick.',
			params: { t: 1, scenario: 3 },
			controls: ['t']
		},
		{
			title: 'The denoising direction',
			body: 'The [[ok:arrows]] show, for every position, where a dot there most likely came from. At high noise they all point vaguely inward. As the noise drops they split and point at the nearest piece of the pattern.\n\nToy part: here the arrows are computed exactly, because the pattern is known. A real diffusion model is a neural network trained to predict them, or equivalently to predict the noise that was added.',
			params: { t: 0.6, field: 1, scenario: 4 },
			controls: ['t', 'field'],
			deepDive: {
				note: 'For a mixture of round Gaussians the best guess has a closed form. Each center k gets a weight r_k (how likely the dot came from it), and the guess blends the pull toward every center.',
				math: [
					{
						tex: '\\hat{x}_0 = \\mathbb{E}[x_0 \\mid x_t] = \\sum_k r_k\\Big(\\mu_k + \\tfrac{\\alpha_t s^2}{v_t}\\,(x_t - \\alpha_t \\mu_k)\\Big),\\qquad v_t = \\alpha_t^2 s^2 + \\sigma_t^2',
						caption: 's is the spread of each bump. The arrow runs from x_t toward the guess.'
					},
					{
						tex: '\\nabla_x \\log p_t(x) = \\frac{\\alpha_t\\,\\hat{x}_0 - x}{\\sigma_t^2}',
						caption: 'The same direction, seen as the gradient of the log density.'
					}
				]
			}
		},
		{
			title: 'Generate: start from pure noise',
			body: 'Press **Generate**. Fresh random dots follow the arrows in 40 small steps, and the pattern appears.\n\nNothing was copied. These dots are new samples that happen to land on the pattern.',
			params: { t: 1, steps: 40, scenario: 5 },
			controls: ['generate', 'again'],
			deepDive: {
				note: 'Each step takes the best guess of the clean dot, works out the noise that would explain the current dot, and re-noises the guess to the next, lower level.',
				math: [
					{
						tex: '\\hat\\varepsilon = \\frac{x_t - \\alpha_t \\hat{x}_0}{\\sigma_t},\\qquad x_s = \\alpha_s\\, \\hat{x}_0 + \\sigma_s\\, \\hat\\varepsilon\\quad (s < t)',
						caption: 'One deterministic reverse step.'
					}
				],
				code: {
					lang: 'ts',
					source:
						'for (let i = 0; i < steps; i++) {\n\tconst tFrom = 1 - i / steps;\n\tconst tTo = 1 - (i + 1) / steps;\n\treverseStep(mixture, points, tFrom, tTo);\n}'
				}
			}
		},
		{
			title: 'Few steps or many',
			body: 'Set **Steps** and press **Generate**. With 1 step every dot jumps to the average of the pattern. A few steps give a rough shape, and more steps give a sharp one.\n\nMore steps cost more computation: a real model runs its network once per step.',
			params: { t: 1, steps: 4, scenario: 6 },
			controls: ['steps', 'generate', 'again']
		},
		{
			title: 'Same noise in, same picture out',
			body: 'Generate twice from the same dots and you get the same picture. **New noise** draws different starting dots, so a different picture appears.\n\nIn real image models the network also receives the text prompt, which steers the arrows toward pictures that match it.',
			params: { t: 1, steps: 40, scenario: 7 },
			controls: ['generate', 'again', 'pattern']
		}
	],
	questions,
	createScene: createDiffusionScene
};

export default lesson;

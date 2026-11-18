import type { LessonDef } from '../../types.ts';
import { createActivationsScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'activations',
	track: 'fundamentals',
	title: 'Activation functions',
	summary: 'The bends that let a network of simple neurons learn curves instead of only straight lines.',
	defaults: { act: 0, z: 1, showDeriv: 0, mode: 0, units: 3 },
	discrete: ['units'],
	controls: [
		{
			kind: 'select',
			key: 'act',
			label: 'Activation',
			options: [
				{ value: 0, label: 'Linear' },
				{ value: 4, label: 'Step' },
				{ value: 1, label: 'Sigmoid' },
				{ value: 3, label: 'Tanh' },
				{ value: 2, label: 'ReLU' },
				{ value: 5, label: 'Leaky ReLU' }
			]
		},
		{ kind: 'slider', key: 'z', label: 'Input z', min: -5, max: 5, step: 0.1, tone: 'data', digits: 1 },
		{ kind: 'slider', key: 'units', label: 'Hidden units', min: 1, max: 6, step: 1, tone: 'param', digits: 0 }
	],
	readouts: [
		{ key: 'out', label: 'Output', tone: 'data', digits: 2 },
		{ key: 'slope', label: 'Slope', tone: 'grad', digits: 2 },
		{ key: 'err', label: 'Fit error', tone: 'neg', digits: 3 }
	],
	legend: [
		{ tone: 'data', label: 'Function output' },
		{ tone: 'grad', label: 'Slope (derivative)' },
		{ tone: 'param', label: 'Hidden units' },
		{ tone: 'muted', label: 'Target curve' }
	],
	steps: [
		{
			title: 'Why a neuron needs a bend',
			body:
				'With no activation, a neuron’s [[data:output]] is just its input: a straight line. Drag z and the output follows one for one.\n\n' +
				'Stacking straight lines only gives another straight line, so a network made of them can never learn a curve. It needs a bend.',
			params: { act: 0 },
			controls: ['z']
		},
		{
			title: 'Step: all or nothing',
			body:
				'The earliest artificial neurons simply fired: output 1 when the input is zero or above, otherwise 0.\n\n' +
				'The [[grad:slope]] is zero everywhere, so a tiny change in the weights changes nothing, and learning gets no hint about which way to move.',
			params: { act: 4, showDeriv: 1 },
			controls: ['z']
		},
		{
			title: 'Sigmoid: a smooth switch',
			body:
				'Sigmoid squeezes any input into the range 0 to 1 with a smooth rise, so a small nudge now changes the [[data:output]] a little.\n\n' +
				'Far from zero the curve flattens and the [[grad:slope]] fades toward zero. In deep networks that slows learning, a problem called vanishing gradients.',
			params: { act: 1, showDeriv: 1 },
			controls: ['z'],
			deepDive: {
				math: [
					{ tex: '\\sigma(z) = \\frac{1}{1+e^{-z}}' },
					{ tex: "\\sigma'(z) = \\sigma(z)\\,(1-\\sigma(z))", caption: 'The slope is at most 0.25, at z = 0.' }
				]
			}
		},
		{
			title: 'Tanh: centered on zero',
			body:
				'Tanh has the same S shape, but its [[data:output]] runs from -1 to 1 and passes through zero in the middle.\n\n' +
				'Its [[grad:slope]] is steeper near the center, up to 1, so signals fade less than with sigmoid, though both still flatten at the ends.',
			params: { act: 3, showDeriv: 1 },
			controls: ['z'],
			deepDive: {
				math: [{ tex: '\\tanh(z) = \\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}' }]
			}
		},
		{
			title: 'ReLU: simple and fast',
			body:
				'ReLU passes positive inputs through unchanged and outputs zero for negative ones. It is cheap to compute, and the [[grad:slope]] is exactly 1 on the positive side, so gradients do not fade.\n\n' +
				'A unit stuck on the negative side stops learning. Leaky ReLU avoids that with a small slope there. Try both.',
			params: { act: 2, showDeriv: 1 },
			controls: ['z', 'act'],
			deepDive: {
				math: [{ tex: '\\mathrm{ReLU}(z)=\\max(0,z)' }],
				code: {
					lang: 'ts',
					source:
						'const relu = (z: number) => Math.max(0, z);\nconst leakyRelu = (z: number) => (z >= 0 ? z : 0.1 * z);'
				}
			}
		},
		{
			title: 'Bends build curves',
			body:
				'Each [[param:hidden unit]] contributes one bent piece. The output adds the pieces up, each scaled by a weight, to approach the target curve.\n\n' +
				'Switch to Linear and the sum stays a straight line, however many units you add.',
			params: { mode: 1, act: 2, units: 3 },
			controls: ['units', 'act']
		},
		{
			title: 'More units, finer shape',
			body:
				'More [[param:units]] means more bends to place, so the [[data:network output]] hugs the target more closely and the error drops.\n\n' +
				'This is why networks with enough non-linear units can approximate very complicated functions. Try each activation and compare.',
			params: { mode: 1, act: 3, units: 6 },
			controls: ['units', 'act']
		}
	],
	createScene: createActivationsScene
};

export default lesson;

import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'[[ok:Show directions]] is on and the noise level is 0.2. The arrows point at nearby pieces of the pattern. You drag the noise level up to 0.9. What do the arrows do?\n\nDecide first, then check with the slider.',
		params: { t: 0.2, field: 1, scenario: 4, pattern: 1 },
		controls: ['t', 'field'],
		options: [
			{
				text: 'They keep pointing at the nearest piece of the pattern.',
				why: 'At high noise a dot has been pushed far from where it started, so its nearest piece of the pattern says little about its origin. The arrows stop following it.'
			},
			{
				text: 'They all point toward the middle.',
				correct: true,
				why: 'From a very noisy dot, any part of the pattern could be the place it came from. The best guess is then about the middle of the whole pattern, so the arrows point inward. As the noise drops, the guess narrows and they split to the nearest pieces.'
			},
			{
				text: 'They disappear, because there is no best guess at high noise.',
				why: 'A best guess always exists, only a poor one. The arrows are still drawn at 0.9, and they all point inward.'
			},
			{
				text: 'They point away from the middle, toward where more noise lies.',
				why: 'The arrows show where a dot most likely came from, which is the direction of less noise. Following them backward is how a dot is cleaned up.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'Press **Generate** and watch where the dots end. Then make one run end with every dot at the same single spot in the middle, not on the pattern.',
		params: { t: 1, steps: 12, scenario: 6, pattern: 0 },
		controls: ['steps', 'generate'],
		goal: ({ params: p, readouts: r }) => shown(p.steps, 0) === 1 && r.step === '1 / 1',
		hint: 'Each step moves the dots along the arrows by a small amount. Think about the shortest possible run, and what the first jump from pure noise can know.',
		solved:
			'With 1 step every dot jumps from the noise straight to the best guess at noise level 1. There a dot says nothing about where it came from, so the guess is the average of the pattern, the middle, for every dot. More steps let the arrows steer each dot at lower noise.'
	},
	{
		kind: 'choice',
		prompt:
			'You press **Generate** and the pattern appears. Without touching anything else, you press **Generate** again. What do you see?\n\nDecide first, then check by pressing it twice.',
		params: { t: 1, steps: 40, scenario: 7, pattern: 0 },
		controls: ['generate', 'again'],
		options: [
			{
				text: 'The dots start from new random noise and land in different places.',
				why: 'New random dots appear only when you press **New noise**. Generate itself uses no randomness: it starts from the dots you have and follows the arrows.'
			},
			{
				text: 'The dots stay where they are, because the pattern is already made.',
				why: 'Generate always starts over from the noise. The dots go back to the blob and the steps run again.'
			},
			{
				text: 'The dots start from the same noise and land in the same places.',
				correct: true,
				why: 'The starting dots are the same and every step is fixed by the arrows, so the run repeats exactly. Only **New noise** changes the starting dots, and with them the final picture.'
			},
			{
				text: 'The pattern comes out sharper, because the dots get a second pass.',
				why: 'Each run starts again from the noise and takes the same number of steps, so the result is the same as before. More steps would give a sharper pattern, not more runs.'
			}
		]
	}
];

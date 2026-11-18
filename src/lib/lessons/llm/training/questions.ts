import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';
import { LOG, curveLossAt } from './data.ts';

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'The [[ok:right token]] at one position is "park". Model A gives it 60% and spreads the other 40% over many tokens. Model B gives it 10% and puts 80% on one wrong token. Which model has the lower [[grad:loss]] at this position?',
		params: { at: 0, chart: 0, story: 0 },
		controls: [],
		options: [
			{
				text: 'Model A, because only the probability given to the right token counts.',
				correct: true,
				why: 'The cost at a position depends only on the probability the model gave to the right token. A full bar costs nothing and a nearly empty bar costs a lot, so 60% costs much less than 10%.'
			},
			{
				text: 'Model B, because a model that is that sure of one token has learned more.',
				why: 'Being sure of a wrong token does not help. Its 80% is on a token that is not "park", so it takes probability away from the right one.'
			},
			{
				text: 'Both the same, because each model is wrong about some tokens.',
				why: 'The loss does not count how many tokens are wrong. It looks at the bar of the right token, and the two bars differ a lot.'
			},
			{
				text: 'There is no way to tell without knowing the whole vocabulary.',
				why: 'The vocabulary does not matter here. The loss at one position is set by the probability of the right token alone.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'The run has 12,000 training steps in all. Find a checkpoint where the [[grad:loss]] is already **below 4** although **less than 5%** of the steps are done.',
		params: { at: 0 },
		controls: ['at'],
		goal: ({ params: p }) => {
			const c = LOG.checkpoints[Math.min(LOG.checkpoints.length - 1, Math.max(0, Math.round(p.at)))];
			return shown(curveLossAt(c.step)) < 4 && c.step < 0.05 * LOG.steps;
		},
		hint: 'Drag the Checkpoint slider and watch the Training steps and Loss chips. Where along the run does the loss fall fastest?',
		solved:
			'Most of the fall comes early. By step 200 the loss is already below 4, and by step 400 it is about 3.3, from about 7.9 at step 0. The other thousands of steps bring it down only to about 2.4.',
		example: { params: { at: 4 } }
	},
	{
		kind: 'choice',
		prompt:
			'Another team trains a model. Its training loss keeps falling, but its loss on [[text:unseen stories]] stops falling and then rises. What is the most likely reason?',
		params: { at: 10, valid: 1, story: 0 },
		controls: ['at'],
		options: [
			{
				text: 'Unseen stories are harder than the training ones, so this is normal.',
				why: 'Unseen stories do read a little differently. But in this lesson the dots follow the training loss closely, and a model that learned patterns that carry over keeps them falling together.'
			},
			{
				text: 'The model is still learning, and unseen stories take longer to show it.',
				why: 'What the model learns from the training text shows on new text at the same time. If the dots stop falling while the training loss keeps falling, the extra gain does not carry over.'
			},
			{
				text: 'It has started to memorize the training stories instead of learning patterns that carry over.',
				correct: true,
				why: 'The training loss can fall by remembering the text it reads. Then the dots stop falling while the training loss keeps going down, because the memorized text does not help on stories the model never saw.'
			},
			{
				text: 'Its weights are being nudged too little to help on new text.',
				why: 'Small nudges would slow both losses down together. A rise on unseen stories while the training loss falls means the weights are fitting the training text in a way that does not carry over.'
			}
		]
	}
];

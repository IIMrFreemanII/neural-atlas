import type { LessonDef } from '../../types.ts';
import { LM_TRAINING } from '../shared/lm-meta.ts';
import { LAST, LOG, choices, nearestCheckpoint } from './data.ts';
import { questions } from './questions.ts';
import { createTrainingScene } from './scene.ts';

const FINAL = LOG.checkpoints[LAST];
const N = LOG.checkpoints.length;
const VOCAB = LOG.vocab.toLocaleString('en-US');
const WEIGHTS = LM_TRAINING.parameters.toLocaleString('en-US');
const STEPS = LOG.steps.toLocaleString('en-US');
const UNIFORM = LOG.uniformLoss.toFixed(2);
const PASSES = (LM_TRAINING.tokensSeen / LM_TRAINING.trainTokens).toFixed(1);

/** The first three words of a story start, with a comma kept next to its word: 'one day, lily ...'. */
function startLabel(prompt: string): string {
	const out: string[] = [];
	let words = 0;
	for (const t of prompt.split(' ')) {
		if (/^[a-z]/.test(t)) {
			if (words === 3) break;
			words++;
			out.push(t);
		} else if (out.length > 0) out[out.length - 1] += t;
	}
	return `${out.join(' ')} ...`;
}

const lesson: LessonDef = {
	id: 'training',
	track: 'llm',
	title: 'Watching a model learn',
	summary: 'Follow the story model from random weights to readable stories, one training step at a time.',
	defaults: { at: 0, prompt: 0, dice: 0, chart: 1, story: 1, probe: 1, valid: 0 },
	discrete: ['at', 'chart', 'story', 'probe', 'valid'],
	controls: [
		{ kind: 'action', id: 'train', label: 'Train', icon: 'play' },
		{ kind: 'action', id: 'reset', label: 'Start over', icon: 'reset' },
		{ kind: 'slider', key: 'at', label: 'Checkpoint', min: 0, max: LAST, step: 1, tone: 'grad', digits: 0 },
		{
			kind: 'select',
			key: 'prompt',
			label: 'Start',
			options: LOG.prompts.map((p, value) => ({ value, label: startLabel(p) }))
		},
		{ kind: 'toggle', key: 'dice', label: 'Roll the dice' }
	],
	readouts: [
		{ key: 'step', label: 'Training steps' },
		{ key: 'loss', label: 'Loss', tone: 'grad', digits: 2 },
		{ key: 'choices', label: 'Like choosing among' }
	],
	legend: [
		{ tone: 'grad', label: 'Loss' },
		{ tone: 'ok', label: 'Right token' },
		{ tone: 'data', label: 'Written by the model' },
		{ tone: 'text', label: 'Given text' }
	],
	steps: [
		{
			title: 'A new model knows nothing',
			body: `Every lesson in this track used the same small language model. Here it is before any training: all ${WEIGHTS} of its weights are random numbers.\n\nAsk it to continue a [[text:story]] and it [[data:writes]] tokens with no pattern. Turn **Roll the dice** off and it always takes its top token instead of drawing one. Then it only repeats itself.`,
			params: { at: 0, chart: 0, probe: 0, dice: 1 },
			controls: ['prompt', 'dice'],
			legend: ['Written by the model', 'Given text'],
			readouts: []
		},
		{
			title: 'The exercise: guess the next token',
			body: `Training is one exercise, repeated millions of times. Stop a sentence before each token and ask the model how likely every token in its vocabulary is to come next.\n\nEach bar shows the probability the model gave to the [[ok:right token]]. The untrained model spreads its bets over all ${VOCAB} tokens, so every bar is nearly empty. Hover a token to see what the model guessed there.`,
			params: { at: 0, chart: 0, story: 0 },
			controls: [],
			legend: ['Right token', 'Given text'],
			readouts: ['choices']
		},
		{
			title: 'Loss: one number for how wrong',
			body: `The [[grad:loss]] turns those bars into one number. A right token with a full bar costs nothing. A right token with a nearly empty bar costs a lot. The loss is the average cost over all tokens.\n\nThe dashed line is the loss of guessing blindly, with every token equally likely: ${UNIFORM}. The untrained model starts just above it. Training has one goal, to push the loss down.`,
			params: { at: 0, story: 0 },
			controls: [],
			legend: ['Loss', 'Right token', 'Given text'],
			readouts: ['loss', 'choices'],
			deepDive: {
				note: `This loss is called cross-entropy. With all ${VOCAB} tokens equally likely it is ln ${LOG.vocab} = ${UNIFORM}.`,
				math: [
					{
						tex: 'L = -\\frac{1}{N}\\sum_{i=1}^{N} \\ln p(w_i \\mid w_1, \\dots, w_{i-1})',
						caption: 'the average cost of the right tokens w_i'
					},
					{ tex: '\\text{perplexity} = e^{L}', caption: 'a loss of L is as unsure as choosing evenly among e^L tokens' }
				]
			}
		},
		{
			title: 'Nudge every weight, then repeat',
			body: `One training step: read ${LOG.batch} snippets of ${LOG.ctx} tokens, measure the [[grad:loss]], and let backpropagation work out how each weight should change to lower it. Then nudge all ${WEIGHTS} weights a tiny bit that way.\n\nPress **Train** to run all ${STEPS} steps. The weights were saved at ${N} moments along the way, called checkpoints. What the model writes and the bars under the sentence change at each one. Every mark on the axis doubles the number of steps, because most of the change happens early.`,
			params: { at: 0 },
			controls: ['train', 'reset', 'at'],
			deepDive: {
				note: `The nudge is done by Adam, an optimizer that adapts the size of the step for each weight. The learning rate climbs to ${LM_TRAINING.learningRate[0]} over the first 200 steps, then falls smoothly to ${LM_TRAINING.learningRate[1]}.`,
				code: {
					lang: 'ts',
					source: `for (let step = 1; step <= ${LOG.steps}; step++) {\n  const batch = pickSnippets(text, ${LOG.batch}, ${LOG.ctx});\n  const { loss, gradients } = lossAndGradients(model, batch);\n  adam.update(model.weights, gradients, learningRate(step));\n}`
				}
			}
		},
		{
			title: 'What it learns first',
			body: 'Drag **Checkpoint** slowly from the start. After 25 steps the model has learned one thing: the full stop is the most common token, so it [[data:writes]] nothing else. By step 100 it uses a few more tokens, still only very common ones. By step 200 it writes whole sentences, but it gets stuck repeating one. From step 400 on it opens a story the way the finished model does.\n\nNow turn **Roll the dice** on, which shows more of what the model considers likely. At step 800 it names a character and uses a different name in the very next sentence. At step 3,200 the name stays the same.',
			params: { at: nearestCheckpoint(25) },
			controls: ['at', 'prompt', 'dice']
		},
		{
			title: 'Tested on stories it never saw',
			body: 'A model can also lower its loss by memorizing the training text. To check for that, some stories were set aside and never used for training.\n\nThe [[text:white dots]] show the loss on those unseen stories. They stay close to the [[grad:training loss]], so the model learned patterns that carry over to new text. When a model memorizes, the dots stop falling while the training loss keeps going down.',
			params: { at: LAST, valid: 1, story: 0 },
			controls: ['at'],
			legend: ['Loss', 'Right token', 'Given text'],
			deepDive: {
				note: 'The last 2% of the stories were set aside before training. The loss on unseen stories is measured on 400 fixed snippets of them, at every checkpoint.'
			}
		},
		{
			title: 'The same loop, far bigger',
			body: `On unseen stories this model ends at a loss of ${FINAL.validLoss.toFixed(2)}. That is like choosing among ${choices(FINAL.validLoss)} tokens at each position, down from ${VOCAB}. It got there in ${STEPS} steps, about 40 minutes on one computer.\n\nThe models you chat with are trained by the same loop: guess the next token, measure the loss, nudge the weights. They have billions of weights, read trillions of tokens and train for weeks on thousands of chips.`,
			params: { at: LAST, valid: 1 },
			controls: ['train', 'reset', 'at', 'prompt', 'dice'],
			deepDive: {
				note: `This model was trained on one file of ${LM_TRAINING.stories.toLocaleString('en-US')} short stories and read it about ${PASSES} times. Its tokens are whole words. Large models read most of their text only once, because they have so much of it.`
			}
		}
	],
	questions,
	createScene: createTrainingScene
};

export default lesson;

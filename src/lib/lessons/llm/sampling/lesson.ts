import type { LessonDef } from '../../types.ts';
import { LM_CONFIG, LM_TRAINING } from '../shared/lm-meta.ts';
import { createSamplingScene } from './scene.ts';

const WORDS = LM_CONFIG.vocab.toLocaleString('en-US');
const WEIGHTS = (Math.round(LM_TRAINING.parameters / 10000) * 10).toLocaleString('en-US');

const lesson: LessonDef = {
	id: 'sampling',
	track: 'llm',
	title: 'Next-token sampling',
	summary: 'See how a model picks its next token: always the top one, or by chance shaped by temperature, top-k and top-p.',
	defaults: { temperature: 1, topK: 0, topP: 1, prompt: 0, strip: 1, scenario: 1 },
	discrete: ['topK', 'strip', 'scenario'],
	controls: [
		{ kind: 'action', id: 'sample', label: 'Pick one token', icon: 'step' },
		{ kind: 'action', id: 'run', label: 'Generate', icon: 'play' },
		{ kind: 'action', id: 'reset', label: 'Start over', icon: 'reset' },
		{ kind: 'slider', key: 'temperature', label: 'Temperature', min: 0, max: 2, step: 0.05, tone: 'grad', digits: 2 },
		{ kind: 'slider', key: 'topK', label: 'Top-k (0 = off)', min: 0, max: 20, step: 1, tone: 'param', digits: 0 },
		{ kind: 'slider', key: 'topP', label: 'Top-p', min: 0.1, max: 1, step: 0.05, tone: 'param', digits: 2 },
		{
			kind: 'select',
			key: 'prompt',
			label: 'Start',
			options: [
				{ value: 0, label: 'once upon a time there was a' },
				{ value: 1, label: 'one day, lily went' },
				{ value: 2, label: 'the little girl wanted to' }
			]
		}
	],
	readouts: [
		{ key: 'kept', label: 'Tokens in play', digits: 0 },
		{ key: 'top', label: 'Most likely', tone: 'data' }
	],
	legend: [
		{ tone: 'data', label: 'Probability' },
		{ tone: 'neg', label: 'Cut off' },
		{ tone: 'ok', label: 'Picked' },
		{ tone: 'text', label: 'Text so far' }
	],
	steps: [
		{
			title: 'A score for every possible token',
			body: `After reading the [[text:text so far]], a language model gives every token in its vocabulary a [[data:probability]]. The bars show the 12 most likely tokens, and the last bar gathers all the rest.\n\nThese are the real scores of a small transformer trained on simple children's stories. It knows ${WORDS} words and has about ${WEIGHTS} thousand weights. The large models you chat with work the same way, with billions of weights and tokens that are pieces of words.`,
			params: { scenario: 1, strip: 0, temperature: 1 },
			controls: [],
			deepDive: {
				note: `This model is a simplified version of a chat model. Its tokens are whole words, not word pieces. Its vocabulary has only ${WORDS} entries, so rarer words are unknown to it and are never offered. It reads at most ${LM_CONFIG.ctx} tokens of context, so older text is ignored. It was trained for about 40 minutes on one computer, so its stories are simple and often odd.`
			}
		},
		{
			title: 'Always take the top token',
			body: 'The simplest rule is to always take the most likely token. This is called **greedy decoding**.\n\nPress **Generate** and watch the [[text:text]] grow. It reads smoothly, but every run gives exactly the same story, and it tends to circle back to the same phrases.',
			params: { scenario: 2, strip: 0, temperature: 0 },
			controls: ['run', 'sample', 'reset']
		},
		{
			title: 'Roll the dice instead',
			body: 'On the strip, each token gets a slice as wide as its [[data:probability]]. A needle lands at a random spot, and the slice it lands on is the [[ok:token picked]].\n\nPress **Pick one token** a few times, then **Generate**. Likely tokens win often, but not always, so every run is different.',
			params: { scenario: 3, strip: 1, temperature: 1 },
			controls: ['sample', 'run', 'reset']
		},
		{
			title: 'Temperature reshapes the odds',
			body: 'Temperature changes the odds before the dice are rolled. Low values sharpen the bars toward the top token: safe, but repetitive. High values flatten them: more surprising, and at the top of the range mostly nonsense.\n\nThe small ticks mark the original odds. Drag the slider, then press **Generate** at both ends.',
			params: { scenario: 4, temperature: 1 },
			controls: ['temperature', 'run', 'reset'],
			deepDive: {
				note: 'Temperature divides the raw scores (logits) before the softmax. As T gets close to 0 the top token takes everything, which is greedy decoding. At T = 1 you get the model\'s own probabilities.',
				math: [
					{
						tex: 'p_i = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}}',
						caption: 'z_i is the raw score of token i, T the temperature'
					}
				],
				code: { lang: 'ts', source: 'const probs = softmax(logits, temperature);' }
			}
		},
		{
			title: 'Top-k keeps only the k best',
			body: 'Top-k keeps only the k most likely tokens. Everything below rank k is [[neg:cut off]], and the rest is rescaled to add up to 100%.\n\nA high temperature makes the text bolder, and top-k stops it from wandering to very unlikely tokens. Try k = 1, then 5, then 0 to turn it off.',
			params: { scenario: 5, temperature: 1.3, topK: 5 },
			controls: ['topK', 'temperature', 'run', 'reset']
		},
		{
			title: 'Top-p keeps just enough to cover p',
			body: 'Top-p keeps the most likely tokens until their probabilities add up to p, and [[neg:cuts off]] the rest. When the model is sure, that is only a few tokens. When it is unsure, it is many.\n\nSwitch between the three starts and compare how many tokens stay in play. After "one day, lily went" the model is nearly sure the next word is "to", so only a handful stay. After the other two it has many ideas, and many stay.',
			params: { scenario: 6, temperature: 1.3, topP: 0.8 },
			controls: ['topP', 'prompt', 'run', 'reset'],
			deepDive: {
				note: 'Sort the tokens from most to least likely, then keep the shortest prefix that reaches probability p. The cut adapts to how sure the model is, unlike top-k, which always keeps the same number.',
				math: [
					{
						tex: 'S = \\text{the smallest set of top tokens with } \\sum_{i \\in S} p_i \\ge p',
						caption: 'the kept set; the probabilities in S are then rescaled to sum to 1'
					}
				],
				code: {
					lang: 'ts',
					source:
						'let sum = 0;\nfor (const i of sortedByProbability) {\n  keep[i] = true;\n  sum += probs[i];\n  if (sum >= p) break;\n}'
				}
			}
		},
		{
			title: 'Tune it yourself',
			body: 'Chat models usually combine a moderate temperature with top-p. There is no single right setting. It is a trade between predictable and varied.\n\nMove both sliders and press **Generate** to find a setting you like.',
			params: { scenario: 7, temperature: 0.8, topP: 0.9 },
			controls: ['temperature', 'topP', 'run', 'reset']
		}
	],
	createScene: createSamplingScene
};

export default lesson;

import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'The bars show the [[data:probability]] of each token. You lower the temperature from 1 to 0.5. What happens to the bars?\n\nDecide first, then check with the slider.',
		params: { scenario: 4, strip: 1, temperature: 1 },
		controls: ['temperature'],
		options: [
			{
				text: 'They all become equally tall, so every token is as likely as any other.',
				why: 'That is what a very high temperature does. Lowering it does the opposite.'
			},
			{
				text: 'The top bar grows and the small bars shrink. The order of the tokens stays the same.',
				correct: true,
				why: 'A low temperature sharpens the odds toward the top token. Here the top token goes from 39% to 94%. The ranking does not change, only how far apart the tokens are.'
			},
			{
				text: 'The bars keep their heights. Only the dice become less random.',
				why: 'Temperature acts before the dice are rolled. It reshapes the bars themselves, which you can see against the small ticks that mark the original odds.'
			},
			{
				text: 'The top bar shrinks, so less likely tokens get a fairer chance.',
				why: 'That is the effect of a higher temperature. Lowering it gives the top token more of the odds.'
			}
		]
	},
	{
		kind: 'choice',
		prompt:
			'Top-p is 0.8 and the temperature is 1.3. After "once upon a time there was a" many tokens stay in play. After "one day, lily went" only a handful do. Why?',
		params: { scenario: 6, temperature: 1.3, topP: 0.8, prompt: 0 },
		controls: ['topP', 'prompt'],
		options: [
			{
				text: 'Top-p keeps just enough tokens to add up to 0.8. When one token already holds most of the probability, few are needed.',
				correct: true,
				why: 'The cut adapts to how sure the model is. After "lily went" the model is nearly sure of one word, so a few tokens already reach 0.8. After the other starts the probability is spread over many tokens, so many are needed.'
			},
			{
				text: 'Top-p keeps every token whose own probability is above 0.8.',
				why: 'No single token comes close to 0.8 here, so that rule would keep almost nothing at every start. Top-p adds probabilities up from the top until they reach 0.8.'
			},
			{
				text: 'Top-p always keeps the same number of tokens. The difference comes from the longer start.',
				why: 'Keeping a fixed number of tokens is what top-k does. Top-p keeps as many as it takes to reach p, so the number changes from one start to the next.'
			},
			{
				text: 'The temperature is applied only to the start with many tokens.',
				why: 'The temperature is the same for all three starts. What differs is how the probability is spread over the tokens.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'The temperature is 1.3, so two runs rarely write the same text. Make the text come out **identical on every run** while keeping the temperature at 1 or higher.',
		params: { scenario: 5, temperature: 1.3, topK: 0, topP: 1 },
		controls: ['temperature', 'topK', 'run', 'reset'],
		goal: ({ params: p }) => shown(p.temperature) >= 1 && shown(p.topK, 0) === 1,
		hint: 'A run is the same every time when the model always takes its top token, as in greedy decoding. Which control can leave only one token in play?',
		solved:
			'With top-k set to 1, only the top token is kept and it gets all of the probability after rescaling. That is greedy decoding, whatever the temperature is. Press **Generate** twice to see that the text repeats.',
		example: { params: { topK: 1 } }
	}
];

import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';

const FULL = { showX2: 1, showBias: 1, showAct: 1 };

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'The input is 0.8 and the [[param:weight]] is 1.5. You change the weight to −1.5 and leave the input alone. What happens to the output?\n\nDecide first, then check with the slider.',
		params: { w1: 1.5 },
		controls: ['w1'],
		options: [
			{
				text: 'It drops to zero.',
				why: 'A weight of zero would do that. A weight of −1.5 is as strong as 1.5. It only points the other way.'
			},
			{
				text: 'It keeps its size and changes sign.',
				correct: true,
				why: 'The neuron multiplies the input by the weight: 0.8 × 1.5 is 1.2, and 0.8 × −1.5 is −1.2. A negative weight flips the sign and leaves the size alone.'
			},
			{
				text: 'Nothing, because the input did not change.',
				why: 'The input is the same, but the neuron multiplies it by the weight. Change either of the two and the product changes.'
			},
			{
				text: 'It gets weaker but stays positive.',
				why: 'A smaller positive weight, such as 0.5, would do that. A negative weight makes a positive input count as negative.'
			}
		]
	},
	{
		kind: 'choice',
		prompt:
			'Both inputs are now 0. Which setting can still change the sum [[data:z]]?\n\nTry the sliders if you are not sure.',
		params: { showX2: 1, showBias: 1, x1: 0, x2: 0, b: 0.5 },
		controls: ['w1', 'w2', 'b'],
		options: [
			{
				text: 'Only the bias.',
				correct: true,
				why: 'Each weight multiplies an input of 0, so both products are 0 whatever the weights are. The bias is added no matter what the inputs are, so here z is simply the bias.'
			},
			{
				text: 'Only the weights.',
				why: 'A weight multiplies its input. With an input of 0 the product is 0 for every weight, so the weights have nothing to act on.'
			},
			{
				text: 'The weights and the bias.',
				why: 'The bias does change z. The weights do not: each one multiplies an input of 0.'
			},
			{
				text: 'Nothing. With inputs of 0 the sum is always 0.',
				why: 'That is true for a neuron without a bias. The bias is added to the sum whatever the inputs are.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'This neuron uses ReLU. Its inputs and weights are fixed. Using only the [[param:bias]], make the [[data:output]] exactly 0.',
		params: { ...FULL, act: 2, x1: 0.8, x2: 0.6, w1: 1, w2: 1, b: 0 },
		controls: ['b'],
		goal: ({ params: p }) => shown(Math.max(0, p.w1 * p.x1 + p.w2 * p.x2 + p.b)) === 0,
		hint: 'ReLU outputs 0 whenever the sum z is zero or negative. The two weighted inputs add up to 1.4.',
		solved:
			'The weighted inputs add up to 1.4, so a bias of −1.4 or lower brings z down to zero or below, and ReLU outputs 0 for every such sum. A negative bias is how a neuron is made hard to switch on.',
		example: { params: { b: -1.5 } }
	}
];

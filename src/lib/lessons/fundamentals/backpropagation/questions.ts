import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'The network guesses ŷ = 0.40 and the target is 1, so the output blame [[grad:δ]] is −0.14. Now you switch the target to 0, which is closer to the guess. What happens to δ?\n\nDecide first, then check with the toggle.',
		params: { phase: 2, target: 1 },
		controls: ['target'],
		options: [
			{
				text: 'It stays at −0.14, because the guess ŷ has not changed.',
				why: 'The target is part of the error ŷ − y, so changing it changes δ even when ŷ stays put.'
			},
			{
				text: 'It turns positive and grows, because the new target makes the guess worse.',
				why: 'The guess 0.40 is closer to 0 than to 1. The gap shrinks from 0.60 to 0.40, so δ shrinks as well.'
			},
			{
				text: 'It turns positive and shrinks.',
				correct: true,
				why: 'δ is the error ŷ − y times how sensitive the output neuron is. The guess 0.40 is now above the target 0, so the error is positive and the push is downward. The gap fell from 0.60 to 0.40 and the sensitivity is the same, so δ goes from −0.14 to about 0.10.'
			},
			{
				text: 'It stays negative and shrinks, because the gap is smaller.',
				why: 'The sign of δ says which way to push the output. The guess is now above the target, so the push flips to downward and δ turns positive. Only its size follows the gap.'
			}
		]
	},
	{
		kind: 'choice',
		prompt:
			'Set the input x₁ to 0. Which weights now have a [[grad:gradient]] of exactly 0?\n\nDecide first, then check with the slider.',
		params: { phase: 4 },
		controls: ['x1', 'x2'],
		options: [
			{
				text: 'None, because every neuron still carries some blame.',
				why: 'The blame at a neuron is only one factor. A weight\'s gradient is that blame times the signal arriving along the weight, and a signal of 0 makes the product 0.'
			},
			{
				text: 'The two weights from x₂ to the hidden neurons.',
				why: 'They carry x₂, which is −0.4 here, not 0. Their gradients stay nonzero. The weights that start at x₁ are the ones with a signal of 0.'
			},
			{
				text: 'The two weights from the hidden neurons to the output ŷ.',
				why: 'They carry the hidden neurons\' outputs, which are about 0.29 and 0.32 with x₁ = 0, so their gradients stay nonzero.'
			},
			{
				text: 'The two weights from x₁ to the hidden neurons.',
				correct: true,
				why: 'A weight\'s gradient is the blame at its destination times the signal at its source. The source here is x₁ = 0, so both products are 0 however large the blame is. The weights from x₂ keep their gradients.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'The target is 1. Using the controls, make the network answer with a [[data:prediction]] ŷ below 0.10 for these same inputs.',
		params: { phase: 5, target: 1 },
		controls: ['update', 'run', 'reset', 'target', 'lr'],
		goal: ({ readouts }) => shown(readouts.pred, 3) < 0.1,
		hint: 'The network only moves its weights toward the target it is given. Think about what the target control changes, and what an update does with it.',
		solved:
			'The target is part of the loss. With the target at 0 the blame δ is positive, so every update lowers the output instead of raising it, and training repeats that until ŷ is near 0. Same network, same inputs, different target: a different answer.'
	}
];

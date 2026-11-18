import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';
import { loss, type ModelId } from './model.ts';

const PLATEAU = { resid: 1, surface: 1, model: 1, w: -1.5, b: 1.2 };

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'The sliders sit at the best line and the [[grad:loss]] is 0.012. Keep [[param:w]] and move [[param:b]] up by 0.3. Then put it back and move it down by 0.3 instead. How do the two losses compare?\n\nDecide first, then check with the sliders.',
		params: { resid: 1, surface: 1, model: 0, w: 0.82, b: -0.26 },
		controls: ['w', 'b'],
		options: [
			{
				text: 'One direction raises the loss and the other lowers it.',
				why: 'The best line is the bottom of the bowl. Every move away from it goes uphill, so no direction can lower the loss.'
			},
			{
				text: 'Both raise it, by about the same amount.',
				correct: true,
				why: 'The bottom of a bowl has walls on both sides. The loss goes from 0.012 to 0.097 for b up by 0.3 and to 0.106 for b down by 0.3. The two are close because this bowl is almost symmetric in b.'
			},
			{
				text: 'Neither changes it much, because w is still the best weight.',
				why: 'The best w and the best b belong together. Moving b lifts or drops the whole line by 0.3, away from the dots, and that adds about 0.3 × 0.3 = 0.09 to the loss.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'Now the model is a tanh neuron and the ball starts high up on the surface. Using [[param:w]] and [[param:b]], find a setting where the [[grad:loss]] shows 0.025 or less.',
		params: PLATEAU,
		controls: ['w', 'b'],
		goal: ({ params: p }) => shown(loss(Math.round(p.model) as ModelId, p.w, p.b), 3) <= 0.025,
		hint: 'Drag the surface to look at it from the side and find where the valley lies. Then move the ball toward it, and let the Loss readout tell you whether each slider moves you down or up.',
		solved:
			'The lowest point of this surface sits near w = 1.6 and b = −0.5, where the loss is about 0.006. This surface is not a bowl. It has a wide flat part and a narrow valley, so from the start the way down is hard to see without looking at the whole surface.',
		example: { params: { w: 1.6, b: -0.5 } }
	},
	{
		kind: 'choice',
		prompt:
			'The ball sits high on the flat part of the tanh surface, where the [[grad:loss]] is 2.449. Moving w from −1.5 to −1.4 only lowers the loss to 2.392. What does this tell a method that wants to walk downhill?\n\nTry the sliders if you are not sure.',
		params: PLATEAU,
		controls: ['w', 'b'],
		options: [
			{
				text: 'This is a low point, because the loss has almost stopped changing.',
				why: 'A flat place is not the same as a low place. The loss here is 2.449, and the lowest loss on this surface is about 0.006, far away from the ball.'
			},
			{
				text: 'The surface is nearly flat here, so the slope gives only a weak hint about which way is down.',
				correct: true,
				why: 'On a plateau a small move changes the loss only a little, so the slope under the ball is small. A method that follows the slope still gets the direction, but it moves slowly.'
			},
			{
				text: 'The slider has no effect on the model at this setting.',
				why: 'The line does change when you move w, and so does the loss, just by little: 2.449 becomes 2.392. A weak effect is still an effect.'
			},
			{
				text: 'The whole surface is flat, so no other setting is any better.',
				why: 'The plateau is wide but it ends. Elsewhere on the same surface the loss falls below 0.01, in the valley near w = 1.6 and b = −0.5.'
			}
		]
	}
];

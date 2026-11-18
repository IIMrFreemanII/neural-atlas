import { derivative, type ActivationId } from '../../../ml/activations.ts';
import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'This unit uses Step, and [[data:z]] is 1.0. A training method nudges z up to 1.1 to see whether the output gets better. What does it learn from that nudge?\n\nDecide first, then check with the slider.',
		params: { act: 4, showDeriv: 1, z: 1 },
		controls: ['z'],
		options: [
			{
				text: 'That a higher z is better, because the output is 1.',
				why: 'The output was 1 before the nudge and is 1 after it. Nothing changed, so nothing says that a higher z is better.'
			},
			{
				text: 'That the output rises a little, as it does for sigmoid.',
				why: 'Step has no gradual rise. Its output jumps from 0 to 1 at z = 0 and is flat everywhere else. A smooth curve such as sigmoid is the one that rises a little.'
			},
			{
				text: 'Nothing: the output does not move, and the [[grad:slope]] is 0.',
				correct: true,
				why: 'Away from z = 0 the output is flat, so a small nudge changes nothing and the slope is 0. Learning gets no hint about which way to move.'
			},
			{
				text: 'Nothing yet, but a bigger nudge in the same direction will show the way.',
				why: 'A bigger nudge in the same direction only changes the output if it crosses z = 0, and then the output jumps by a whole step. The slope still says 0 on both sides.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'Sigmoid is chosen. Move [[data:z]] to a place where the [[grad:slope]] readout shows 0.04 or less.',
		params: { act: 1, showDeriv: 1, z: 0 },
		controls: ['z'],
		goal: ({ params: p }) => shown(derivative(Math.round(p.act) as ActivationId, p.z)) <= 0.04,
		hint: 'The slope is the steepness of the curve. Look for the parts of the sigmoid where the curve has gone flat.',
		solved:
			'The slope of sigmoid is 0.25 at z = 0 and fades quickly: beyond about 3 on either side it is under 0.05. A unit that sits out there passes almost no learning signal back, which is the vanishing gradient problem.',
		example: { params: { z: 4 } }
	},
	{
		kind: 'choice',
		prompt:
			'The network is fitted with 1 hidden unit and a Linear activation. The [[neg:fit error]] reads 0.565. You raise [[param:units]] to 6. What happens to the error?\n\nDecide first, then check with the sliders.',
		params: { mode: 1, act: 0, units: 1 },
		controls: ['units', 'act'],
		options: [
			{
				text: 'It drops, because every extra unit adds a bend.',
				why: 'A Linear unit has no bend. Each one is still a straight line, so extra units add nothing the first one could not already do.'
			},
			{
				text: 'It rises, because there are more pieces to get wrong.',
				why: 'The fit takes the best weights it can find. A unit that does not help can always be given a weight that cancels it, so more units never make the best fit worse.'
			},
			{
				text: 'It drops to nearly zero, because six units can match any curve.',
				why: 'Six units can match a very bent curve only if they bend. With ReLU the same six units reach 0.127, and Linear is far from that.'
			},
			{
				text: 'It stays at 0.565.',
				correct: true,
				why: 'A weighted sum of straight lines is one straight line, so six Linear units fit no better than one. Switch to ReLU and the same six units bring the error down to 0.127.'
			}
		]
	}
];

import { forward } from '../../../ml/mlp.ts';
import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';
import { buildNet } from './network.ts';

const net = buildNet();

/** The network's output ŷ for the two inputs, computed with the lesson's own network. */
const output = (x1: number, x2: number): number => {
	const t = forward(net, [x1, x2]);
	return t.a[t.a.length - 1][0];
};

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'Show the weight grid of the output layer, the single last neuron. It receives the three activations of the second hidden layer. How many rows and columns does its grid have?\n\nDecide first, then check with the layer selector.',
		params: { matrix: 1, layer: 1 },
		controls: ['layer'],
		options: [
			{
				text: '3 rows and 1 column.',
				why: 'This turns the idea around. A grid has one row per neuron of the layer and one column per input of it, so three inputs make three columns.'
			},
			{
				text: '1 row and 3 columns.',
				correct: true,
				why: 'There is one row for the one output neuron and one column for each of its three inputs. The grid holds one weight per connection, so it has 3 weights.'
			},
			{
				text: '3 rows and 3 columns.',
				why: 'That is the grid of the second hidden layer, which has three neurons that each take three inputs. The output layer has only one neuron, so it has one row.'
			},
			{
				text: '1 row and 1 column.',
				why: 'A grid holds one weight for every connection into the layer. The output neuron has three connections, so one cell is too few.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'The prediction [[data:ŷ]] is now about 0.45. Using only the two inputs, make it 0.20 or lower.',
		params: { matrix: 0, layer: 1 },
		controls: ['x1', 'x2'],
		goal: ({ params: p }) => shown(output(p.x1, p.x2)) <= 0.2,
		hint: 'The weights are fixed, so there is no rule to read off. Move one input at a time and watch which direction lowers ŷ for each of them.',
		solved:
			'Each input reaches the output along many paths through the [[param:weights]], and the paths can add up or cancel. With x₁ at 0.8, raising x₂ from −0.5 to 0.5 takes ŷ from 0.45 to 0.17.',
		example: { params: { x1: 0.8, x2: 0.5 } }
	},
	{
		kind: 'choice',
		prompt:
			'Only [[data:x₂]] changes, from −0.5 to 0.5, and x₁ stays at 0.8. Which neurons end up with a different value?\n\nDecide first, then check with the slider.',
		params: { x1: 0.8, x2: -0.5, matrix: 0 },
		controls: ['x2'],
		options: [
			{
				text: 'Every neuron after the inputs: both hidden layers and the output.',
				correct: true,
				why: 'Each neuron of the first hidden layer is connected to x₂, so all three change. Each later layer takes the layer before it as input, so the change passes on. In the first hidden layer the values go from 0.57, 0.22, 0.19 to 0.92, −0.47, 0.75, and ŷ goes from 0.45 to 0.17.'
			},
			{
				text: 'Only the first hidden layer, because only it is connected to the inputs.',
				why: 'The second hidden layer is fed by the first one. When the values it receives change, its own sums change too, and so does everything after it.'
			},
			{
				text: 'Only the neurons with a large weight from x₂.',
				why: 'Every neuron of the first hidden layer has a connection from x₂. A small weight only makes the effect smaller. And the later layers have no weight from x₂ at all, yet they still change.'
			}
		]
	}
];

import type { Question } from '../../types.ts';

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'Suppose the attention layer wrote an [[ok:update]] of all zeros for the lane of "play". What would that lane hold after the layer?',
		params: { focus: 3, zoom: 0, scenario: 3 },
		controls: ['play'],
		options: [
			{
				text: 'All zeros, because the layer replaced the vector.',
				why: 'A layer does not replace the vector. Its output is added to what is already there, so the old numbers stay.'
			},
			{
				text: 'The same vector as before the layer.',
				correct: true,
				why: 'The lane holds before + update. With an update of zero, the sum is the old vector. This is why old information survives unless an update writes over it.'
			},
			{
				text: 'The vector of the token before it, "to".',
				why: 'Attention can gather information from earlier tokens, but what it gathers is added to the lane, not swapped in. With an update of zero nothing is added at all.'
			},
			{
				text: 'The vector rescaled to mean 0 and spread 1.',
				why: 'That is the normalizing step. It feeds the layer a rescaled copy, while the lane itself keeps its own vector, and the update is added to it.'
			}
		]
	},
	{
		kind: 'choice',
		prompt:
			'In the attention step, which tokens can the lane of "play" gather information from?\n\nDecide first, then click the lanes to check.',
		params: { focus: 2, zoom: 0, scenario: 2 },
		controls: ['play'],
		options: [
			{
				text: 'Itself and the tokens before it: "she", "loved", "to".',
				correct: true,
				why: 'Each token gathers from itself and from the tokens before it. The first token can only look at itself.'
			},
			{
				text: 'All five tokens, including "with", because the whole sentence is on screen.',
				why: 'A token gathers only from itself and the tokens before it. "with" comes after "play", so the lane of "play" cannot gather from it.'
			},
			{
				text: 'Only the tokens before it, not itself.',
				why: 'A token can also gather from itself. The first token has nothing before it, and it still gathers from itself.'
			},
			{
				text: 'Only the token right before it, "to".',
				why: 'A token can gather from itself and from every token before it, not just the nearest one. The arcs show how much it takes from each.'
			}
		]
	},
	{
		kind: 'choice',
		prompt:
			'With all 4 blocks the model\'s first guess for the next token is "her". Read after only 1 block, its first guess is "his". Why is the early guess worse?\n\nDrag **Blocks** to compare the two.',
		params: { focus: 7, zoom: 1, depth: 4, scenario: 7 },
		controls: ['depth'],
		options: [
			{
				text: 'The first block has weaker weights than the later ones.',
				why: 'Each block has its own weights, but the first block is not what holds the guess back. The stream is not finished after one block: three more blocks still have changes to add.'
			},
			{
				text: 'With 1 block the last lane can look at fewer of the earlier tokens.',
				why: 'The number of blocks does not change what a lane can look at. In every block a token gathers from itself and from all the tokens before it.'
			},
			{
				text: 'Each block refines the stream, and the model was trained to be read only after the last one.',
				correct: true,
				why: 'Every block reads the stream left by the one below and adds its own changes. After one block, three of the four have not written yet, and the model was never trained to give its answer that early.'
			},
			{
				text: 'With 1 block the model picks from a shorter list of words.',
				why: 'The top vector is compared against all 2,048 words however many blocks ran. Only the vector that is compared changes.'
			}
		]
	}
];

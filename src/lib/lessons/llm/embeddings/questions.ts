import { nearest } from '../../../ml/embedding.ts';
import type { Question } from '../../types.ts';
import { CATEGORIES, VECTORS, WORDS, indexOf } from './data.ts';

const ANIMALS = CATEGORIES.findIndex((c) => c.name === 'Animals');

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'Cat and dog are both animals, and nobody told the layout that. Why are their vectors still close together?',
		params: { colors: 1, word: indexOf('cat'), neighbors: 0 },
		controls: ['spin'],
		options: [
			{
				text: 'The numbers come from which words appear near each other, and cat and dog keep similar company.',
				correct: true,
				why: 'GloVe counted which words appear near which in Wikipedia and news text. Words that turn up in similar sentences end up with similar numbers, and that is what puts them close together.'
			},
			{
				text: 'Someone sorted the words into kinds first, then gave each kind similar numbers.',
				why: 'The kinds are our own colors. The vectors never saw them, and a few words sit between kinds.'
			},
			{
				text: 'Their token IDs are close, so the table stores them next to each other.',
				why: 'An ID is only a row number and means nothing by itself. Moving a word to another row would change nothing about its vector.'
			},
			{
				text: 'Both words are three letters long, so their vectors look alike.',
				why: 'The numbers come from the words around a word, not from its letters. A word spelled quite differently can sit just as close.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'Colors are ours, and the vectors never saw them. Show it: click an **animal**, then use **Neighbors** until a line reaches a word that is **not an animal**.',
		params: { colors: 1, word: indexOf('cat'), neighbors: 0 },
		controls: ['neighbors', 'spin'],
		goal: ({ params: p, readouts }) => {
			const i = WORDS.findIndex((w) => w.word === readouts.word);
			const k = Math.round(p.neighbors);
			if (i < 0 || k < 1 || WORDS[i].category !== ANIMALS) return false;
			return nearest(VECTORS, VECTORS[i], k, [i]).some((n) => WORDS[n.index].category !== ANIMALS);
		},
		hint: 'Lines go to the closest words by their 100 numbers, whatever kind they are. Pick an animal and show more and more neighbors.',
		solved:
			'The neighbors are the words whose vectors are closest, counted by the angle between them. A few words are used in ways that sit between two kinds, so a person or a size can land among the animals.'
	},
	{
		kind: 'choice',
		prompt:
			'Neighbors is on for cat. Sometimes a line reaches past a dot that looks closer on screen and ends at one that looks farther. What explains it?',
		params: { colors: 1, word: indexOf('cat'), neighbors: 5 },
		controls: ['neighbors', 'spin'],
		options: [
			{
				text: 'Lines go first to the words of the same color, and the closer dot has another color.',
				why: 'The colors are ours. The vectors never saw them, and the lines follow the numbers only.'
			},
			{
				text: 'Lines go to the words that appear most often in the text the vectors came from.',
				why: 'How often a word appears is not what decides it. Neighbors are the words whose vectors point most nearly the same way as the selected word.'
			},
			{
				text: 'Neighbors are found with all 100 numbers, and the picture keeps only a third of the spread.',
				correct: true,
				why: 'The picture shows 3 directions of the 100, and they keep only about a third (33%) of the spread. Two dots can look close here and still differ in the other 97 directions, where the neighbors are found.'
			},
			{
				text: 'The picture is drawn from different vectors than the ones used to find neighbors.',
				why: 'Both use the same 74 vectors. The picture is a flattened view of them, and flattening is what loses information.'
			}
		]
	}
];

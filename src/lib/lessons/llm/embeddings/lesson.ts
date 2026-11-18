import type { LessonDef } from '../../types.ts';
import { indexOf } from './data.ts';
import { questions } from './questions.ts';
import { createEmbeddingsScene } from './scene.ts';

const CAT = indexOf('cat');

const lesson: LessonDef = {
	id: 'embeddings',
	track: 'llm',
	title: 'Embedding space',
	summary: 'Every token becomes a list of numbers. Similar words land close together, and directions carry meaning.',
	defaults: {
		colors: 1,
		neighbors: 0,
		relation: 0,
		analogy: 0,
		showAnalogy: 0,
		spin: 1,
		word: CAT,
		scenario: 1
	},
	discrete: ['neighbors', 'word', 'showAnalogy', 'scenario'],
	controls: [
		{ kind: 'slider', key: 'neighbors', label: 'Neighbors', min: 0, max: 8, step: 1, tone: 'data', digits: 0 },
		{
			kind: 'select',
			key: 'relation',
			label: 'Direction',
			options: [
				{ value: 0, label: 'None' },
				{ value: 1, label: 'To woman' },
				{ value: 2, label: 'To capital' },
				{ value: 3, label: 'To past' },
				{ value: 4, label: 'To young' },
				{ value: 5, label: 'To bigger' }
			]
		},
		{
			kind: 'select',
			key: 'analogy',
			label: 'Analogy',
			options: [
				{ value: 0, label: 'royalty' },
				{ value: 1, label: 'capitals' },
				{ value: 2, label: 'verbs' },
				{ value: 3, label: 'animals' },
				{ value: 4, label: 'a miss' }
			]
		},
		{ kind: 'action', id: 'play', label: 'Play', icon: 'play' },
		{ kind: 'toggle', key: 'spin', label: 'Rotate' }
	],
	readouts: [
		{ key: 'word', label: 'Selected' },
		{ key: 'best', label: 'Closest word', tone: 'ok' }
	],
	legend: [
		{ tone: 'param', label: 'People' },
		{ tone: 'ok', label: 'Animals' },
		{ tone: 'grad', label: 'Food' },
		{ tone: 'data', label: 'Places' },
		{ tone: 'neg', label: 'Actions' },
		{ tone: 'text', label: 'Sizes' }
	],
	steps: [
		{
			title: 'A token ID means nothing by itself',
			body: 'A token ID is just a row number. The model looks that row up in a table and gets a vector: a list of numbers, 100 here, shown in the panel as a grid of cells.\n\nEach dot is one word, placed by its vector. These are real GloVe vectors, learned from Wikipedia and news text by counting which words appear near which. Only 74 words are shown. Hover and click dots.',
			params: { colors: 0, word: CAT },
			controls: ['spin']
		},
		{
			title: 'Similar words land close together',
			body: 'Now color shows the kind of word: [[param:people]], [[ok:animals]], [[grad:food]], [[data:places]], [[neg:actions]] and [[text:sizes]].\n\nNobody told the layout what the kinds are. Words that mean similar things simply have similar numbers, so they sit near each other. The colors are ours: the vectors never saw them, and a few words sit between kinds.',
			params: { colors: 1 },
			controls: ['spin']
		},
		{
			title: 'Nearest neighbors',
			body: 'Lines reach from the selected word to its closest [[data:neighbors]]. Slide **Neighbors** to show more of them, and click any dot to ask the same question about another word. Only the 74 words shown are searched.\n\nCloseness is measured by the angle between two vectors, not by distance on screen.',
			params: { word: CAT, neighbors: 5 },
			controls: ['neighbors', 'spin'],
			deepDive: {
				note: 'Closeness is the cosine of the angle between two vectors: 1 means they point the same way, 0 means they are unrelated, negative means opposite.',
				math: [{ tex: '\\cos\\theta = \\frac{u \\cdot v}{\\lVert u \\rVert \\, \\lVert v \\rVert} = \\frac{\\sum_i u_i v_i}{\\sqrt{\\sum_i u_i^2}\\,\\sqrt{\\sum_i v_i^2}}' }],
				code: {
					lang: 'ts',
					source:
						'const cosine = (a, b) => dot(a, b) / (norm(a) * norm(b));\nconst scored = vectors.map((v, i) => ({ i, s: cosine(v, query) }));\nscored.sort((p, q) => q.s - p.s);'
				}
			}
		},
		{
			title: 'Directions carry meaning',
			body: 'Pick a **Direction**. The arrows from a male word to its female partner mostly point the same way, so "more female" is roughly a direction in this space. They are not identical: in all 100 numbers they mostly agree, but this 3D picture flattens some of them, so a few look short or tilted.\n\nTry the other directions: country to capital, present to past, adult to young, big to bigger.',
			params: { word: indexOf('man'), relation: 1 },
			controls: ['relation', 'spin']
		},
		{
			title: 'Arithmetic on meaning',
			body: 'Take [[param:king]], subtract man, add woman, and you land close to [[ok:queen]], the nearest of the 74 words. It is a close call, not an exact hit. The [[param:arrows]] show the same step being copied from one pair of words to another.\n\nPress **Play**, then pick another **Analogy**. The last one, man is to boy as king is to ?, misses: it lands on queen, not prince. That is normal for real vectors. They are noisy, and some relations are weaker than others.',
			params: { word: indexOf('queen'), analogy: 0, showAnalogy: 1 },
			controls: ['play', 'analogy'],
			deepDive: {
				note: 'To finish "a is to b as c is to ?", build one vector and look for the word nearest to it by cosine. The three input words are skipped in the search, otherwise the answer would usually be one of them.',
				math: [{ tex: 'v = b - a + c, \\qquad \\text{answer} = \\operatorname*{arg\\,max}_{w \\notin \\{a,b,c\\}} \\cos(v, w)' }],
				code: {
					lang: 'ts',
					source: 'const v = analogy(vec[man], vec[woman], vec[king]); // woman - man + king\nconst [best] = nearest(vec, v, 1, [man, woman, king]);'
				}
			}
		},
		{
			title: 'A shadow of a bigger space',
			body: 'The picture shows 3 directions, chosen to keep as much of the spread as possible. They keep only about a third of it. Each word really has 100 numbers, and bigger models use more.\n\nNeighbors are found with all 100, which is why a line sometimes reaches past a dot that looks closer.',
			params: { word: CAT, neighbors: 5 },
			controls: ['neighbors', 'spin'],
			deepDive: {
				note: 'The 3D picture is a principal component analysis. It centers all the vectors, finds the 3 directions along which they vary the most (the top eigenvectors of their covariance matrix), and reports each word\'s coordinates along them. The picture keeps about a third (33%) of the total variance in this table; the rest is flattened away.'
			}
		},
		{
			title: 'One vector per word is not enough',
			body: 'A word keeps the same vector in every sentence, whatever it means there. "Mouse" sits in one place whether it is an animal or something you click.\n\nThe next lesson, self-attention, lets each token adjust its vector using the words around it.',
			params: { word: indexOf('mouse') },
			controls: ['neighbors', 'relation', 'spin']
		}
	],
	questions,
	createScene: createEmbeddingsScene
};

export default lesson;

import type { LessonDef } from '../../types.ts';
import { questions } from './questions.ts';
import { createKMeansScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'k-means',
	track: 'classic-ml',
	title: 'K-means clustering',
	summary: 'Group unlabeled points by alternating assignment and averaging.',
	defaults: { k: 3, dataset: 0, init: 0, prep: 2, live: 0, spokes: 0, chart: 0, autorun: 0, seed: 4, scenario: 1 },
	discrete: ['k', 'prep', 'live', 'spokes', 'chart', 'autorun', 'seed', 'scenario'],
	controls: [
		{ kind: 'action', id: 'run', label: 'Run', icon: 'play' },
		{ kind: 'action', id: 'assign', label: 'Assign', icon: 'step' },
		{ kind: 'action', id: 'move', label: 'Move centers', icon: 'step' },
		{ kind: 'action', id: 'reseed', label: 'New start', icon: 'shuffle' },
		{ kind: 'slider', key: 'k', label: 'Clusters (k)', min: 1, max: 6, step: 1, tone: 'param', digits: 0 },
		{
			kind: 'select',
			key: 'dataset',
			label: 'Data',
			options: [
				{ value: 0, label: 'Three blobs' },
				{ value: 1, label: 'Five blobs' },
				{ value: 2, label: 'Moons' },
				{ value: 3, label: 'Noise' }
			]
		},
		{
			kind: 'select',
			key: 'init',
			label: 'Start',
			options: [
				{ value: 0, label: 'Random' },
				{ value: 1, label: 'k-means++' }
			]
		}
	],
	readouts: [
		{ key: 'iter', label: 'Rounds', digits: 0 },
		{ key: 'inertia', label: 'Spread', tone: 'grad', digits: 1 }
	],
	legend: [
		{ tone: 'muted', label: 'Points' },
		{ tone: 'text', label: 'Centers' },
		{ tone: 'grad', label: 'Spread' }
	],
	steps: [
		{
			title: 'Points with no labels',
			body: 'These twelve thousand [[muted:points]] come with no labels. Nobody has said which point belongs where, yet you can see clumps.\n\n**Clustering** means finding such groups automatically. K-means is the simplest way to do it.',
			params: { scenario: 1, prep: 0 },
			controls: []
		},
		{
			title: 'Guess k centers',
			body: 'K-means starts by dropping **k** [[text:centers]] onto random points. The guess is usually poor, and that is fine.\n\nPress **New start** a few times to see different guesses.',
			params: { scenario: 2, prep: 1 },
			controls: ['reseed']
		},
		{
			title: 'Assign every point to its nearest center',
			body: 'Each point takes the color of the closest [[text:center]]. Here colors only name clusters; they carry no other meaning.\n\nDrag a center and watch thousands of points switch sides at once.',
			params: { scenario: 3, prep: 2, live: 1 },
			controls: ['assign']
		},
		{
			title: 'Move each center to the middle of its points',
			body: 'Press **Move centers**. Each [[text:center]] jumps to the average position of its own points, and thin [[grad:lines]] show which points pull on it.\n\nAfter the move, some points are closer to a different center, so press **Assign** again.',
			params: { scenario: 4, prep: 2, spokes: 1 },
			controls: ['move', 'assign']
		},
		{
			title: 'Repeat until nothing changes',
			body: 'Press **Run** and the two steps alternate by themselves. The chart tracks the [[grad:spread]], the total squared distance from every point to its center.\n\nIt falls every round, then flattens. When no center moves any more, the run is done.',
			params: { scenario: 5, prep: 2 },
			controls: ['run', 'assign', 'move', 'reseed'],
			deepDive: {
				note: 'K-means lowers this number (called inertia) with every assign and every move, so it must stop somewhere. It stops at a local minimum, not always the best one.',
				math: [
					{
						tex: 'J = \\sum_{i=1}^{n} \\lVert x_i - \\mu_{c(i)} \\rVert^2, \\qquad \\mu_j = \\frac{1}{|S_j|} \\sum_{i \\in S_j} x_i',
						caption: 'c(i) is the nearest center of point i, and S_j is the set of points assigned to center j.'
					}
				],
				code: {
					lang: 'ts',
					source: `let centers = initCenters(points, k);
for (let round = 0; round < maxRounds; round++) {
  const labels = assign(points, centers);      // nearest center
  const next = means(points, labels, centers); // average of each group
  if (maxShift(centers, next) < 1e-4) break;
  centers = next;
}`
				}
			}
		},
		{
			title: 'A bad start can get stuck',
			body: 'Five blobs, five centers. A random start often puts two centers in one clump and leaves another clump unclaimed. They never escape.\n\nPress **Run**, then switch **Start** to k-means++ and run again: it spreads the first guesses apart.',
			params: { scenario: 6, dataset: 1, k: 5, init: 0, seed: 1 },
			controls: ['run', 'reseed', 'init'],
			deepDive: {
				note: 'K-means++ picks the first center at random. Every next center is a point chosen with probability proportional to its squared distance from the nearest center already picked, so far-away points are likely and nearby points are not.',
				math: [
					{
						tex: 'P(x_i \\text{ is next}) = \\frac{D(x_i)^2}{\\sum_j D(x_j)^2}, \\qquad D(x) = \\min_{c \\in \\text{picked}} \\lVert x - c \\rVert'
					}
				]
			}
		},
		{
			title: 'How many clusters?',
			body: 'K-means needs **k** up front. More centers always lower the [[grad:spread]], but the curve bends where extra centers stop helping. That bend is the elbow.\n\nSlide **k** and compare: the data has five blobs.',
			params: { scenario: 7, dataset: 1, k: 3, init: 1, chart: 1, autorun: 1 },
			controls: ['k', 'run']
		},
		{
			title: 'Your turn',
			body: 'On the moons, k-means cuts straight across both bands, because it can only make round-ish regions.\n\nTry the noise data too: it finds clusters that are not really there. Drag centers, change **k** and **Start**, and press **Run**.',
			params: { scenario: 8, dataset: 2, k: 2, init: 1, live: 1, autorun: 1 }
		}
	],
	questions,
	createScene: createKMeansScene
};

export default lesson;

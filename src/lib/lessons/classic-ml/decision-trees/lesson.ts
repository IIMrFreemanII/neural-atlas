import type { LessonDef } from '../../types.ts';
import { BOUNDS, MAX_DEPTH, START_CUT } from './data.ts';
import { questions } from './questions.ts';
import { createDecisionTreesScene } from './scene.ts';

// Pages before the test points exist show no test chip and no test accuracy.
const TRAIN_ONLY = { legend: ['Class A', 'Class B'], readouts: ['acc'] };

const lesson: LessonDef = {
	id: 'decision-trees',
	track: 'classic-ml',
	title: 'Decision trees and forests',
	summary: 'Grow a tree of yes-or-no questions, watch it overfit, then let a forest of trees vote.',
	defaults: { depth: 2, split: 0, axis: 0, cut: 0, probe: 0, test: 0, chart: 0, sample: 0, ghosts: 0, forest: 0, trees: 25, scenario: 1 },
	discrete: ['depth', 'split', 'probe', 'test', 'chart', 'sample', 'ghosts', 'forest', 'trees', 'scenario'],
	controls: [
		{ kind: 'slider', key: 'depth', label: 'Max depth', min: 0, max: MAX_DEPTH, step: 1, tone: 'param', digits: 0 },
		{
			kind: 'select',
			key: 'axis',
			label: 'Cut direction',
			options: [
				{ value: 0, label: 'Vertical line (x₁)' },
				{ value: 1, label: 'Horizontal line (x₂)' }
			]
		},
		{ kind: 'slider', key: 'cut', label: 'Cut at', min: BOUNDS[0], max: BOUNDS[2], step: 0.01, tone: 'grad', digits: 2 },
		{ kind: 'slider', key: 'trees', label: 'Trees', min: 1, max: 50, step: 1, tone: 'param', digits: 0 },
		{ kind: 'action', id: 'best', label: 'Find best', icon: 'step' },
		{ kind: 'action', id: 'grow', label: 'Grow', icon: 'play' },
		{ kind: 'action', id: 'resample', label: 'New sample', icon: 'shuffle' }
	],
	readouts: [
		{ key: 'acc', label: 'Training accuracy', tone: 'data' },
		{ key: 'test', label: 'Test accuracy', tone: 'ok' }
	],
	legend: [
		{ tone: 'data', label: 'Class A' },
		{ tone: 'neg', label: 'Class B' },
		{ tone: 'ok', label: 'Test points' }
	],
	steps: [
		{
			title: 'A flowchart of yes-or-no questions',
			body: 'A **decision tree** asks a question about one coordinate, such as "is x₂ below 0.14?". Inside each answer it asks another question, and so on. The [[text:questions]] are the boxes in the diagram, and each line on the plane is one of them.\n\nWhen the questions run out, the dot is in a **leaf**: one box of the plane. The box is tinted by the [[data:class A]] and [[neg:class B]] dots inside it, and that color is the tree\'s answer there. Hover a box to see its leaf in the diagram.',
			params: { scenario: 1, depth: 2 },
			...TRAIN_ONLY,
			controls: []
		},
		{
			title: 'Follow one point down',
			body: 'Drag the [[grad:probe]] around the plane. To classify it, the tree asks the root question first, then the question in the box it landed in, and so on down the [[grad:path]] in the diagram.\n\nThe leaf at the end answers with its majority class. The label beside the probe gives the leaf\'s class and how many of its dots agree.',
			params: { scenario: 2, depth: 2, probe: 1 },
			...TRAIN_ONLY,
			controls: []
		},
		{
			title: 'Finding the best first question',
			body: 'How does a tree pick its questions? It tries every cut and keeps the best. The chart shows how mixed the two sides of the [[grad:cut]] are for every cut position. Low means both sides are nearly pure, with 0 meaning each side holds a single class.\n\nMove **Cut at** to the lowest point of the curve, or press **Find best**. Switch **Cut direction** to try horizontal cuts too.',
			params: { scenario: 3, depth: 0, split: 1, axis: START_CUT.axis, cut: START_CUT.threshold },
			...TRAIN_ONLY,
			controls: ['axis', 'cut', 'best'],
			deepDive: {
				note: 'The mix of a group of dots is measured with Gini impurity. A cut is scored by the impurity of its two sides, weighted by how many dots each side holds. The tree tries every midpoint between neighbouring dots, on both axes, and keeps the lowest score.',
				math: [
					{ tex: 'G = 1 - p_A^2 - p_B^2 = 2\\,p_A\\,(1 - p_A)', caption: 'Gini impurity of one group: 0 when it holds a single class, 0.5 when it is an even mix' },
					{
						tex: 'G_{\\text{split}} = \\frac{n_L}{n} G_L + \\frac{n_R}{n} G_R',
						caption: 'The score of a cut: the lowest value over all candidate cuts is chosen'
					}
				]
			}
		},
		{
			title: 'Grow it one level at a time',
			body: 'Press **Grow**. Every box is split again by its own best question, found the same way as the first one. Each press adds a level, and the tree gets more leaves.\n\nTraining accuracy climbs because smaller boxes can follow the [[data:class A]] and [[neg:class B]] dots more closely.',
			params: { scenario: 4, depth: 1 },
			...TRAIN_ONLY,
			controls: ['grow', 'depth'],
			deepDive: {
				note: 'The tree is greedy: each question is the best one right now, without looking ahead. That is not always the best tree overall, but finding the best tree means trying every possible set of questions, which is far too slow.'
			}
		},
		{
			title: 'Grown until every leaf is pure',
			body: 'Let the tree grow to its maximum depth and every leaf ends up holding a single class. [[data:Training accuracy]] reaches 100%.\n\nLook at the tiny boxes drawn around single noisy dots. The tree did not learn the shape of the two moons. It memorized the dots.',
			params: { scenario: 5, depth: MAX_DEPTH },
			...TRAIN_ONLY,
			controls: ['depth']
		},
		{
			title: 'Too deep overfits',
			body: 'The [[ok:test points]] were not used to grow the tree. The chart shows both accuracies at every depth: [[data:training accuracy]] only climbs, while [[ok:test accuracy]] peaks and then falls.\n\nSlide **Max depth** and find the peak. Past it, the tree is learning noise.',
			params: { scenario: 6, depth: MAX_DEPTH, test: 1, chart: 1 },
			controls: ['depth']
		},
		{
			title: 'Trees are unstable',
			body: 'Press **New sample**. A fresh sample of the same moons gives a very different tree. The faint lines are the cuts of the trees grown on the other samples, and they disagree a lot.\n\nThis is high **variance**: the tree depends on exactly which dots it saw. Lower **Max depth** and the trees agree more.',
			params: { scenario: 7, depth: MAX_DEPTH, ghosts: 1 },
			...TRAIN_ONLY,
			controls: ['resample', 'depth'],
			deepDive: {
				note: 'A shallow tree asks few questions, so a few dots cannot change them much. A deep tree reshapes itself around individual dots. The Overfitting lesson calls this trade-off bias and variance.'
			}
		},
		{
			title: 'A forest votes',
			body: 'A **forest** is many trees that vote. Each tree is grown on a random resample of the dots, with some dots used twice and some left out, and it picks the direction of each question at random. So the trees differ, and their mistakes differ too.\n\nThe background is the average answer of all the trees, so the boundary is smoother, and the [[ok:test accuracy]] beats any single tree. Change **Trees** and watch the small trees below.',
			params: { scenario: 8, forest: 1, trees: 25, test: 1 },
			controls: ['trees'],
			deepDive: {
				note: 'Training each tree on a random resample and averaging the votes is called bagging, short for bootstrap aggregating. A random forest adds the random choice of direction at each question.',
				code: {
					lang: 'ts',
					source:
						"import { buildForest, forestProb } from '../ml/tree.ts';\n\nconst trees = buildForest(data, 25, 10, seed); // 25 trees, depth 10\nconst p = forestProb(trees, x1, x2);"
				}
			}
		}
	],
	questions,
	createScene: createDecisionTreesScene
};

export default lesson;

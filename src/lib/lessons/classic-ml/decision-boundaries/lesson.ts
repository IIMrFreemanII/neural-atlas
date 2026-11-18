import type { LessonDef } from '../../types.ts';
import { questions } from './questions.ts';
import { createDecisionBoundariesScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'decision-boundaries',
	track: 'classic-ml',
	title: 'Decision boundaries',
	summary: 'See how a straight line, a vote of neighbors and a tree of questions each carve up the same points.',
	defaults: { model: 0, dataset: 0, k: 7, depth: 3, noise: 0.05, trained: 1, scenario: 1 },
	discrete: ['trained', 'scenario'],
	controls: [
		{ kind: 'action', id: 'run', label: 'Train', icon: 'play' },
		{ kind: 'action', id: 'reset', label: 'Reset', icon: 'reset' },
		{
			kind: 'select',
			key: 'model',
			label: 'Model',
			options: [
				{ value: 0, label: 'Logistic regression' },
				{ value: 1, label: 'k-nearest neighbors' },
				{ value: 2, label: 'Decision tree' }
			]
		},
		{
			kind: 'select',
			key: 'dataset',
			label: 'Data',
			options: [
				{ value: 0, label: 'Blobs' },
				{ value: 1, label: 'Moons' },
				{ value: 2, label: 'Circles' },
				{ value: 3, label: 'XOR' }
			]
		},
		{ kind: 'slider', key: 'k', label: 'Neighbors (k)', min: 1, max: 25, step: 2, tone: 'param', digits: 0 },
		{ kind: 'slider', key: 'depth', label: 'Tree depth', min: 0, max: 8, step: 1, tone: 'param', digits: 0 },
		{ kind: 'slider', key: 'noise', label: 'Noise', min: 0, max: 0.3, step: 0.01, tone: 'data', digits: 2 }
	],
	readouts: [
		{ key: 'acc', label: 'Training accuracy', tone: 'ok' },
		{ key: 'info', label: 'Model' }
	],
	legend: [
		{ tone: 'data', label: 'Class A' },
		{ tone: 'neg', label: 'Class B' },
		{ tone: 'text', label: 'Boundary' },
		{ tone: 'grad', label: 'Misclassified' }
	],
	steps: [
		{
			title: 'Classifying means coloring the plane',
			body: 'Every dot has a class: [[data:class A]] or [[neg:class B]]. A classifier has to answer for every possible position, not only the dots, and the background color shows that answer.\n\nThe [[text:white line]] is where the answer flips. It is called the **decision boundary**.',
			params: { scenario: 1 },
			controls: []
		},
		{
			title: 'Logistic regression draws one straight line',
			body: 'This model can only draw a straight boundary. Its line starts in the wrong place, so some dots carry a [[grad:ring]] for a wrong answer.\n\nPress **Train** and watch the line turn and shift until the classes are apart. The color fades near the line because the model outputs a probability, not just a label.',
			params: { scenario: 2, trained: 0 },
			controls: ['run', 'reset'],
			deepDive: {
				note: 'The model scores a position with a weighted sum, squashes it into a probability, and training lowers the cross-entropy loss by gradient descent.',
				math: [
					{ tex: 'p(\\text{A} \\mid x) = \\sigma(w \\cdot x + b) = \\frac{1}{1 + e^{-(w_1 x_1 + w_2 x_2 + b)}}' },
					{
						tex: 'L = -\\frac{1}{n} \\sum_i \\big[ y_i \\log p_i + (1 - y_i) \\log (1 - p_i) \\big]',
						caption: 'The boundary is the line where p = 0.5, that is w · x + b = 0.'
					}
				]
			}
		},
		{
			title: 'A straight line is not always enough',
			body: 'These two interleaved moons cannot be split by any straight line. The [[grad:ringed dots]] sit on the wrong side, and no amount of training fixes that.\n\nTry the other datasets with the **Data** control and see which ones a line can handle.',
			params: { scenario: 3, dataset: 1 },
			controls: ['dataset']
		},
		{
			title: 'k-nearest neighbors asks the neighbors',
			body: 'This model has no training step. To classify a spot, it finds the **k** closest dots and takes a vote.\n\nMove the pointer over the map to see the vote: lines reach the neighbors, and the label counts how many are [[data:class A]] or [[neg:class B]].',
			params: { scenario: 4, model: 1, dataset: 1, k: 7 },
			controls: ['k']
		},
		{
			title: 'Small k memorizes, large k blurs',
			body: 'With **k** = 1 every noisy dot gets its own little island, and training accuracy is a perfect 100%. That is not a good sign, because the model has memorized the noise.\n\nRaise **k** and the islands dissolve into smoother regions. Add more **Noise** and compare.',
			params: { scenario: 5, model: 1, dataset: 1, k: 1, noise: 0.18 },
			controls: ['k', 'noise'],
			deepDive: {
				note: 'A small k follows the data closely and reacts to every outlier. A large k averages over many dots, so it is stable but can wash out real detail. The Overfitting lesson takes this trade-off further.'
			}
		},
		{
			title: 'A decision tree asks yes or no questions',
			body: 'Depth 1 is a single question, such as whether a dot sits above or below one height. Each extra level asks one more question inside every region.\n\nThe [[text:thin lines]] are the questions, so the boundary becomes a staircase of straight cuts. Raise **Tree depth** to add more.',
			params: { scenario: 6, model: 2, dataset: 1, depth: 1 },
			controls: ['depth'],
			deepDive: {
				note: 'At each node the tree tries every cut on each axis and keeps the one that makes the two sides purest. Purity is measured with Gini impurity. The Decision trees and forests lesson grows a tree one question at a time.',
				math: [
					{ tex: 'G = 1 - p_A^2 - p_B^2 = 2\\,p_A\\,(1 - p_A)' },
					{
						tex: 'G_{\\text{split}} = \\frac{n_L}{n} G_L + \\frac{n_R}{n} G_R',
						caption: 'The chosen cut is the one with the lowest weighted impurity.'
					}
				]
			}
		},
		{
			title: 'Same data, three answers',
			body: 'Switch the **Model** and the **Data** and look at the signature of each: logistic regression gives one line, k-nearest neighbors gives soft blobs, and a tree gives boxes.\n\nNone is best everywhere. Each one fits some shapes well and others badly.',
			params: { scenario: 7, model: 2, dataset: 2, depth: 4 }
		}
	],
	questions,
	createScene: createDecisionBoundariesScene
};

export default lesson;

import { bestSplit, splitImpurity } from '../../../ml/tree.ts';
import type { Question } from '../../types.ts';
import { MAX_DEPTH, START_CUT, bestTestDepth, trainingData } from './data.ts';

// Every question below uses its own `scenario`, and none changes `sample`, so the scene shows sample 0.
const train = trainingData(0);
const best = bestSplit(train)!;
/** How close to the best cut counts as pure enough, in Gini units. */
const SLACK = 0.01;
const lowest = best.impurity + SLACK;

/** The best cut on the slider's grid, as the "Find best" button rounds it. */
export const bestCut = { axis: best.axis, cut: Math.round(best.threshold * 100) / 100 };

const bestDepth = bestTestDepth();

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'This tree classifies every training dot correctly. What should you expect on new points from the same moons?\n\nSlide **Max depth** down to see how the boxes change.',
		params: { scenario: 9, depth: MAX_DEPTH },
		controls: ['depth'],
		options: [
			{
				text: 'Also 100%, because it never makes a mistake.',
				why: 'The 100% is measured on the dots the tree was grown from. New points are not those dots, and a tree that has memorized them can be wrong on points it has not seen.'
			},
			{
				text: 'Lower accuracy, because many small boxes were shaped around noisy dots.',
				correct: true,
				why: 'The tree grew until every box was pure, so it carved a tiny box around each noisy dot. New points do not land on those dots, and the tiny boxes answer wrongly for many of them.'
			},
			{
				text: 'Exactly 50%, because it learned nothing real.',
				why: 'It still learned the broad shape of the two moons, so most new points land in a box with the right class. It is only the small boxes around noisy dots that do harm.'
			},
			{
				text: 'Higher than on the training dots, because more leaves means more detail.',
				why: 'More detail helps only when the detail is real. Here it is noise from the training dots, so more leaves can only help on those dots and not on new points.'
			}
		]
	},
	{
		kind: 'task',
		prompt: 'Place the first question yourself: move the cut until the two sides are as pure as they can be.',
		params: { scenario: 10, depth: 0, split: 1, axis: START_CUT.axis, cut: START_CUT.threshold },
		controls: ['axis', 'cut'],
		goal: ({ params: p }) => splitImpurity(train, p.axis === 1 ? 1 : 0, p.cut) <= lowest,
		hint: 'Watch the [[grad:marker]] on the chart and the number beside it. Try both cut directions, and look for the lowest point of the curve.',
		solved:
			'This is what the tree does at its root: it tries every cut on both axes and keeps the one with the lowest impurity. Each box then repeats the search on its own dots.',
		example: { params: { axis: bestCut.axis, cut: bestCut.cut } }
	},
	{
		kind: 'task',
		prompt: 'Without the chart, find the depth whose tree does best on the [[ok:test points]].',
		params: { scenario: 11, depth: MAX_DEPTH, test: 1, chart: 0 },
		controls: ['depth'],
		goal: ({ params: p }) => Math.round(p.depth) === bestDepth,
		hint: 'Slide **Max depth** and watch the **Test accuracy** chip. You are looking for its highest value.',
		solved:
			'Test accuracy rises while the tree learns the real shape, then falls once it starts chasing noisy dots. The best tree is neither the shallowest nor the deepest.',
		example: { params: { depth: bestDepth } }
	}
];

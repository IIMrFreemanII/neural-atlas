import { buildTree, treeProb } from '../../../ml/tree.ts';
import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';
import { makeBoundaryData, N_POINTS } from './data.ts';

/** Training accuracy in percent of a tree of the given depth, as the scene computes it. */
function treeAccuracy(dataset: number, noise: number, depth: number): number {
	const data = makeBoundaryData(dataset, noise, 11);
	const tree = buildTree(data, depth);
	let correct = 0;
	for (let i = 0; i < N_POINTS; i++) {
		const p = treeProb(tree, data.x[2 * i], data.x[2 * i + 1]);
		if ((p >= 0.5 ? 1 : 0) === data.y[i]) correct++;
	}
	return (correct / N_POINTS) * 100;
}

/** The smallest depth whose tree reaches the accuracy target on this data. */
function shallowest(dataset: number, noise: number, target: number): number {
	for (let d = 0; d <= 8; d++) if (shown(treeAccuracy(dataset, noise, d), 0) >= target) return d;
	return 8;
}

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'The straight-line model is trained on the [[data:class A]] and [[neg:class B]] dots of the **XOR** data, where the two classes sit in opposite corners. About how many dots can it classify correctly?\n\nDecide first, then check with the **Data** control.',
		params: { model: 0, dataset: 0, trained: 1 },
		controls: ['dataset'],
		options: [
			{
				text: 'Almost all of them, because the classes sit in four tidy groups.',
				why: 'Tidy groups do not help a straight line. In XOR each side of any line holds dots of both classes, so the line cannot pull them apart.'
			},
			{
				text: 'Almost all of them, if it trains for long enough.',
				why: 'Training moves the line to its best position, but its best position is still a straight line. More steps cannot add a bend that the model does not have.'
			},
			{
				text: 'About half of them, no better than guessing.',
				correct: true,
				why: 'Any straight line cuts XOR so that each side holds both classes in similar numbers. The best the line can do is a little above half, and training does not change that.'
			},
			{
				text: 'Very few of them, because the line puts the classes on the wrong sides.',
				why: 'A line that gets most dots wrong could simply be flipped to get most of them right. The model would find that flip during training, so it never settles far below half.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'The [[text:tree]] is learning XOR. Find the smallest **Tree depth** at which it classifies at least 97% of the dots correctly.',
		params: { model: 2, dataset: 3, depth: 1, noise: 0.05 },
		controls: ['depth'],
		goal: ({ params: p }) =>
			p.model === 2 &&
			p.depth === shallowest(p.dataset, p.noise, 97) &&
			shown(treeAccuracy(p.dataset, p.noise, p.depth), 0) >= 97,
		hint: 'Each level asks one more question inside every region. In XOR the first cut leaves both sides mixed, so it takes more than one or two. Raise the depth, then step back down to the lowest level that still reaches the target.',
		solved:
			'The first cuts do little, because every side of an XOR cut still holds both classes. Only once the tree has cut each side again do the four corners come apart, and a deeper tree only adds questions that were not needed.',
		example: { params: { depth: 4 } }
	},
	{
		kind: 'choice',
		prompt:
			'The [[data:class A]] and [[neg:class B]] dots of the noisy moons are classified by their nearest neighbors. With **k** = 1 the training accuracy is 100%. Raise **k** to 7 and it drops to 93%. What does the drop tell you?',
		params: { scenario: 5, model: 1, dataset: 1, k: 1, noise: 0.18 },
		controls: ['k'],
		options: [
			{
				text: 'The model got worse, because it now gets more dots wrong.',
				why: 'Training accuracy only says how well the model knows these dots. A model that gets every one right can have done it by copying the noise.'
			},
			{
				text: 'With k = 1 the model copied the noise, and k = 7 gives that up for a smoother boundary.',
				correct: true,
				why: 'With k = 1 every noisy dot makes an island of its own, so each dot is right about itself. With k = 7 a stray dot is outvoted by its neighbors and counts as wrong, but the boundary no longer bends around the noise.'
			},
			{
				text: 'A bigger k always raises accuracy, so the model must be broken.',
				why: 'A bigger k averages over more dots. That smooths the boundary, and a smooth boundary cannot follow every stray dot, so the training accuracy drops.'
			}
		]
	}
];

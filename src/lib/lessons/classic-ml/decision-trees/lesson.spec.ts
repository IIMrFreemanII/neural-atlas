import { describe, expect, it } from 'vitest';
import { bestSplit, splitImpurity } from '../../../ml/tree.ts';
import { startState } from '../../questions.ts';
import { START_CUT, trainingData } from './data.ts';
import lesson from './lesson.ts';

const train = trainingData(0);
const best = bestSplit(train)!;

describe('decision trees lesson pages', () => {
	// The scene starts a page afresh (the probe, the hover) when `scenario` changes.
	it('gives every step and question its own scenario number', () => {
		const pages = [...lesson.steps, ...(lesson.questions ?? [])];
		const numbers = pages.map((p) => ({ ...lesson.defaults, ...p.params }).scenario);
		expect(numbers.every((n) => Number.isInteger(n))).toBe(true);
		expect(new Set(numbers).size).toBe(pages.length);
	});

	it('starts the "best first question" step and its question at the starting cut', () => {
		const step = lesson.steps.find((s) => s.params?.split === 1)!;
		const q = lesson.questions!.find((q) => q.kind === 'task' && q.params?.split === 1)!;
		for (const p of [step.params!, q.params!]) {
			expect(p.axis).toBe(START_CUT.axis);
			expect(p.cut).toBe(START_CUT.threshold);
		}
	});

	it('starts the cut question unsolved and solves it with the best cut', () => {
		const q = lesson.questions!.find((q) => q.kind === 'task' && q.params?.split === 1);
		if (!q || q.kind !== 'task') throw new Error('no cut question');
		const start = startState(lesson, q);
		expect(q.goal(start)).toBe(false);
		expect(q.goal({ ...start, params: { ...start.params, ...q.example!.params } })).toBe(true);
		expect(splitImpurity(train, START_CUT.axis, START_CUT.threshold)).toBeGreaterThan(best.impurity + 0.01);
	});

	it('starts the depth question unsolved', () => {
		const q = lesson.questions!.find((q) => q.kind === 'task' && q.params?.test === 1);
		if (!q || q.kind !== 'task') throw new Error('no depth question');
		expect(q.goal(startState(lesson, q))).toBe(false);
	});
});

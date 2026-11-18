import { describe, expect, it } from 'vitest';
import { checkQuestions, questionControls, startState } from '../../questions.ts';
import lesson from './lesson.ts';

describe('vision transformer lesson pages', () => {
	// The scene starts a page afresh (the drawing and the shuffle) when `scenario` changes.
	it('gives every step and question its own scenario number', () => {
		const pages = [...lesson.steps, ...(lesson.questions ?? [])];
		const numbers = pages.map((p) => ({ ...lesson.defaults, ...p.params }).scenario);
		expect(numbers.every((n) => Number.isInteger(n))).toBe(true);
		expect(new Set(numbers).size).toBe(pages.length);
	});

	it('has well formed questions', () => {
		expect(checkQuestions(lesson)).toEqual([]);
	});

	it('starts the "find the head" task unsolved, and its example solves it', () => {
		const q = lesson.questions!.find((x) => x.kind === 'task')!;
		if (q.kind !== 'task') throw new Error('expected a task');
		const start = startState(lesson, q);
		expect(q.goal(start)).toBe(false);
		expect(questionControls(lesson, q).map((c) => ('key' in c ? c.key : c.id))).toEqual(['layer', 'head']);
		expect(q.goal({ ...start, params: { ...start.params, ...q.example!.params } })).toBe(true);
	});
});

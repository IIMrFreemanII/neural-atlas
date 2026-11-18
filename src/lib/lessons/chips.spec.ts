import { describe, expect, test } from 'vitest';
import { allLessons, lessonKey } from '../content/curriculum.ts';
import { chipProblems, pageLegend, pageReadouts } from './chips.ts';
import type { LessonDef } from './types.ts';

describe.each(allLessons.filter((m) => m.status === 'ready').map((m) => [lessonKey(m.track, m.id), m] as const))(
	'legend and readouts of %s',
	(_key, meta) => {
		test('every page names only chips the lesson has', async () => {
			expect(chipProblems((await meta.load!()).default)).toEqual([]);
		});
	}
);

describe('chips', () => {
	const lesson = {
		legend: [
			{ tone: 'data', label: 'A' },
			{ tone: 'grad', label: 'B' }
		],
		readouts: [{ key: 'x', label: 'X' }],
		steps: [{ title: '', body: '' }, { title: '', body: '', legend: ['B', 'C'], readouts: [] }],
		questions: [{ kind: 'choice', prompt: '', options: [], readouts: ['y'] }]
	} as unknown as LessonDef;

	test('a page without lists shows everything, a list filters in the lesson order', () => {
		expect(pageLegend(lesson, lesson.steps[0]).map((l) => l.label)).toEqual(['A', 'B']);
		expect(pageLegend(lesson, lesson.steps[1]).map((l) => l.label)).toEqual(['B']);
		expect(pageReadouts(lesson, lesson.steps[1])).toEqual([]);
	});

	test('unknown names are reported', () => {
		expect(chipProblems(lesson)).toEqual([
			"step 2: legend 'C' does not exist in the lesson",
			"question 1: readout 'y' does not exist in the lesson"
		]);
	});
});

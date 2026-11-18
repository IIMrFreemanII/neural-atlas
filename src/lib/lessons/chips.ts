/**
 * Which legend chips and readouts a page (a step or a question) shows. A page lists them by legend label
 * and readout key; a page that lists nothing shows all of the lesson's. No DOM and no GPU code.
 */
import type { LessonDef, Question, Readout, Step } from './types.ts';

type Page = Pick<Step | Question, 'legend' | 'readouts'>;
type LegendItem = NonNullable<LessonDef['legend']>[number];

export function pageLegend(lesson: LessonDef, page: Page): LegendItem[] {
	const all = lesson.legend ?? [];
	return page.legend === undefined ? all : all.filter((l) => page.legend!.includes(l.label));
}

export function pageReadouts(lesson: LessonDef, page: Page): Readout[] {
	const all = lesson.readouts ?? [];
	return page.readouts === undefined ? all : all.filter((r) => page.readouts!.includes(r.key));
}

/** What is wrong with the pages' legend and readout lists: names that the lesson does not have. */
export function chipProblems(lesson: LessonDef): string[] {
	const labels = new Set((lesson.legend ?? []).map((l) => l.label));
	const keys = new Set((lesson.readouts ?? []).map((r) => r.key));
	const out: string[] = [];
	const pages: [string, Page][] = [
		...lesson.steps.map((s, i): [string, Page] => [`step ${i + 1}`, s]),
		...(lesson.questions ?? []).map((q, i): [string, Page] => [`question ${i + 1}`, q])
	];
	for (const [name, page] of pages) {
		for (const l of page.legend ?? []) if (!labels.has(l)) out.push(`${name}: legend '${l}' does not exist in the lesson`);
		for (const k of page.readouts ?? []) if (!keys.has(k)) out.push(`${name}: readout '${k}' does not exist in the lesson`);
	}
	return out;
}

/**
 * Helpers for a lesson's "check your understanding" questions: the state a question starts from, the
 * rounding a goal should use, and a checker that lists what is wrong with a lesson's questions.
 * No DOM and no GPU code.
 */
import { parseBody } from './markup.ts';
import type { Control, LessonDef, Params, Question, QuizState } from './types.ts';

export const controlId = (c: Control): string => ('id' in c ? c.id : c.key);

/**
 * A value as the learner reads it on screen: rounded to `digits` decimals (readouts and sliders show 2
 * unless they say otherwise). Compare this in a goal, not the raw value, so that the page never says
 * "not done" while the number on screen already meets the target. Returns NaN for anything that is not a
 * finite number, and every comparison with NaN is false.
 */
export function shown(v: unknown, digits = 2): number {
	const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN;
	return Number.isFinite(n) ? Number(n.toFixed(digits)) : NaN;
}

/** Params while a question is open, like a step's. */
export function questionParams(lesson: LessonDef, q: Question): Params {
	return { ...lesson.defaults, ...q.params };
}

/** The lesson's controls that are visible while a question is open. */
export function questionControls(lesson: LessonDef, q: Question): Control[] {
	if (q.controls === undefined) return lesson.controls;
	return lesson.controls.filter((c) => q.controls!.includes(controlId(c)));
}

/** Content of every 'text' control before the learner types. */
export function initialTexts(lesson: LessonDef): Record<string, string> {
	const out: Record<string, string> = {};
	for (const c of lesson.controls) if (c.kind === 'text') out[c.id] = c.initial;
	return out;
}

/** What a goal sees when its question opens, before the scene has emitted anything. */
export function startState(lesson: LessonDef, q: Question): QuizState {
	return { params: questionParams(lesson, q), readouts: {}, texts: initialTexts(lesson) };
}

const onGrid = (v: number, min: number, step: number) => {
	const k = (v - min) / step;
	return Math.abs(k - Math.round(k)) < 1e-6;
};

/** Problems in one piece of learner-facing text. */
function textProblems(where: string, text: string): string[] {
	const out: string[] = [];
	if (!text.trim()) return [`${where}: is empty`];
	if (text.includes('—')) out.push(`${where}: contains an em dash`);
	if (text.includes('!')) out.push(`${where}: contains an exclamation mark`);
	for (const para of parseBody(text)) {
		for (const run of para) {
			if (run.kind === 'text' && /\[\[|\]\]|\*\*/.test(run.text)) {
				out.push(`${where}: has markup that does not parse: "${run.text.slice(0, 60)}"`);
			}
		}
	}
	return out;
}

/**
 * Everything wrong with a lesson's questions, one sentence each. An empty list means they are fine.
 * A lesson without questions gives an empty list.
 */
export function checkQuestions(lesson: LessonDef): string[] {
	const qs = lesson.questions;
	if (qs === undefined) return [];
	const out: string[] = [];
	if (qs.length < 2 || qs.length > 3) out.push(`has ${qs.length} questions, expected 2 or 3`);
	const ids = new Set(lesson.controls.map(controlId));

	qs.forEach((q, i) => {
		const at = `question ${i + 1}`;
		out.push(...textProblems(`${at} prompt`, q.prompt));
		for (const k of Object.keys(q.params ?? {})) {
			if (!(k in lesson.defaults)) out.push(`${at}: param '${k}' is not in the lesson's defaults`);
		}
		for (const id of q.controls ?? []) {
			if (!ids.has(id)) out.push(`${at}: control '${id}' does not exist in the lesson`);
		}

		if (q.kind === 'choice') {
			if (q.options.length < 3 || q.options.length > 4) {
				out.push(`${at}: has ${q.options.length} options, expected 3 or 4`);
			}
			const right = q.options.filter((o) => o.correct === true).length;
			if (right !== 1) out.push(`${at}: has ${right} correct options, expected exactly 1`);
			if (new Set(q.options.map((o) => o.text.trim())).size !== q.options.length) {
				out.push(`${at}: two options have the same text`);
			}
			q.options.forEach((o, j) => {
				out.push(...textProblems(`${at} option ${j + 1} text`, o.text));
				out.push(...textProblems(`${at} option ${j + 1} why`, o.why));
			});
			return;
		}

		out.push(...textProblems(`${at} hint`, q.hint));
		out.push(...textProblems(`${at} solved`, q.solved));
		const start = startState(lesson, q);
		try {
			if (q.goal(start)) out.push(`${at}: the goal is already reached in the starting state`);
		} catch (e) {
			out.push(`${at}: the goal throws in the starting state (${String(e)})`);
		}
		if (!q.example) return;

		const visible = questionControls(lesson, q);
		const exParams = q.example.params ?? {};
		const exTexts = q.example.texts ?? {};
		if (Object.keys(exParams).length + Object.keys(exTexts).length === 0) {
			out.push(`${at}: the example is empty`);
		}
		for (const [k, v] of Object.entries(exParams)) {
			const c = visible.find((c) => 'key' in c && c.key === k);
			if (!c) {
				out.push(`${at}: the example sets '${k}', but no visible control writes it`);
			} else if (c.kind === 'slider') {
				if (v < c.min - 1e-9 || v > c.max + 1e-9 || !onGrid(v, c.min, c.step)) {
					out.push(`${at}: the example sets '${k}' to ${v}, which the slider cannot reach`);
				}
			} else if (c.kind === 'toggle') {
				if (v !== 0 && v !== 1) out.push(`${at}: the example sets toggle '${k}' to ${v}`);
			} else if (c.kind === 'select') {
				if (!c.options.some((o) => o.value === v)) {
					out.push(`${at}: the example sets '${k}' to ${v}, which is not one of its options`);
				}
			}
		}
		for (const [id, text] of Object.entries(exTexts)) {
			const c = visible.find((c) => c.kind === 'text' && c.id === id);
			if (!c || c.kind !== 'text') {
				out.push(`${at}: the example types into '${id}', but no visible text control has that id`);
			} else if (text.length > (c.maxLength ?? 60)) {
				out.push(`${at}: the example text for '${id}' is longer than the control allows`);
			}
		}
		try {
			const reached = q.goal({
				params: { ...start.params, ...exParams },
				readouts: {},
				texts: { ...start.texts, ...exTexts }
			});
			if (!reached) out.push(`${at}: the example does not reach the goal`);
		} catch (e) {
			out.push(`${at}: the goal throws on the example (${String(e)})`);
		}
	});
	return out;
}

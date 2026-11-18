import { describe, expect, test } from 'vitest';
import { allLessons, lessonKey } from '../content/curriculum.ts';
import { createNullScene } from './null-scene.ts';
import { checkQuestions, questionControls, shown, startState } from './questions.ts';
import type { LessonDef, Question } from './types.ts';

/** Every lesson must have questions. */
const REQUIRE_ALL = true;

describe.each(allLessons.map((m) => [lessonKey(m.track, m.id), m] as const))('questions of %s', (_key, meta) => {
	test('are well formed', async () => {
		const def = (await meta.load!()).default;
		if (!def.questions) {
			expect(REQUIRE_ALL, 'this lesson has no questions').toBe(false);
			return;
		}
		expect(checkQuestions(def)).toEqual([]);
	});
});

const choice = (over: Partial<Extract<Question, { kind: 'choice' }>> = {}): Question => ({
	kind: 'choice',
	prompt: 'Which one?',
	options: [
		{ text: 'A', correct: true, why: 'Because.' },
		{ text: 'B', why: 'No.' },
		{ text: 'C', why: 'No.' }
	],
	...over
});

const task = (over: Partial<Extract<Question, { kind: 'task' }>> = {}): Question => ({
	kind: 'task',
	prompt: 'Raise **a** above 0.5.',
	goal: ({ params }) => shown(params.a) > 0.5,
	hint: 'Use the slider.',
	solved: 'Done.',
	example: { params: { a: 0.75 } },
	...over
});

function lesson(questions: Question[]): LessonDef {
	return {
		id: 'fake',
		track: 'fundamentals',
		title: 'Fake',
		summary: '',
		defaults: { a: 0, on: 0, mode: 0 },
		controls: [
			{ kind: 'slider', key: 'a', label: 'A', min: 0, max: 1, step: 0.05 },
			{ kind: 'toggle', key: 'on', label: 'On' },
			{ kind: 'select', key: 'mode', label: 'Mode', options: [{ value: 0, label: 'x' }, { value: 2, label: 'y' }] },
			{ kind: 'text', id: 'say', label: 'Say', initial: 'hello', maxLength: 8 }
		],
		steps: [{ title: 'One', body: 'Text.' }],
		questions,
		createScene: createNullScene
	};
}

const problems = (...qs: Question[]) => checkQuestions(lesson(qs));

describe('checkQuestions', () => {
	test('accepts good questions and a lesson without any', () => {
		expect(problems(choice(), task())).toEqual([]);
		expect(checkQuestions({ ...lesson([]), questions: undefined })).toEqual([]);
	});

	test('counts questions, options and correct answers', () => {
		expect(problems(choice())).toEqual(['has 1 questions, expected 2 or 3']);
		const two = choice({ options: [{ text: 'A', correct: true, why: 'y' }, { text: 'B', why: 'n' }] });
		expect(problems(two, task())).toEqual(['question 1: has 2 options, expected 3 or 4']);
		const none = choice({ options: [{ text: 'A', why: 'y' }, { text: 'B', why: 'n' }, { text: 'C', why: 'n' }] });
		expect(problems(none, task())).toEqual(['question 1: has 0 correct options, expected exactly 1']);
		const same = choice({ options: [{ text: 'A', correct: true, why: 'y' }, { text: 'A', why: 'n' }, { text: 'C', why: 'n' }] });
		expect(problems(same, task())).toEqual(['question 1: two options have the same text']);
	});

	test('checks the text', () => {
		expect(problems(choice({ prompt: ' ' }), task())).toEqual(['question 1 prompt: is empty']);
		expect(problems(choice({ prompt: 'Wow!' }), task())).toEqual(['question 1 prompt: contains an exclamation mark']);
		expect(problems(choice({ prompt: 'A — B' }), task())).toEqual(['question 1 prompt: contains an em dash']);
		expect(problems(choice({ prompt: 'A [[nope:term]] here' }), task())[0]).toMatch(/markup that does not parse/);
		expect(problems(choice({ prompt: 'A [[data:term]] and **bold** here' }), task())).toEqual([]);
	});

	test('checks params and controls against the lesson', () => {
		expect(problems(choice({ params: { zz: 1 } }), task())).toEqual(["question 1: param 'zz' is not in the lesson's defaults"]);
		expect(problems(choice({ controls: ['zz'] }), task())).toEqual(["question 1: control 'zz' does not exist in the lesson"]);
	});

	test('a task must start unsolved and its example must solve it', () => {
		expect(problems(choice(), task({ params: { a: 0.8 } }))).toEqual(['question 2: the goal is already reached in the starting state']);
		expect(problems(choice(), task({ example: { params: { a: 0.5 } } }))).toEqual(['question 2: the example does not reach the goal']);
		expect(problems(choice(), task({ example: {} }))).toContain('question 2: the example is empty');
		expect(problems(choice(), task({ example: undefined }))).toEqual([]);
		const throws = task({ goal: ({ readouts }) => (readouts.x as string).length > 0, example: undefined });
		expect(problems(choice(), throws)[0]).toMatch(/the goal throws in the starting state/);
	});

	test('an example must be reachable with the visible controls', () => {
		expect(problems(choice(), task({ example: { params: { a: 0.77 } } }))).toEqual(["question 2: the example sets 'a' to 0.77, which the slider cannot reach"]);
		expect(problems(choice(), task({ example: { params: { a: 1.5 } } }))).toEqual(["question 2: the example sets 'a' to 1.5, which the slider cannot reach"]);
		expect(problems(choice(), task({ controls: ['on'] }))).toEqual(["question 2: the example sets 'a', but no visible control writes it"]);
		const sel = task({ goal: ({ params }) => params.mode === 1, example: { params: { mode: 1 } } });
		expect(problems(choice(), sel)).toEqual(["question 2: the example sets 'mode' to 1, which is not one of its options"]);
		const text = task({ goal: ({ texts }) => texts.say === 'bye', example: { texts: { say: 'bye' } } });
		expect(problems(choice(), text)).toEqual([]);
		const long = task({ goal: ({ texts }) => texts.say.length > 8, example: { texts: { say: 'much too long' } } });
		expect(problems(choice(), long)).toEqual(["question 2: the example text for 'say' is longer than the control allows"]);
	});
});

describe('helpers', () => {
	test('shown rounds like the screen does and gives NaN for non-numbers', () => {
		expect(shown(0.896)).toBe(0.9);
		expect(shown(0.004)).toBe(0);
		expect(shown(2.2e-16)).toBe(0);
		expect(shown(1.2346, 3)).toBe(1.235);
		expect(shown('0.5')).toBe(0.5);
		expect(shown('11 tokens')).toBeNaN();
		expect(shown(undefined)).toBeNaN();
		expect(shown('')).toBeNaN();
		expect(shown(NaN) > 0).toBe(false);
	});

	test('startState and questionControls follow the question', () => {
		const q = task({ params: { on: 1 }, controls: ['a'] });
		const def = lesson([choice(), q]);
		expect(startState(def, q)).toEqual({ params: { a: 0, on: 1, mode: 0 }, readouts: {}, texts: { say: 'hello' } });
		expect(questionControls(def, q).map((c) => ('id' in c ? c.id : c.key))).toEqual(['a']);
		expect(questionControls(def, choice()).length).toBe(4);
	});
});

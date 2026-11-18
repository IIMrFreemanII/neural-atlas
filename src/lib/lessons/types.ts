/**
 * Lesson and curriculum contracts. Types only. Do not add runtime code here.
 */
import type { SceneFactory, Tone } from '../gpu/types.ts';

export type TrackId = 'fundamentals' | 'classic-ml' | 'llm' | 'vision';

/** All scene params are a flat map of numbers. Toggles are 0 or 1. Selects are an option's numeric value. */
export type Params = Record<string, number>;

export type Control =
	| {
			kind: 'slider';
			/** Param key this slider writes. */
			key: string;
			label: string;
			min: number;
			max: number;
			step: number;
			/** Accent color of the slider fill. Default 'param'. */
			tone?: Tone;
			/** Decimal places in the value readout. Default 2. */
			digits?: number;
	  }
	| { kind: 'toggle'; key: string; label: string }
	| { kind: 'select'; key: string; label: string; options: { value: number; label: string }[] }
	| {
			kind: 'action';
			/** Sent to scene.command(id). */
			id: string;
			label: string;
			icon?: 'play' | 'step' | 'reset' | 'shuffle';
	  }
	| {
			kind: 'text';
			/** Every edit is sent to scene.command(`${id}:${value}`). Steps list this id in `controls`. */
			id: string;
			label: string;
			/** Text shown before the learner types. The scene must start from the same text. */
			initial: string;
			/** Default 60. */
			maxLength?: number;
	  };

/** A live value the scene emits with ctx.emit(key, value), shown as a chip over the stage. */
export interface Readout {
	key: string;
	label: string;
	tone?: Tone;
	/** Decimal places when the value is a number. Default 2. */
	digits?: number;
}

export interface DeepDive {
	/** KaTeX source strings, rendered in display mode, in order. */
	math?: { tex: string; caption?: string }[];
	code?: { lang: 'ts' | 'python'; source: string };
	/** Plain text shown above the math. */
	note?: string;
}

export interface Step {
	title: string;
	/**
	 * Narration text. Paragraphs are separated by a blank line. Two inline marks are supported:
	 *   **bold**            -> emphasized text
	 *   [[tone:some text]]  -> a term colored with that Tone; hovering it highlights the scene
	 * No HTML.
	 */
	body: string;
	/** Params for this step = { ...lesson.defaults, ...step.params }. Not cumulative across steps. */
	params?: Params;
	/**
	 * Which controls are visible at this step: slider/toggle/select keys and action/text ids.
	 * Undefined = all of the lesson's controls. Empty array = none.
	 */
	controls?: string[];
	deepDive?: DeepDive;
}

/** What a task question's goal can look at. */
export interface QuizState {
	/** The lesson params right now (what the controls show). */
	params: Params;
	/** Latest value of each readout the scene has emitted, by Readout.key. A key is missing until the scene emits it. */
	readouts: Record<string, number | string>;
	/** Current content of each 'text' control, by id. */
	texts: Record<string, string>;
}

interface QuestionBase {
	/** The question, in the narration markup of Step.body. */
	prompt: string;
	/** Params while this question is open = { ...lesson.defaults, ...params }, exactly like a step. */
	params?: Params;
	/** Visible controls while this question is open, like Step.controls. Undefined = all. */
	controls?: string[];
}

/** Pick one answer. The visual stays live, so the learner can try things out before answering. */
export interface ChoiceQuestion extends QuestionBase {
	kind: 'choice';
	/**
	 * 3 or 4 options, exactly one with `correct: true`. `why` is shown when the option is picked: for the
	 * correct one it gives the reason, for a wrong one it says what is wrong with that idea.
	 */
	options: { text: string; correct?: true; why: string }[];
}

/** Reach a state in the visual. Solved the first time `goal` is true after the learner touched something. */
export interface TaskQuestion extends QuestionBase {
	kind: 'task';
	/** Must be false in the question's starting state and must not throw when readouts are missing. */
	goal: (s: QuizState) => boolean;
	/** Shown when the learner asks for a hint. Points the way without giving the settings. */
	hint: string;
	/** Shown once the goal is reached: why it worked. */
	solved: string;
	/**
	 * Control settings that reach the goal from the starting state. Required when `goal` reads only params
	 * and texts (a unit test then proves the task can be solved). Leave it out only when `goal` reads readouts.
	 */
	example?: { params?: Params; texts?: Record<string, string> };
}

export type Question = ChoiceQuestion | TaskQuestion;

export interface LessonDef {
	/** URL slug, unique within its track, e.g. 'neuron'. */
	id: string;
	track: TrackId;
	title: string;
	summary: string;
	defaults: Params;
	/**
	 * Param keys that must jump instead of tween when the step changes.
	 * Keys of toggle and select controls always jump and need not be listed.
	 */
	discrete?: string[];
	controls: Control[];
	readouts?: Readout[];
	legend?: { tone: Tone; label: string }[];
	steps: Step[];
	/** "Check your understanding": 2 or 3 questions, each shown on its own page after the last step. */
	questions?: Question[];
	createScene: SceneFactory;
}

export interface LessonMeta {
	id: string;
	track: TrackId;
	title: string;
	summary: string;
	minutes: number;
	/** 'soon' lessons are listed but not openable. */
	status: 'ready' | 'soon';
	/** Lesson keys in the form '<track>/<id>'. */
	prerequisites: string[];
	/** Present only when status is 'ready'. */
	load?: () => Promise<{ default: LessonDef }>;
}

export interface Track {
	id: TrackId;
	title: string;
	blurb: string;
	tone: Tone;
	lessons: LessonMeta[];
}

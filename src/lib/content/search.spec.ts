import { describe, expect, it } from 'vitest';
import { highlight, matchRange, rankLessons, rankSteps, type StepEntry } from './search.ts';

describe('matchRange', () => {
	it('finds a substring ignoring case', () => {
		expect(matchRange('Gradient descent', 'GRAD')).toEqual([0, 4]);
		expect(matchRange('Vanishing gradient', 'grad')).toEqual([10, 14]);
	});

	it('ignores diacritics in the text and the query', () => {
		expect(matchRange('Café au lait', 'cafe')).toEqual([0, 4]);
		expect(matchRange('cafe', 'café')).toEqual([0, 4]);
	});

	it('prefers the start of a word to an earlier match inside a word', () => {
		// 'ar' is inside 'learn' first, but starts the word 'array'.
		expect(matchRange('learn array', 'ar')).toEqual([6, 8]);
	});

	it('falls back to a match inside a word', () => {
		expect(matchRange('Backpropagation', 'prop')).toEqual([4, 8]);
	});

	it('returns null for no match or a blank query', () => {
		expect(matchRange('neuron', 'xyz')).toBeNull();
		expect(matchRange('neuron', '   ')).toBeNull();
		expect(matchRange('neuron', '')).toBeNull();
	});

	it('maps back to the original text when folding changes its length', () => {
		const text = 'école normale'; // "école" with a combining accent
		const r = matchRange(text, 'ecole');
		expect(r).toEqual([0, 6]);
		expect(text.slice(...r!)).toBe('école');
	});
});

describe('highlight', () => {
	it('splits around the match and joins back to the text', () => {
		const segs = highlight('Gradient descent', 'dient');
		expect(segs).toEqual([
			{ text: 'Gra', mark: false },
			{ text: 'dient', mark: true },
			{ text: ' descent', mark: false }
		]);
		expect(segs.map((s) => s.text).join('')).toBe('Gradient descent');
	});

	it('returns the whole text unmarked without a match', () => {
		expect(highlight('neuron', 'zz')).toEqual([{ text: 'neuron', mark: false }]);
	});
});

const lesson = (id: string, title: string, summary: string, status: 'ready' | 'soon' = 'ready') => ({
	id,
	title,
	summary,
	status
});

describe('rankLessons', () => {
	const lessons = [
		lesson('a', 'Backpropagation', 'The chain rule hands out gradient shares.'),
		lesson('b', 'Gradient descent', 'Step downhill.'),
		lesson('c', 'Loss landscape', 'A surface over parameters.'),
		lesson('d', 'Gradient clipping', 'Soon.', 'soon'),
		lesson('e', 'Using the gradient', 'Word start in the middle.')
	];

	it('ranks title starts, then word starts, then matches in the summary', () => {
		expect(rankLessons(lessons, 'grad').map((l) => l.id)).toEqual(['b', 'e', 'a']);
	});

	it('leaves out lessons that are not ready', () => {
		expect(rankLessons(lessons, 'clipping')).toEqual([]);
	});

	it('keeps curriculum order among equal matches and respects the limit', () => {
		expect(rankLessons(lessons, 'a', 2).map((l) => l.id)).toEqual(['a', 'b']);
	});

	it('returns nothing for a blank query', () => {
		expect(rankLessons(lessons, ' ')).toEqual([]);
	});
});

describe('rankSteps', () => {
	const step = (title: string, stepIndex: number): StepEntry => ({
		track: 'fundamentals',
		lesson: 'backpropagation',
		lessonTitle: 'Backpropagation',
		stepIndex,
		title
	});

	it('ranks by title match kind and limits the count', () => {
		const steps = [step('Nudge every weight', 0), step('Every weight gets a gradient', 1), step('Gradient flows back', 2)];
		expect(rankSteps(steps, 'grad').map((s) => s.stepIndex)).toEqual([2, 1]);
		expect(rankSteps(steps, 'e', 1)).toHaveLength(1);
	});
});

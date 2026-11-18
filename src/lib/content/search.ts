import type { LessonMeta } from '../lessons/types.ts';

/**
 * Matching for quick jump. Pure functions: no DOM, no stores.
 *
 * Matching ignores case and diacritics ('Café' matches 'cafe'). A match at the start of a word ranks before
 * a match inside a word.
 */

/** Lower-cased text without accents, and for every character of it the index it came from in the original. */
function fold(text: string): { folded: string; origin: number[] } {
	let folded = '';
	const origin: number[] = [];
	let i = 0;
	for (const ch of text) {
		const f = ch.normalize('NFD').replace(/\p{M}/gu, '').toLowerCase();
		for (let k = 0; k < f.length; k++) origin.push(i);
		folded += f;
		i += ch.length;
	}
	return { folded, origin };
}

function foldQuery(query: string): string {
	return fold(query.trim()).folded;
}

/** 0 = the match is the start of the text, 1 = the start of a later word, 2 = inside a word, null = no match. */
type Kind = 0 | 1 | 2;

function find(folded: string, q: string): { index: number; kind: Kind } | null {
	if (!q) return null;
	let inside = -1;
	let from = 0;
	for (;;) {
		const at = folded.indexOf(q, from);
		if (at === -1) break;
		if (at === 0) return { index: 0, kind: 0 };
		if (!/[\p{L}\p{N}]/u.test(folded[at - 1])) return { index: at, kind: 1 };
		if (inside === -1) inside = at;
		from = at + 1;
	}
	return inside === -1 ? null : { index: inside, kind: 2 };
}

/**
 * Where the query matches in the text, as [start, end) offsets into the original text, or null. A match at
 * the start of a word wins over an earlier match inside a word. A blank query matches nothing.
 */
export function matchRange(text: string, query: string): [number, number] | null {
	const q = foldQuery(query);
	const { folded, origin } = fold(text);
	const hit = find(folded, q);
	if (!hit) return null;
	const start = origin[hit.index];
	const last = origin[hit.index + q.length - 1];
	// The end is one character after the last matched one, which may be more than one code unit.
	const end = last + (String.fromCodePoint(text.codePointAt(last) ?? 0).length || 1);
	return [start, end];
}

export interface Segment {
	text: string;
	mark: boolean;
}

/** The text cut into pieces, with the match (if any) marked. The pieces joined give back the text. */
export function highlight(text: string, query: string): Segment[] {
	const r = matchRange(text, query);
	if (!r) return [{ text, mark: false }];
	const [a, b] = r;
	return [
		{ text: text.slice(0, a), mark: false },
		{ text: text.slice(a, b), mark: true },
		{ text: text.slice(b), mark: false }
	].filter((s) => s.text !== '');
}

/**
 * Items that match, best first. `fields` gives the texts of an item in order of importance: a match in an
 * earlier field beats any match in a later one, and within a field a better kind of match wins. Ties keep
 * the order of `items`.
 */
function rank<T>(items: readonly T[], query: string, fields: (item: T) => string[], limit: number): T[] {
	const q = foldQuery(query);
	if (!q) return [];
	const scored: { item: T; score: number; order: number }[] = [];
	items.forEach((item, order) => {
		const texts = fields(item);
		for (let f = 0; f < texts.length; f++) {
			const hit = find(fold(texts[f]).folded, q);
			if (hit) {
				scored.push({ item, score: f * 3 + hit.kind, order });
				return;
			}
		}
	});
	scored.sort((a, b) => a.score - b.score || a.order - b.order);
	return scored.slice(0, limit).map((s) => s.item);
}

/** Ready lessons whose title (then summary) matches. */
export function rankLessons<T extends Pick<LessonMeta, 'title' | 'summary' | 'status'>>(
	lessons: readonly T[],
	query: string,
	limit = 6
): T[] {
	return rank(lessons.filter((l) => l.status === 'ready'), query, (l) => [l.title, l.summary], limit);
}

/** One step of one lesson, as the quick jump lists it. */
export interface StepEntry {
	track: string;
	lesson: string;
	lessonTitle: string;
	/** Zero-based. */
	stepIndex: number;
	title: string;
}

/** Steps whose title matches. */
export function rankSteps(steps: readonly StepEntry[], query: string, limit = 6): StepEntry[] {
	return rank(steps, query, (s) => [s.title], limit);
}

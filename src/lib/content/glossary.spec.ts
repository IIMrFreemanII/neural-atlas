import { describe, expect, it } from 'vitest';
import { allLessons } from './curriculum.ts';
import { findEntry, glossary, searchGlossary, termsForLesson } from './glossary.ts';

const keys = allLessons.map((l) => `${l.track}/${l.id}`);

describe('glossary data', () => {
	it('has between 70 and 110 entries', () => {
		expect(glossary.length).toBeGreaterThanOrEqual(70);
		expect(glossary.length).toBeLessThanOrEqual(110);
	});

	it('has unique kebab-case ids', () => {
		const ids = glossary.map((e) => e.id);
		expect(new Set(ids).size).toBe(ids.length);
		for (const id of ids) expect(id).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
	});

	it('is sorted by term, ignoring case', () => {
		const terms = glossary.map((e) => e.term.toLowerCase());
		expect(terms).toEqual([...terms].sort());
	});

	it('starts every term with a letter', () => {
		for (const e of glossary) expect(e.term).toMatch(/^[A-Za-z]/);
	});

	it('keeps definitions short and plain', () => {
		for (const e of glossary) {
			expect(e.definition.length, e.id).toBeLessThanOrEqual(240);
			expect(e.definition.endsWith('.'), e.id).toBe(true);
			expect(e.definition, e.id).not.toContain('—');
			expect(e.definition, e.id).not.toContain('!');
		}
	});

	it('only lists lessons that exist, and at least one', () => {
		for (const e of glossary) {
			expect(e.lessons.length, e.id).toBeGreaterThan(0);
			for (const k of e.lessons) expect(keys, `${e.id}: ${k}`).toContain(k);
			expect(new Set(e.lessons).size, e.id).toBe(e.lessons.length);
		}
	});

	it('gives every lesson at least 3 terms', () => {
		for (const k of keys) expect(termsForLesson(k).length, k).toBeGreaterThanOrEqual(3);
	});

	it('relates only to other existing entries', () => {
		for (const e of glossary) {
			for (const r of e.related ?? []) {
				expect(findEntry(r), `${e.id} -> ${r}`).toBeDefined();
				expect(r, e.id).not.toBe(e.id);
			}
		}
	});
});

describe('findEntry', () => {
	it('finds an entry by id', () => {
		expect(findEntry('gradient')?.term).toBe('gradient');
		expect(findEntry('relu')?.term).toBe('ReLU');
	});

	it('returns undefined for an unknown id', () => {
		expect(findEntry('no-such-term')).toBeUndefined();
	});
});

describe('termsForLesson', () => {
	it('lists introduced terms first, each group in glossary order', () => {
		const key = 'fundamentals/gradient-descent';
		const list = termsForLesson(key);
		const intro = list.filter((e) => e.lessons[0] === key);
		const rest = list.filter((e) => e.lessons[0] !== key);
		expect(intro.length).toBeGreaterThan(0);
		expect(rest.length).toBeGreaterThan(0);
		expect(list).toEqual([...intro, ...rest]);
		expect(intro).toEqual(glossary.filter((e) => intro.includes(e)));
		expect(rest).toEqual(glossary.filter((e) => rest.includes(e)));
		for (const e of list) expect(e.lessons).toContain(key);
	});

	it('returns nothing for an unknown lesson', () => {
		expect(termsForLesson('nope/nothing')).toEqual([]);
	});
});

describe('searchGlossary', () => {
	it('returns every entry for a blank query', () => {
		expect(searchGlossary('')).toEqual(glossary);
		expect(searchGlossary('   ')).toEqual(glossary);
	});

	it('is case-insensitive', () => {
		expect(searchGlossary('RELU')).toEqual(searchGlossary('relu'));
		expect(searchGlossary('relu')[0].id).toBe('relu');
	});

	it('matches aliases', () => {
		expect(searchGlossary('BPE')[0].id).toBe('byte-pair-encoding');
		expect(searchGlossary('cnn')[0].id).toBe('convolutional-neural-network');
	});

	it('ranks prefix matches before substring matches before definition matches', () => {
		const q = 'sampl';
		const result = searchGlossary(q);
		const names = (e: (typeof glossary)[number]) =>
			[e.term, ...(e.aliases ?? [])].map((s) => s.toLowerCase());
		const rank = (e: (typeof glossary)[number]) =>
			names(e).some((n) => n.startsWith(q)) ? 0 : names(e).some((n) => n.includes(q)) ? 1 : 2;
		const ranks = result.map(rank);
		expect(ranks).toEqual([...ranks].sort());
		expect(new Set(ranks)).toEqual(new Set([0, 1, 2]));
		expect(result[0].id).toBe('sampling');
	});

	it('keeps glossary order inside each group', () => {
		const result = searchGlossary('layer');
		const prefix = result.filter((e) => e.term.toLowerCase().startsWith('layer'));
		expect(prefix.map((e) => e.id)).toEqual(
			glossary.filter((e) => prefix.includes(e)).map((e) => e.id)
		);
	});

	it('never returns an entry twice', () => {
		for (const q of ['a', 'e', 'loss', 'token', 'step', 'the']) {
			const ids = searchGlossary(q).map((e) => e.id);
			expect(new Set(ids).size, q).toBe(ids.length);
		}
	});

	it('returns nothing for a nonsense query', () => {
		expect(searchGlossary('zzqxjv')).toEqual([]);
	});
});

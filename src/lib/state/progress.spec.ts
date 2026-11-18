import { describe, expect, it } from 'vitest';

// The module reads localStorage when it loads. Without a DOM that read fails and is caught, so the
// store starts empty, which is what these tests want.
describe('progress questions', () => {
	it('remembers solved questions, sorted, once', async () => {
		const { progress } = await import('./progress.svelte.ts');
		expect(progress.isSolved('a/b', 2)).toBe(false);
		progress.solve('a/b', 2);
		progress.solve('a/b', 0);
		progress.solve('a/b', 2);
		expect(progress.isSolved('a/b', 0)).toBe(true);
		expect(progress.isSolved('a/b', 2)).toBe(true);
		expect(progress.isSolved('a/b', 1)).toBe(false);
		expect(progress.isSolved('a/c', 0)).toBe(false);
	});
});

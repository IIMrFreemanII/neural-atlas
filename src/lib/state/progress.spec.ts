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

	it('counts the solved questions of a lesson', async () => {
		const { progress } = await import('./progress.svelte.ts');
		expect(progress.solvedCount('x/y', 3)).toBe(0);
		progress.solve('x/y', 1);
		progress.solve('x/y', 2);
		expect(progress.solvedCount('x/y', 3)).toBe(2);
		// A question that the lesson no longer has does not count.
		expect(progress.solvedCount('x/y', 2)).toBe(1);
		expect(progress.solvedCount('x/z', 3)).toBe(0);
	});
});

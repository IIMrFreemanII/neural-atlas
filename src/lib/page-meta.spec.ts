import { describe, expect, it } from 'vitest';
import { allLessons } from './content/curriculum.ts';
import { headTags, pageMeta } from './page-meta.ts';

const LESSON = '/learn/[track]/[lesson]';

describe('page meta', () => {
	it('names every ready lesson and uses its summary', async () => {
		for (const meta of allLessons.filter((l) => l.status === 'ready')) {
			const m = pageMeta(LESSON, { track: meta.track, lesson: meta.id });
			expect(m.title).toBe(`${meta.title} · Neural Atlas`);
			expect(m.description).toBe(meta.summary);
			// The title the app sets once it runs must agree with the one in the HTML.
			expect((await meta.load!()).default.title, `${meta.track}/${meta.id}`).toBe(meta.title);
		}
	});

	it('gives the other pages their own title', () => {
		expect(pageMeta('/', {}).title).toBe('Neural Atlas');
		expect(pageMeta('/learn', {}).title).toBe('Lessons · Neural Atlas');
		expect(pageMeta('/glossary', {}).title).toBe('Glossary · Neural Atlas');
		expect(pageMeta('/playground', {}).title).toBe('Train a network · Neural Atlas');
	});

	it('falls back to the home page for unknown addresses and lessons not written yet', () => {
		const home = pageMeta('/', {});
		expect(pageMeta(null, {})).toEqual(home);
		expect(pageMeta(LESSON, { track: 'nope', lesson: 'none' })).toEqual(home);
		const soon = allLessons.find((l) => l.status === 'soon');
		if (soon) expect(pageMeta(LESSON, { track: soon.track, lesson: soon.id })).toEqual(home);
	});

	it('gives a full page address and image only when it has them', () => {
		const meta = pageMeta('/', {});
		const urls = { page: 'https://example.org/site/', image: 'https://example.org/site/previews/home.jpg' };
		const tags = headTags(meta, urls);
		expect(tags).toContain('<meta property="og:image" content="https://example.org/site/previews/home.jpg" />');
		expect(tags).toContain('<meta property="og:url" content="https://example.org/site/" />');
		expect(tags).toContain('content="summary_large_image"');
		const bare = headTags(meta, null);
		expect(bare).not.toContain('og:image');
		expect(bare).not.toContain('og:url');
		expect(bare).toContain('<meta name="twitter:card" content="summary" />');
	});

	it('escapes text that would break the HTML', () => {
		const tags = headTags({ title: 'A <b> & "c"', description: 'x"><script>', image: '/x.jpg' }, null);
		expect(tags).toContain('<title>A &lt;b&gt; &amp; &quot;c&quot;</title>');
		expect(tags).toContain('content="x&quot;&gt;&lt;script&gt;"');
		expect(tags).not.toContain('<script>');
	});
});

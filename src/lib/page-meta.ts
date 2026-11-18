import { findLesson } from './content/curriculum.ts';

export interface PageMeta {
	/** The browser tab's title. */
	title: string;
	/** One or two sentences for search results and link previews. */
	description: string;
	/** The link-preview image, a path under static/ (1200x630, taken by scripts/capture-previews.ts). */
	image: string;
}

const SITE = 'Neural Atlas';

const HOME: PageMeta = {
	title: SITE,
	description:
		'Interactive visuals that take neural networks, classic ML and language models apart, one idea at a time.',
	image: '/previews/home.jpg'
};

const PAGES: Record<string, PageMeta> = {
	'/': HOME,
	'/learn': {
		title: `Lessons · ${SITE}`,
		description:
			'Four tracks of interactive lessons: neural net fundamentals, classic ML, LLMs and transformers, vision and generative models.',
		image: HOME.image
	},
	'/playground': {
		title: `Playground · ${SITE}`,
		description: 'Free play with a small network: pick the data, shape the network and watch it learn.',
		image: HOME.image
	},
	'/glossary': {
		title: `Glossary · ${SITE}`,
		description: 'Short definitions for the terms the lessons use, with links to where each one is shown.',
		image: HOME.image
	}
};

function lesson(track: string, id: string): PageMeta | null {
	const found = findLesson(track, id);
	if (!found || found.meta.status !== 'ready') return null;
	return {
		title: `${found.meta.title} · ${SITE}`,
		description: found.meta.summary,
		image: `/previews/${track}/${id}.jpg`
	};
}

/**
 * The title, description and preview image of the page at a route, as SvelteKit names it (`event.route.id`, null when no
 * route matches). The app sets the tab title itself once it runs; this is for readers that do not run it:
 * link previews and search engines.
 */
export function pageMeta(routeId: string | null, params: Record<string, string>): PageMeta {
	if (routeId === '/learn/[track]/[lesson]') return lesson(params.track, params.lesson) ?? HOME;
	return (routeId && PAGES[routeId]) || HOME;
}

function escape(text: string): string {
	return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/**
 * The tags `app.html` gets in place of `%neural-atlas.head%`. `urls` are the page's and its image's full
 * addresses; link previews need full addresses, so without them the page gets no image.
 */
export function headTags(meta: PageMeta, urls: { page: string; image: string } | null): string {
	const title = escape(meta.title);
	const description = escape(meta.description);
	const absolute = urls
		? [
				`<meta property="og:url" content="${escape(urls.page)}" />`,
				`<meta property="og:image" content="${escape(urls.image)}" />`,
				`<meta property="og:image:width" content="1200" />`,
				`<meta property="og:image:height" content="630" />`,
				`<meta property="og:image:alt" content="${title}" />`
			]
		: [];
	return [
		`<title>${title}</title>`,
		`<meta name="description" content="${description}" />`,
		`<meta property="og:site_name" content="${SITE}" />`,
		`<meta property="og:type" content="website" />`,
		`<meta property="og:title" content="${title}" />`,
		`<meta property="og:description" content="${description}" />`,
		...absolute,
		`<meta name="twitter:card" content="${urls ? 'summary_large_image' : 'summary'}" />`
	].join('\n\t\t');
}

import { findLesson } from './content/curriculum.ts';

export interface PageMeta {
	/** The browser tab's title. */
	title: string;
	/** One or two sentences for search results and link previews. */
	description: string;
}

const SITE = 'Neural Atlas';

const HOME: PageMeta = {
	title: SITE,
	description:
		'Interactive visuals that take neural networks, classic ML and language models apart, one idea at a time.'
};

const PAGES: Record<string, PageMeta> = {
	'/': HOME,
	'/learn': {
		title: `Lessons · ${SITE}`,
		description:
			'Four tracks of interactive lessons: neural net fundamentals, classic ML, LLMs and transformers, vision and generative models.'
	},
	'/glossary': {
		title: `Glossary · ${SITE}`,
		description: 'Short definitions for the terms the lessons use, with links to where each one is shown.'
	}
};

function lesson(track: string, id: string): PageMeta | null {
	const found = findLesson(track, id);
	if (!found || found.meta.status !== 'ready') return null;
	return { title: `${found.meta.title} · ${SITE}`, description: found.meta.summary };
}

/**
 * The title and description of the page at a route, as SvelteKit names it (`event.route.id`, null when no
 * route matches). The app sets the tab title itself once it runs; this is for readers that do not run it:
 * link previews and search engines.
 */
export function pageMeta(routeId: string | null, params: Record<string, string>): PageMeta {
	if (routeId === '/learn/[track]/[lesson]') return lesson(params.track, params.lesson) ?? HOME;
	// The playground sends you to the training lesson.
	if (routeId === '/playground') return lesson('fundamentals', 'train-a-network') ?? HOME;
	return (routeId && PAGES[routeId]) || HOME;
}

function escape(text: string): string {
	return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/** The tags `app.html` gets in place of `%neural-atlas.head%`. */
export function headTags(meta: PageMeta): string {
	const title = escape(meta.title);
	const description = escape(meta.description);
	return [
		`<title>${title}</title>`,
		`<meta name="description" content="${description}" />`,
		`<meta property="og:site_name" content="${SITE}" />`,
		`<meta property="og:type" content="website" />`,
		`<meta property="og:title" content="${title}" />`,
		`<meta property="og:description" content="${description}" />`,
		`<meta name="twitter:card" content="summary" />`
	].join('\n\t\t');
}

import type { Handle } from '@sveltejs/kit/hooks';
import { headTags, pageMeta } from '#lib/page-meta.ts';

// Runs when the build prerenders each page (and in `vite dev`). The pages are empty shells, because the
// app renders only in the browser, so this is what writes each page's own title and description into its
// HTML for link previews and search engines.
export const handle: Handle = ({ event, resolve }) =>
	resolve(event, {
		transformPageChunk: ({ html }) =>
			html.replace('%neural-atlas.head%', headTags(pageMeta(event.route.id, event.params)))
	});

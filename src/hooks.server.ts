import type { Handle } from '@sveltejs/kit/hooks';
import { asset } from '$app/paths';
import { headTags, pageMeta } from '#lib/page-meta.ts';

/** What `url.origin` is while prerendering when the build is not told the site's real origin. */
const UNKNOWN_ORIGIN = 'http://sveltekit-prerender';

// Runs when the build prerenders each page (and in `vite dev`). The pages are empty shells, because the
// app renders only in the browser, so this is what writes each page's own title, description and preview
// image into its HTML for link previews and search engines.
export const handle: Handle = ({ event, resolve }) => {
	const meta = pageMeta(event.route.id, event.params);
	// Link previews need full addresses. `npm run deploy` gives the build the site's origin (SITE_ORIGIN);
	// a build without it leaves the image out rather than point it at a made-up host. The 404 page (no
	// route) has no address of its own.
	const urls =
		event.url.origin === UNKNOWN_ORIGIN || event.route.id === null
			? null
			: {
					page: event.url.href,
					// `asset()` may give a path relative to the page, so resolve it against the page's address.
					image: new URL(asset(meta.image as Parameters<typeof asset>[0]), event.url).href
				};
	return resolve(event, {
		transformPageChunk: ({ html }) => html.replace('%neural-atlas.head%', headTags(meta, urls))
	});
};

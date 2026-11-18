import { error } from '@sveltejs/kit';

// Engine test pages: available while developing, not in the built site.
export function load() {
	if (!import.meta.env.DEV) error(404, 'Not found');
}

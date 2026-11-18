import { redirect } from '@sveltejs/kit';
import { resolve } from '$app/paths';

// The playground is the free-play lesson at the end of the fundamentals track.
export function load() {
	redirect(307, resolve('/learn/[track]/[lesson]', { track: 'fundamentals', lesson: 'train-a-network' }));
}

import { redirect } from '@sveltejs/kit';

// The playground is the free-play lesson at the end of the fundamentals track.
export function load() {
	redirect(307, '/learn/fundamentals/train-a-network');
}

import { error } from '@sveltejs/kit';
import { allLessons, findLesson, nextReadyLesson } from '#lib/content/curriculum.ts';

/** The lessons the build writes a page for. Nothing links to them in the empty prerendered pages. */
export function entries() {
	return allLessons.filter((l) => l.status === 'ready').map((l) => ({ track: l.track, lesson: l.id }));
}

export async function load({ params }: { params: { track: string; lesson: string } }) {
	const found = findLesson(params.track, params.lesson);
	if (!found || found.meta.status === 'soon') error(404, 'Lesson not found');
	const { track, meta } = found;
	const mod = await meta.load!();
	return { lesson: mod.default, meta, track, next: nextReadyLesson(meta.track, meta.id) };
}

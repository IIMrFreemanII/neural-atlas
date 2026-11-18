import { error } from '@sveltejs/kit';
import { findLesson, nextReadyLesson } from '#lib/content/curriculum.ts';

export async function load({ params }: { params: { track: string; lesson: string } }) {
	const found = findLesson(params.track, params.lesson);
	if (!found || found.meta.status === 'soon') error(404, 'Lesson not found');
	const { track, meta } = found;
	const mod = await meta.load!();
	return { lesson: mod.default, meta, track, next: nextReadyLesson(meta.track, meta.id) };
}

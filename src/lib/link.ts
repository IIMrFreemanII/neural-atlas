import { resolve } from '$app/paths';
import type { LessonMeta } from './lessons/types.ts';

/**
 * Address of a lesson's page for a link. Unlike `lessonHref` it includes the base path the site is served
 * from (for example `/neural-atlas` on GitHub Pages), so use this one for every `href`.
 */
export function lessonLink(meta: Pick<LessonMeta, 'track' | 'id'>): string {
	return resolve('/learn/[track]/[lesson]', { track: meta.track, lesson: meta.id });
}

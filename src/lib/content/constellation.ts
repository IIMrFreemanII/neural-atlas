import type { TrackId } from '../lessons/types.ts';
import { allLessons, lessonKey } from './curriculum.ts';

/**
 * Layout of the home page's constellation map, in a fixed design space of MAP_W x MAP_H units with
 * the origin at the top left and y pointing down. The HTML overlay places links by percentage and the
 * GPU scene draws in the same space (flipped vertically), so both always line up.
 */
export const MAP_W = 1000;
export const MAP_H = 520;

const POSITIONS: Record<string, [x: number, y: number]> = {
	// A zigzag down the left edge.
	'classic-ml/linear-regression': [90, 120],
	'classic-ml/decision-boundaries': [175, 195],
	'classic-ml/k-means': [95, 290],
	'classic-ml/overfitting': [180, 385],
	// A dipper in the middle: three handle stars, then the four corners of the bowl.
	'fundamentals/neuron': [300, 270],
	'fundamentals/activations': [365, 205],
	'fundamentals/forward-pass': [450, 225],
	'fundamentals/loss-landscape': [525, 270],
	'fundamentals/gradient-descent': [530, 350],
	'fundamentals/backpropagation': [635, 360],
	'fundamentals/train-a-network': [645, 275],
	// A zigzag in the upper right, ending toward the middle.
	'llm/tokenization': [930, 75],
	'llm/embeddings': [850, 120],
	'llm/attention': [915, 195],
	'llm/transformer-block': [790, 210],
	'llm/sampling': [745, 100],
	// A shallow arc along the bottom.
	'vision/convolution': [390, 445],
	'vision/cnn-features': [760, 395],
	'vision/diffusion': [880, 455]
};

/** Where each track's name is written (center of the text). */
export const TRACK_LABELS: Record<TrackId, [x: number, y: number]> = {
	'classic-ml': [135, 62],
	fundamentals: [480, 158],
	llm: [838, 36],
	vision: [640, 488]
};

export interface Star {
	/** Index into allLessons, and into `stars`. */
	index: number;
	/** Lesson key, '<track>/<id>'. */
	key: string;
	track: TrackId;
	x: number;
	y: number;
}

export interface StarLink {
	/** Star index of the prerequisite. */
	from: number;
	/** Star index of the lesson that builds on it. */
	to: number;
	/** True when the two lessons are in different tracks. */
	cross: boolean;
}

export const stars: Star[] = allLessons.map((l, index) => {
	const key = lessonKey(l.track, l.id);
	const p = POSITIONS[key];
	if (!p) throw new Error(`No constellation position for ${key}`);
	return { index, key, track: l.track, x: p[0], y: p[1] };
});

const indexOf = new Map(stars.map((s) => [s.key, s.index]));

export const links: StarLink[] = allLessons.flatMap((l, to) =>
	l.prerequisites.map((key) => {
		const from = indexOf.get(key);
		if (from === undefined) throw new Error(`Unknown prerequisite ${key}`);
		return { from, to, cross: stars[from].track !== l.track };
	})
);

/** Bit i of the result is set when lesson i (index into allLessons) is complete. */
export function doneMask(isComplete: (key: string) => boolean): number {
	let mask = 0;
	for (const s of stars) if (isComplete(s.key)) mask |= 1 << s.index;
	return mask;
}

/**
 * The lesson to suggest next: the first one, in curriculum order, that is not complete and whose
 * prerequisites all are. -1 when everything is complete.
 */
export function nextUp(mask: number): number {
	for (const s of stars) {
		if (mask & (1 << s.index)) continue;
		const ready = links.every((l) => l.to !== s.index || (mask & (1 << l.from)) !== 0);
		if (ready) return s.index;
	}
	return -1;
}

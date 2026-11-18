import type { TrackId } from '../lessons/types.ts';
import { allLessons, lessonKey } from './curriculum.ts';

/**
 * Layout of the home page's constellation map, in a fixed design space of MAP_W x MAP_H units with
 * the origin at the top left and y pointing down. The SVG map draws in this space, and the hover card
 * is placed by percentage of it.
 */
export const MAP_W = 884;
export const MAP_H = 732;

const POSITIONS: Record<string, [x: number, y: number]> = {
	// A zigzag down the left edge.
	'classic-ml/linear-regression': [70, 340],
	'classic-ml/decision-boundaries': [140, 400],
	'classic-ml/k-means': [80, 470],
	'classic-ml/overfitting': [170, 530],
	'classic-ml/decision-trees': [280, 580],
	// The neural net track in the middle: a loop that ends on the training lesson.
	'fundamentals/neuron': [340, 300],
	'fundamentals/activations': [400, 250],
	'fundamentals/forward-pass': [480, 300],
	'fundamentals/loss-landscape': [440, 380],
	'fundamentals/gradient-descent': [360, 430],
	'fundamentals/backpropagation': [450, 480],
	'fundamentals/train-a-network': [560, 450],
	// A zigzag in the upper right.
	'llm/tokenization': [560, 190],
	'llm/embeddings': [630, 240],
	'llm/attention': [730, 200],
	'llm/transformer-block': [800, 280],
	'llm/sampling': [720, 345],
	'llm/training': [828, 384],
	// A shallow arc along the bottom right.
	'vision/convolution': [650, 500],
	'vision/cnn-features': [700, 570],
	'vision/vit': [820, 470],
	'vision/autoencoder': [840, 550],
	'vision/diffusion': [790, 600]
};

/**
 * Where each track's name is written, and which way the text runs from that point. Each sits in
 * open paper, clear of dots, links and lesson names.
 */
export const TRACK_LABELS: Record<TrackId, [x: number, y: number, anchor: 'start' | 'end']> = {
	'classic-ml': [48, 268, 'start'],
	fundamentals: [250, 215, 'start'],
	llm: [560, 150, 'start'],
	vision: [836, 672, 'end']
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

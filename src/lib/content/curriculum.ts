import type { LessonMeta, Track, TrackId } from '../lessons/types.ts';

type Seed = [id: string, title: string, summary: string, minutes: number];

/**
 * Prerequisites that cross tracks, by lesson key. Within a track every lesson also builds on the one
 * before it. Each of these is an idea the later lesson uses without re-teaching it.
 */
const CROSS_TRACK: Record<string, string[]> = {
	// A kernel is a weighted sum, the same operation as a neuron.
	'vision/convolution': ['fundamentals/neuron'],
	// The feature maps come from a network whose kernels were learned by training.
	'vision/cnn-features': ['fundamentals/train-a-network'],
	// A block is a stack of layers that vectors flow through.
	'llm/transformer-block': ['fundamentals/forward-pass'],
	// The same training loop as the small network there, on a language model.
	'llm/training': ['fundamentals/train-a-network']
};

/**
 * How many questions a ready lesson has. Written here, not read from the lesson, so that the lists and the
 * map can show a count without loading any lesson. `questions.spec.ts` checks it against every lesson.
 */
const QUESTIONS = 3;
/** Lessons whose number of questions is not QUESTIONS, by lesson key. */
const QUESTION_COUNT: Record<string, number> = {};

function build(track: TrackId, seeds: Seed[], ready: Record<string, LessonMeta['load']>): LessonMeta[] {
	return seeds.map(([id, title, summary, minutes], i) => {
		const load = ready[id];
		const meta: LessonMeta = {
			id,
			track,
			title,
			summary,
			minutes,
			status: load ? 'ready' : 'soon',
			prerequisites: [
				...(i > 0 ? [`${track}/${seeds[i - 1][0]}`] : []),
				...(CROSS_TRACK[`${track}/${id}`] ?? [])
			],
			questions: load ? (QUESTION_COUNT[`${track}/${id}`] ?? QUESTIONS) : 0
		};
		if (load) meta.load = load;
		return meta;
	});
}

export const tracks: Track[] = [
	{
		id: 'fundamentals',
		title: 'Neural net fundamentals',
		blurb: 'From a single neuron to a trained network, built up one piece at a time.',
		tone: 'data',
		lessons: build(
			'fundamentals',
			[
				['neuron', 'The neuron', 'A neuron weighs its inputs, adds a bias and squashes the result into an output.', 6],
				['activations', 'Activation functions', 'Nonlinear functions let a stack of neurons draw curved boundaries.', 6],
				['forward-pass', 'Forward pass', 'Follow a single input as it flows through the layers of a network.', 7],
				['loss-landscape', 'Loss landscape', 'Loss is a surface over parameter space, and training is a walk across it.', 7],
				['gradient-descent', 'Gradient descent', 'Step downhill along the negative gradient to lower the loss.', 8],
				['backpropagation', 'Backpropagation', 'The chain rule hands every weight its share of the blame.', 10],
				['train-a-network', 'Train a network', 'Put every piece together and watch a small network learn a pattern.', 9]
			],
			{
				neuron: () => import('../lessons/fundamentals/neuron/lesson.ts'),
				activations: () => import('../lessons/fundamentals/activations/lesson.ts'),
				'forward-pass': () => import('../lessons/fundamentals/forward-pass/lesson.ts'),
				'loss-landscape': () => import('../lessons/fundamentals/loss-landscape/lesson.ts'),
				'gradient-descent': () => import('../lessons/fundamentals/gradient-descent/lesson.ts'),
				backpropagation: () => import('../lessons/fundamentals/backpropagation/lesson.ts'),
				'train-a-network': () => import('../lessons/fundamentals/train-a-network/lesson.ts')
			}
		)
	},
	{
		id: 'classic-ml',
		title: 'Classic machine learning',
		blurb: 'The workhorse models that still solve most real problems.',
		tone: 'ok',
		lessons: build(
			'classic-ml',
			[
				['linear-regression', 'Linear regression', 'Fit a line to noisy points by minimizing squared error.', 5],
				['decision-boundaries', 'Decision boundaries', 'See how different classifiers carve a plane into regions.', 6],
				['k-means', 'K-means clustering', 'Group unlabeled points by alternating assignment and averaging.', 5],
				['overfitting', 'Overfitting and underfitting', 'Too simple a model misses the pattern, too flexible a model memorizes noise.', 7]
			],
			{
				'linear-regression': () => import('../lessons/classic-ml/linear-regression/lesson.ts'),
				'decision-boundaries': () => import('../lessons/classic-ml/decision-boundaries/lesson.ts'),
				'k-means': () => import('../lessons/classic-ml/k-means/lesson.ts'),
				overfitting: () => import('../lessons/classic-ml/overfitting/lesson.ts')
			}
		)
	},
	{
		id: 'llm',
		title: 'LLMs and transformers',
		blurb: 'How language models turn text into tokens, vectors and predictions.',
		tone: 'param',
		lessons: build(
			'llm',
			[
				['tokenization', 'Tokenization', 'Text is split into subword pieces before a model ever sees it.', 5],
				['embeddings', 'Embedding space', 'Tokens become points in a space where distance carries meaning.', 7],
				['attention', 'Self-attention', 'Every token looks at every other token and decides what matters.', 10],
				['transformer-block', 'Inside a transformer block', 'Attention, a feed-forward layer and residual paths make one block.', 9],
				['sampling', 'Next-token sampling', 'Temperature and top-p control how a model picks the next word.', 6],
				['training', 'Watching a model learn', 'Follow the story model from random weights to readable stories, one training step at a time.', 8]
			],
			{
				tokenization: () => import('../lessons/llm/tokenization/lesson.ts'),
				embeddings: () => import('../lessons/llm/embeddings/lesson.ts'),
				attention: () => import('../lessons/llm/attention/lesson.ts'),
				'transformer-block': () => import('../lessons/llm/transformer-block/lesson.ts'),
				sampling: () => import('../lessons/llm/sampling/lesson.ts'),
				training: () => import('../lessons/llm/training/lesson.ts')
			}
		)
	},
	{
		id: 'vision',
		title: 'Vision and generative',
		blurb: 'Filters that see edges, and models that paint images out of noise.',
		tone: 'grad',
		lessons: build(
			'vision',
			[
				['convolution', 'Convolution', 'Slide a small kernel across an image to detect local patterns.', 6],
				['cnn-features', 'CNN feature maps', 'Stacked convolutions build from edges to textures to objects.', 8],
				['diffusion', 'Diffusion', 'Generate an image by learning to remove noise step by step.', 9]
			],
			{
				convolution: () => import('../lessons/vision/convolution/lesson.ts'),
				'cnn-features': () => import('../lessons/vision/cnn-features/lesson.ts'),
				diffusion: () => import('../lessons/vision/diffusion/lesson.ts')
			}
		)
	}
];

export function lessonKey(track: TrackId, id: string): string {
	return `${track}/${id}`;
}

export function findLesson(track: string, id: string): { track: Track; meta: LessonMeta } | null {
	const t = tracks.find((x) => x.id === track);
	const meta = t?.lessons.find((l) => l.id === id);
	return t && meta ? { track: t, meta } : null;
}

export function nextReadyLesson(track: TrackId, id: string): LessonMeta | null {
	const all = tracks.flatMap((t) => t.lessons);
	const i = all.findIndex((l) => l.track === track && l.id === id);
	if (i === -1) return null;
	return all.slice(i + 1).find((l) => l.status === 'ready') ?? null;
}

/** Every lesson of every track, in curriculum order. */
export const allLessons: LessonMeta[] = tracks.flatMap((t) => t.lessons);

/** How many questions these lessons have together. */
export function questionTotal(lessons: LessonMeta[]): number {
	return lessons.reduce((sum, l) => sum + l.questions, 0);
}

export function lessonHref(meta: LessonMeta): string {
	return `/learn/${meta.track}/${meta.id}`;
}

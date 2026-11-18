/**
 * Shape of the training log shown by the "Watching a model learn" lesson. Types only.
 * The data itself (log.ts) is written by scripts/build-training-log.ts from the weights that
 * scripts/train-tinystories.ts saved while it trained the model of the language lessons.
 */

/** What the model was like at one moment of training. */
export interface Checkpoint {
	/** Training steps done when these weights were saved. 0 = before any training. */
	step: number;
	/** Mean loss of the training batches of up to 10 steps around this step. For step 0: the loss of the first batch, before any update. */
	trainLoss: number;
	/** Loss on stories kept out of training: the mean over 400 fixed windows of 32 tokens. */
	validLoss: number;
	/**
	 * greedy[p]: what the model writes after TrainLog.prompts[p], always taking its most likely token.
	 * Token texts, the prompt not included. Ends early, with '<end>' as the last entry, if the model ends the story.
	 */
	greedy: string[][];
	/** sampled[p]: the same, but each token drawn from the model's own probabilities (temperature 1), with the same random numbers at every checkpoint. */
	sampled: string[][];
	/** probe[i]: the probability the model gave to the true next word TrainLog.probe[i + 1] after reading probe[0..i]. */
	probe: number[];
	/** guess[i]: the token the model found most likely at that position. */
	guess: string[];
	/** guessP[i]: the probability it gave to guess[i]. */
	guessP: number[];
	/** Mean of -ln(probe[i]): the loss on the probe sentence alone. */
	probeLoss: number;
}

export interface TrainLog {
	/** Story starts, as text (tokens separated by spaces). */
	prompts: string[];
	/** Tokens of the sentence the model is asked to predict word by word. */
	probe: string[];
	/** Size of the vocabulary. */
	vocab: number;
	/** ln(vocab): the loss of giving every token the same probability. */
	uniformLoss: number;
	/** Sequences per training step. */
	batch: number;
	/** Tokens per sequence. */
	ctx: number;
	/** Steps of the whole run. */
	steps: number;
	/** The training loss curve: curveLoss[i] is the mean batch loss of the steps close to step curveStep[i]. Steps ascend, starting at 1. */
	curveStep: number[];
	curveLoss: number[];
	/** In training order. The first has step 0. */
	checkpoints: Checkpoint[];
}

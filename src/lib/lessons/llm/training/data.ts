import { TRAIN_LOG } from './log.ts';

export const LOG = TRAIN_LOG;
/** Index of the last checkpoint. */
export const LAST = LOG.checkpoints.length - 1;

/** Steps that make the first doubling on the chart axis. */
const UNIT = 25;
const SPAN = Math.log2(1 + LOG.steps / UNIT);

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

/**
 * Horizontal chart position of a training step: 0 at step 0, 1 at the last step of the whole run
 * (LOG.steps). log2(1 + step / 25), normalised, so each doubling of the steps takes about the same width.
 */
export function axisX(step: number): number {
	return Math.log2(1 + Math.max(0, step) / UNIT) / SPAN;
}

/** Inverse of axisX. */
export function stepAtX(x: number): number {
	return UNIT * (Math.pow(2, x * SPAN) - 1);
}

/** The training step at a fractional checkpoint position (0 = first checkpoint, LAST = last), interpolated in axisX. Clamped. */
export function stepOfPosition(position: number): number {
	const p = clamp(position, 0, LAST);
	const i = Math.min(LAST, Math.floor(p));
	if (i >= LAST) return LOG.checkpoints[LAST].step;
	const f = p - i;
	if (f <= 0) return LOG.checkpoints[i].step;
	const a = axisX(LOG.checkpoints[i].step);
	const b = axisX(LOG.checkpoints[i + 1].step);
	return stepAtX(a + (b - a) * f);
}

/**
 * The training loss at any step from 0 to the last curve step: the step-0 checkpoint's trainLoss at 0, then
 * linear interpolation (in axisX) through the curve points. Clamped.
 */
export function curveLossAt(step: number): number {
	const xs = [0, ...LOG.curveStep.map(axisX)];
	const ys = [LOG.checkpoints[0].trainLoss, ...LOG.curveLoss];
	const x = clamp(axisX(step), 0, xs[xs.length - 1]);
	let lo = 0;
	let hi = xs.length - 1;
	while (hi - lo > 1) {
		const mid = (lo + hi) >> 1;
		if (xs[mid] <= x) lo = mid;
		else hi = mid;
	}
	if (hi === lo || xs[hi] === xs[lo]) return ys[lo];
	return ys[lo] + ((ys[hi] - ys[lo]) * (x - xs[lo])) / (xs[hi] - xs[lo]);
}

/** Index of the checkpoint whose step is nearest to `step`. */
export function nearestCheckpoint(step: number): number {
	let best = 0;
	for (let i = 1; i < LOG.checkpoints.length; i++) {
		if (Math.abs(LOG.checkpoints[i].step - step) < Math.abs(LOG.checkpoints[best].step - step)) best = i;
	}
	return best;
}

/** How a token is written on screen: '<end>' reads 'end of story', everything else is its text. */
export function tokenLabel(text: string): string {
	return text === '<end>' ? 'end of story' : text;
}

/** Math.round(Math.exp(loss)): a model with this loss is as unsure as someone choosing evenly among this many tokens. */
export function choices(loss: number): number {
	return Math.round(Math.exp(loss));
}

/** A step count with thousands separators: 12000 -> '12,000'. */
export function formatStep(step: number): string {
	return Math.round(step).toLocaleString('en-US');
}

/** A probability as a label: '<1%' below 0.005, otherwise the rounded percent ('4%', '100%'). */
export function percent(p: number): string {
	if (p < 0.005) return '<1%';
	return `${Math.round(p * 100)}%`;
}

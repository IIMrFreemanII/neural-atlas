import { describe, expect, it } from 'vitest';
import {
	LAST,
	LOG,
	axisX,
	choices,
	curveLossAt,
	formatStep,
	nearestCheckpoint,
	percent,
	stepAtX,
	stepOfPosition,
	tokenLabel
} from './data.ts';

describe('chart axis', () => {
	it('maps step 0 to 0 and the end of the run to 1', () => {
		expect(axisX(0)).toBe(0);
		expect(axisX(LOG.steps)).toBeCloseTo(1, 12);
	});

	it('only goes up', () => {
		let prev = -1;
		for (const s of [0, 1, 5, 25, 100, 400, 1600, 6400, LOG.steps]) {
			expect(axisX(s)).toBeGreaterThan(prev);
			prev = axisX(s);
		}
	});

	it('is undone by stepAtX', () => {
		for (const s of [0, 1, 25, 333, 1600, 12000]) expect(stepAtX(axisX(s))).toBeCloseTo(s, 6);
	});
});

describe('position along the checkpoints', () => {
	it('is the checkpoint step at every whole position', () => {
		for (let k = 0; k <= LAST; k++) expect(stepOfPosition(k)).toBeCloseTo(LOG.checkpoints[k].step, 6);
	});

	it('lies between two checkpoints in between, and is clamped outside', () => {
		for (let k = 0; k < LAST; k++) {
			const s = stepOfPosition(k + 0.5);
			expect(s).toBeGreaterThan(LOG.checkpoints[k].step);
			expect(s).toBeLessThan(LOG.checkpoints[k + 1].step);
		}
		expect(stepOfPosition(-3)).toBe(LOG.checkpoints[0].step);
		expect(stepOfPosition(LAST + 5)).toBe(LOG.checkpoints[LAST].step);
	});
});

describe('curveLossAt', () => {
	it('starts at the loss of the untrained model', () => {
		expect(curveLossAt(0)).toBe(LOG.checkpoints[0].trainLoss);
	});

	it('hits the curve at its points', () => {
		LOG.curveStep.forEach((s, i) => expect(Math.abs(curveLossAt(s) - LOG.curveLoss[i])).toBeLessThan(1e-9));
	});

	it('stays between its neighbours and is clamped past the end', () => {
		const a = LOG.curveStep[10];
		const b = LOG.curveStep[11];
		const mid = curveLossAt((a + b) / 2);
		expect(mid).toBeLessThanOrEqual(Math.max(LOG.curveLoss[10], LOG.curveLoss[11]) + 1e-9);
		expect(mid).toBeGreaterThanOrEqual(Math.min(LOG.curveLoss[10], LOG.curveLoss[11]) - 1e-9);
		const lastLoss = LOG.curveLoss[LOG.curveLoss.length - 1];
		expect(curveLossAt(LOG.steps * 2)).toBe(lastLoss);
	});
});

describe('nearestCheckpoint', () => {
	it('finds the checkpoint at or nearest to a step', () => {
		LOG.checkpoints.forEach((c, i) => expect(nearestCheckpoint(c.step)).toBe(i));
		expect(nearestCheckpoint(-50)).toBe(0);
		expect(nearestCheckpoint(LOG.steps * 10)).toBe(LAST);
	});
});

describe('labels', () => {
	it('writes the end token in words', () => {
		expect(tokenLabel('<end>')).toBe('end of story');
		expect(tokenLabel('park')).toBe('park');
	});

	it('turns a loss into a number of choices', () => {
		expect(choices(Math.log(12))).toBe(12);
		expect(choices(0)).toBe(1);
	});

	it('formats step counts', () => {
		expect(formatStep(0)).toBe('0');
		expect(formatStep(800)).toBe('800');
		expect(formatStep(12000)).toBe('12,000');
		expect(formatStep(1234.6)).toBe('1,235');
	});

	it('formats probabilities', () => {
		expect(percent(0)).toBe('<1%');
		expect(percent(0.0049)).toBe('<1%');
		expect(percent(0.005)).toBe('1%');
		expect(percent(0.04)).toBe('4%');
		expect(percent(0.999)).toBe('100%');
		expect(percent(1)).toBe('100%');
	});
});

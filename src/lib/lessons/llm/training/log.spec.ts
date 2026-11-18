import { describe, expect, it } from 'vitest';
import { LM_TRAINING, LM_VOCAB } from '../shared/lm-meta.ts';
import { PROMPTS } from '../sampling/data.ts';
import { TRAIN_LOG as log } from './log.ts';

const vocab = new Set(LM_VOCAB);
const STORY_TOKENS = 28;

describe('training log', () => {
	it('has checkpoints that start at step 0 and ascend', () => {
		expect(log.checkpoints.length).toBeGreaterThanOrEqual(2);
		expect(log.checkpoints[0].step).toBe(0);
		for (let i = 1; i < log.checkpoints.length; i++) {
			expect(log.checkpoints[i].step).toBeGreaterThan(log.checkpoints[i - 1].step);
		}
		expect(log.checkpoints[log.checkpoints.length - 1].step).toBeLessThanOrEqual(log.steps);
	});

	it('has a probe sentence without unknown words', () => {
		expect(log.probe.length).toBeGreaterThanOrEqual(8);
		expect(log.probe).not.toContain('<unk>');
		for (const t of log.probe) expect(vocab.has(t)).toBe(true);
	});

	it('has consistent probe readings at every checkpoint', () => {
		for (const c of log.checkpoints) {
			const n = log.probe.length - 1;
			expect(c.probe.length).toBe(n);
			expect(c.guess.length).toBe(n);
			expect(c.guessP.length).toBe(n);
			for (let i = 0; i < n; i++) {
				expect(c.probe[i]).toBeGreaterThanOrEqual(0);
				expect(c.probe[i]).toBeLessThanOrEqual(1);
				expect(c.guessP[i]).toBeGreaterThanOrEqual(0);
				expect(c.guessP[i]).toBeLessThanOrEqual(1);
				expect(vocab.has(c.guess[i])).toBe(true);
				expect(c.guess[i]).not.toBe('<unk>');
				if (c.guess[i] === log.probe[i + 1]) expect(Math.abs(c.guessP[i] - c.probe[i])).toBeLessThanOrEqual(1e-4);
				else expect(c.guessP[i]).toBeGreaterThanOrEqual(c.probe[i] - 1e-4);
			}
		}
	});

	it('stores a probe loss that matches the probe probabilities', () => {
		for (const c of log.checkpoints) {
			const used = c.probe.filter((p) => p > 0);
			if (used.length < c.probe.length) continue;
			const mean = c.probe.reduce((a, p) => a - Math.log(p), 0) / c.probe.length;
			expect(Math.abs(c.probeLoss - mean)).toBeLessThan(0.02);
		}
	});

	it('has one short story per prompt, from known tokens', () => {
		expect(log.prompts).toEqual(PROMPTS);
		for (const c of log.checkpoints) {
			for (const stories of [c.greedy, c.sampled]) {
				expect(stories.length).toBe(log.prompts.length);
				for (const s of stories) {
					expect(s.length).toBeGreaterThanOrEqual(1);
					expect(s.length).toBeLessThanOrEqual(STORY_TOKENS);
					s.forEach((t, i) => {
						expect(vocab.has(t)).toBe(true);
						expect(t).not.toBe('<unk>');
						if (t === '<end>') expect(i).toBe(s.length - 1);
					});
				}
			}
		}
	});

	it('has a loss curve with ascending steps and finite losses', () => {
		expect(log.curveStep.length).toBe(log.curveLoss.length);
		expect(log.curveStep[0]).toBe(1);
		for (let i = 1; i < log.curveStep.length; i++) expect(log.curveStep[i]).toBeGreaterThan(log.curveStep[i - 1]);
		for (const l of log.curveLoss) {
			expect(Number.isFinite(l)).toBe(true);
			expect(l).toBeGreaterThan(0);
		}
	});

	it('starts near the loss of guessing', () => {
		expect(log.vocab).toBe(LM_VOCAB.length);
		expect(Math.abs(log.uniformLoss - Math.log(log.vocab))).toBeLessThan(1e-4);
		const first = log.checkpoints[0];
		expect(Math.abs(first.validLoss - log.uniformLoss)).toBeLessThan(0.6);
		expect(Math.abs(first.trainLoss - log.uniformLoss)).toBeLessThan(0.6);
	});
});

const at = (step: number) => {
	const c = log.checkpoints.find((x) => x.step === step);
	if (!c) throw new Error(`no checkpoint at step ${step}`);
	return c;
};

describe('the log belongs to the shipped model', () => {
	it('ends at the step and the held-out loss recorded with the weights', () => {
		const last = log.checkpoints[log.checkpoints.length - 1];
		expect(last.step).toBe(LM_TRAINING.steps);
		expect(log.steps).toBe(LM_TRAINING.steps);
		expect(last.validLoss).toBe(LM_TRAINING.validLoss);
		expect(log.batch).toBe(LM_TRAINING.batch);
	});
});

// The lesson's narration states these facts about this particular run. If the model is retrained and one
// fails, rewrite that sentence of lesson.ts to match the new log.
describe('what the narration says about this training run', () => {
	it('step 1: the untrained model draws tokens with no pattern, and repeats itself without the dice', () => {
		const c = at(0);
		expect(new Set(c.sampled[0]).size).toBeGreaterThanOrEqual(20);
		for (const story of c.greedy) expect(new Set(story).size).toBeLessThanOrEqual(4); // of 28 tokens
	});
	it('step 2: every bar of the untrained model is nearly empty', () => {
		for (const p of at(0).probe) expect(p).toBeLessThan(0.005);
	});
	it('step 3: the untrained model starts just above the loss of guessing blindly', () => {
		const first = at(0).trainLoss;
		expect(first).toBeGreaterThan(log.uniformLoss);
		expect(first).toBeLessThan(log.uniformLoss + 0.5);
	});
	it('step 5: full stops, then a stuck sentence, then the finished opening', () => {
		expect(at(25).greedy[0].every((t) => t === '.')).toBe(true);
		const frequent = new Set(['big', '.', '"', 'i', ',']);
		expect(at(100).greedy[0].every((t) => frequent.has(t))).toBe(true);
		const stuck = at(200).greedy[0].join(' ');
		expect(stuck.split('he was very happy .').length - 1).toBeGreaterThanOrEqual(3);
		const opening = 'little girl named lily . she loved to play with her toys';
		expect(at(400).greedy[0].join(' ').startsWith(opening)).toBe(true);
		expect(at(LM_TRAINING.steps).greedy[0].join(' ').startsWith(opening)).toBe(true);
	});
	it('step 5: with the dice, a name is lost at step 800 and kept at step 3,200', () => {
		expect(at(800).sampled[0].join(' ')).toContain('named spot . max loved');
		expect(at(3200).sampled[0].join(' ')).toContain('named mark . mark loved');
	});
	it('step 6: the loss on unseen stories stays close to the training loss', () => {
		for (const c of log.checkpoints) expect(Math.abs(c.validLoss - c.trainLoss)).toBeLessThan(0.15);
	});
});

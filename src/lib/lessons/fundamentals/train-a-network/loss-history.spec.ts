import { describe, expect, it } from 'vitest';
import { LossHistory, MAX_SAMPLES } from './loss-history.ts';

describe('LossHistory', () => {
	it('records every step while short', () => {
		const h = new LossHistory();
		h.reset(1);
		for (let i = 0; i < 10; i++) h.push(1 - i * 0.05);
		expect(h.values.length).toBe(10);
		expect(h.interval).toBe(1);
	});

	it('halves and doubles the interval past the cap', () => {
		const h = new LossHistory();
		h.reset(1);
		for (let i = 0; i < MAX_SAMPLES + 1; i++) h.push(i);
		expect(h.interval).toBe(2);
		expect(h.values.length).toBe(Math.floor((MAX_SAMPLES + 1) / 2));
		const n = h.values.length;
		h.push(1000);
		expect(h.values.length).toBe(n + 1);
		h.push(1001);
		expect(h.values.length).toBe(n + 1);
	});

	it('tracks the max since reset and ignores non-finite values', () => {
		const h = new LossHistory();
		h.reset(0.7);
		h.push(0.9);
		h.push(NaN);
		expect(h.max).toBe(0.9);
		h.reset(0.5);
		expect(h.max).toBe(0.5);
		expect(h.values.length).toBe(0);
	});
});

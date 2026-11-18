import { describe, expect, it } from 'vitest';
import { buildNet, outputRange } from './network.ts';

describe('forward-pass network', () => {
	it('output moves visibly as the inputs sweep', () => {
		const r = outputRange(buildNet());
		expect(r.min).toBeLessThan(0.3);
		expect(r.max).toBeGreaterThan(0.7);
	});
});

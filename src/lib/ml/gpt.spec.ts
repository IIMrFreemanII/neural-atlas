import { describe, expect, it } from 'vitest';
import { createWorkspace, initGpt, layout, loss, lossAndGrad, nextLogits, paramCount, readout, trace, type Gpt, type GptConfig } from './gpt.ts';

const CONFIG: GptConfig = { vocab: 11, d: 8, layers: 2, heads: 2, ctx: 6 };
// Repeats 3 and 5.
const SEQ = [3, 5, 1, 3, 7, 5, 2];

function makeModel(config = CONFIG, seed = 5): Gpt {
	const m = initGpt(config, seed);
	const params = Float64Array.from(m.params);
	// Biases start at zero; give them values so their gradients are tested too.
	const lay = layout(config);
	for (const L of lay.layers) {
		for (let i = 0; i < 4 * config.d; i++) params[L.b1 + i] = 0.05 * Math.sin(i + 1);
		for (let i = 0; i < config.d; i++) params[L.b2 + i] = 0.05 * Math.cos(i + 1);
	}
	return { config, params };
}

function gradCheck(n: number): number {
	const model = makeModel();
	const ws = createWorkspace(CONFIG);
	const grads = new Float64Array(paramCount(CONFIG));
	lossAndGrad(model, ws, SEQ, n, grads);
	const lay = layout(CONFIG);
	const d = CONFIG.d;
	const indices: number[] = [];
	const inputs = new Set(SEQ.slice(0, n));
	const rows = new Set<number>([...SEQ.slice(0, n + 1), 0, 9]);
	for (const t of rows) for (let c = 0; c < d; c++) indices.push(lay.tok + t * d + c);
	for (let i = 0; i < CONFIG.ctx * d; i++) indices.push(lay.pos + i);
	const L0 = lay.layers[0];
	for (let i = L0.wq; i < lay.total; i++) indices.push(i);
	expect(inputs.size).toBeGreaterThan(0);
	let worst = 0;
	const eps = 1e-5;
	for (const idx of indices) {
		const orig = model.params[idx];
		model.params[idx] = orig + eps;
		const up = loss(model, ws, SEQ, n);
		model.params[idx] = orig - eps;
		const down = loss(model, ws, SEQ, n);
		model.params[idx] = orig;
		const numeric = (up - down) / (2 * eps);
		const err = Math.abs(grads[idx] - numeric);
		const bound = 1e-6 + 1e-4 * Math.abs(numeric);
		expect(err, `param ${idx}: analytic ${grads[idx]} numeric ${numeric}`).toBeLessThanOrEqual(bound);
		worst = Math.max(worst, err / (1e-6 + Math.abs(numeric)));
	}
	return worst;
}

describe('gradient check', () => {
	it('matches central differences for n = ctx', () => {
		const worst = gradCheck(6);
		console.log(`gradient check n=6: worst error / (1e-6 + |numeric|) = ${worst.toExponential(3)}`);
	});
	it('matches central differences for n = 3', () => {
		const worst = gradCheck(3);
		console.log(`gradient check n=3: worst error / (1e-6 + |numeric|) = ${worst.toExponential(3)}`);
	});
});

describe('lossAndGrad', () => {
	it('adds into grads', () => {
		const model = makeModel();
		const ws = createWorkspace(CONFIG);
		const once = new Float64Array(paramCount(CONFIG));
		lossAndGrad(model, ws, SEQ, 6, once);
		const twice = new Float64Array(paramCount(CONFIG));
		lossAndGrad(model, ws, SEQ, 6, twice);
		lossAndGrad(model, ws, SEQ, 6, twice);
		for (let i = 0; i < once.length; i++) expect(Math.abs(twice[i] - 2 * once[i])).toBeLessThanOrEqual(1e-12 * (1 + Math.abs(once[i])));
	});
	it('returns the same value as loss', () => {
		const model = makeModel();
		const ws = createWorkspace(CONFIG);
		const l = lossAndGrad(model, ws, SEQ, 6, new Float64Array(paramCount(CONFIG)));
		expect(loss(model, ws, SEQ, 6)).toBe(l);
	});
});

describe('trace and readout', () => {
	it('is causal and its attention rows sum to 1', () => {
		const model = makeModel();
		const a = trace(model, [3, 5, 1, 3, 7]);
		const b = trace(model, [3, 5, 1, 3, 9]);
		for (let l = 0; l < CONFIG.layers; l++) {
			for (let i = 0; i < 4; i++) {
				for (const key of ['x', 'norm1', 'attnOut', 'mid', 'norm2', 'hidden', 'mlpOut', 'out'] as const) {
					expect(a.layers[l][key][i]).toEqual(b.layers[l][key][i]);
				}
			}
			expect(a.layers[l].out[4]).not.toEqual(b.layers[l].out[4]);
			for (let h = 0; h < CONFIG.heads; h++) {
				a.layers[l].heads[h].forEach((r, i) => {
					expect(r.reduce((s, z) => s + z, 0)).toBeCloseTo(1, 10);
					for (let j = i + 1; j < r.length; j++) expect(r[j]).toBe(0);
				});
			}
		}
	});
	it('reads out to the real logits and the real loss', () => {
		const model = makeModel();
		const ws = createWorkspace(CONFIG);
		const n = 6;
		const tr = trace(model, SEQ.slice(0, n));
		const last = tr.layers[CONFIG.layers - 1].out;
		const direct = nextLogits(model, SEQ.slice(0, n));
		const viaReadout = readout(model, last[n - 1]);
		for (let t = 0; t < CONFIG.vocab; t++) expect(Math.abs(viaReadout[t] - direct[t])).toBeLessThan(1e-4);
		let total = 0;
		for (let i = 0; i < n; i++) {
			const lg = Array.from(readout(model, last[i]));
			const max = Math.max(...lg);
			const lse = max + Math.log(lg.reduce((s, z) => s + Math.exp(z - max), 0));
			total += lse - lg[SEQ[i + 1]];
		}
		expect(Math.abs(total / n - loss(model, ws, SEQ, n))).toBeLessThan(1e-5);
	});
	it('nextLogits uses only the last ctx tokens', () => {
		const model = makeModel();
		const long = [4, 8, 2, 2, 9, 1, 3, 5, 6, 7];
		const a = nextLogits(model, long);
		const b = nextLogits(model, long.slice(-CONFIG.ctx));
		expect(Array.from(a)).toEqual(Array.from(b));
	});
});

describe('training', () => {
	it('300 plain SGD steps on one sequence bring the loss below 0.1', () => {
		const model = makeModel();
		const ws = createWorkspace(CONFIG);
		const grads = new Float64Array(paramCount(CONFIG));
		let last = Infinity;
		for (let step = 0; step < 300; step++) {
			grads.fill(0);
			last = lossAndGrad(model, ws, SEQ, 6, grads);
			for (let i = 0; i < grads.length; i++) model.params[i] -= 0.1 * grads[i];
		}
		expect(last).toBeLessThan(0.1);
	});
});

/**
 * Trains the tiny CNN of the "CNN feature maps" lesson on generated shape drawings and writes its weights.
 * Run from the project root:  node --experimental-strip-types scripts/train-shapes-cnn.ts
 * Fixed seeds: running it again reproduces src/lib/lessons/vision/cnn-features/weights.ts.
 */
import { writeFileSync } from 'node:fs';
import { backward, C1, C2, CLASSES, forward, initCnn, type CnnWeights } from '../src/lib/ml/cnn.ts';
import { mulberry32 } from '../src/lib/ml/rng.ts';
import { randomShape, SAMPLES } from '../src/lib/lessons/vision/cnn-features/data.ts';

const INIT_SEED = 7;
const TRAIN_SEED = 1001;
const EVAL_SEED = 2002;
const STEPS = 30000;
const BATCH = 32;
const LR = 0.003;
const LR_END = 0.0003;
const BETA1 = 0.9;
const BETA2 = 0.999;
const EPS = 1e-8;
const OUT = new URL('../src/lib/lessons/vision/cnn-features/weights.ts', import.meta.url);

const keys = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3'] as const;
const w = initCnn(INIT_SEED);
const m = Object.fromEntries(keys.map((k) => [k, new Float64Array(w[k].length)])) as Record<(typeof keys)[number], Float64Array>;
const v = Object.fromEntries(keys.map((k) => [k, new Float64Array(w[k].length)])) as Record<(typeof keys)[number], Float64Array>;

function evaluate(weights: CnnWeights, seed: number, n: number): number {
	const rand = mulberry32(seed);
	let ok = 0;
	for (let i = 0; i < n; i++) {
		const s = randomShape(rand);
		const p = forward(weights, s.image).probs;
		let best = 0;
		for (let c = 1; c < CLASSES; c++) if (p[c] > p[best]) best = c;
		if (best === s.label) ok++;
	}
	return ok / n;
}

const t0 = Date.now();
const rand = mulberry32(TRAIN_SEED);
let runLoss = 0;
for (let step = 1; step <= STEPS; step++) {
	const acc = Object.fromEntries(keys.map((k) => [k, new Float64Array(w[k].length)])) as Record<(typeof keys)[number], Float64Array>;
	for (let b = 0; b < BATCH; b++) {
		const s = randomShape(rand);
		const t = forward(w, s.image);
		runLoss += -Math.log(Math.max(t.probs[s.label], 1e-12)) / BATCH;
		const g = backward(w, t, s.label);
		for (const k of keys) for (let i = 0; i < g[k].length; i++) acc[k][i] += g[k][i] / BATCH;
	}
	const lr = LR + (LR_END - LR) * (step / STEPS);
	for (const k of keys) {
		const mk = m[k];
		const vk = v[k];
		for (let i = 0; i < w[k].length; i++) {
			const gi = acc[k][i];
			mk[i] = BETA1 * mk[i] + (1 - BETA1) * gi;
			vk[i] = BETA2 * vk[i] + (1 - BETA2) * gi * gi;
			const mh = mk[i] / (1 - BETA1 ** step);
			const vh = vk[i] / (1 - BETA2 ** step);
			w[k][i] -= (lr * mh) / (Math.sqrt(vh) + EPS);
		}
	}
	if (step % 2500 === 0) {
		console.log(`step ${step}  loss ${(runLoss / 2500).toFixed(4)}  held-out acc ${evaluate(w, EVAL_SEED, 500).toFixed(3)}`);
		runLoss = 0;
	}
}

// Round to 4 decimals first, so the file and the reported accuracy describe the same numbers.
for (const k of keys) for (let i = 0; i < w[k].length; i++) w[k][i] = Math.round(w[k][i] * 1e4) / 1e4;
const finalAcc = evaluate(w, EVAL_SEED, 2000);
console.log(`final held-out accuracy (2000 samples, seed ${EVAL_SEED}): ${finalAcc.toFixed(4)}`);
const names = ['circle', 'square', 'triangle', 'cross'];
SAMPLES.forEach((img, i) => {
	const p = forward(w, img).probs;
	console.log(`sample ${names[i]}: ${names.map((n, c) => `${n} ${p[c].toFixed(3)}`).join('  ')}`);
});

const arr = (a: Float32Array) => `new Float32Array([${Array.from(a).join(', ')}])`;
const src = `// Written by scripts/train-shapes-cnn.ts. Do not edit by hand.
// Adam, learning rate ${LR} falling to ${LR_END}, batch ${BATCH}, ${STEPS} steps, on freshly generated drawings.
// Init seed ${INIT_SEED}, training seed ${TRAIN_SEED}, evaluation seed ${EVAL_SEED}.
// Held-out accuracy on 2000 generated drawings: ${(finalAcc * 100).toFixed(2)}%.
import type { CnnWeights } from '../../../ml/cnn.ts';

export const WEIGHTS: CnnWeights = {
	w1: ${arr(w.w1)},
	b1: ${arr(w.b1)},
	w2: ${arr(w.w2)},
	b2: ${arr(w.b2)},
	w3: ${arr(w.w3)},
	b3: ${arr(w.b3)}
};
`;
writeFileSync(OUT, src);
console.log(`wrote weights (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
void C1;
void C2;

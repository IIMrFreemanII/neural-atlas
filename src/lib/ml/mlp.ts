import { mulberry32 } from './rng.ts';

export type Act = 'linear' | 'sigmoid' | 'tanh' | 'relu';
export type LossKind = 'mse' | 'bce';

export interface MLP {
	sizes: number[];
	hidden: Act;
	output: Act;
	/** W[l][j * sizes[l] + i] connects unit i of layer l to unit j of layer l+1. */
	W: Float64Array[];
	b: Float64Array[];
}

export function createMLP(sizes: number[], hidden: Act, output: Act, seed: number): MLP {
	if (sizes.length < 2) throw new Error('MLP needs at least 2 layers');
	const rand = mulberry32(seed);
	const W: Float64Array[] = [];
	const b: Float64Array[] = [];
	for (let l = 0; l + 1 < sizes.length; l++) {
		const fanIn = sizes[l];
		const fanOut = sizes[l + 1];
		const isLast = l === sizes.length - 2;
		const act = isLast ? output : hidden;
		const limit = act === 'relu' ? Math.sqrt(6 / fanIn) : Math.sqrt(6 / (fanIn + fanOut));
		const w = new Float64Array(fanIn * fanOut);
		for (let k = 0; k < w.length; k++) w[k] = (rand() * 2 - 1) * limit;
		W.push(w);
		b.push(new Float64Array(fanOut));
	}
	return { sizes: sizes.slice(), hidden, output, W, b };
}

export function cloneMLP(net: MLP): MLP {
	return {
		sizes: net.sizes.slice(),
		hidden: net.hidden,
		output: net.output,
		W: net.W.map((w) => w.slice()),
		b: net.b.map((v) => v.slice())
	};
}

export function actFn(a: Act, z: number): number {
	switch (a) {
		case 'linear':
			return z;
		case 'sigmoid':
			return 1 / (1 + Math.exp(-z));
		case 'tanh':
			return Math.tanh(z);
		case 'relu':
			return z > 0 ? z : 0;
	}
}

export function actDeriv(a: Act, z: number): number {
	switch (a) {
		case 'linear':
			return 1;
		case 'sigmoid': {
			const s = 1 / (1 + Math.exp(-z));
			return s * (1 - s);
		}
		case 'tanh': {
			const t = Math.tanh(z);
			return 1 - t * t;
		}
		case 'relu':
			return z > 0 ? 1 : 0;
	}
}

export interface Trace {
	z: Float64Array[];
	a: Float64Array[];
}

export function forward(net: MLP, x: ArrayLike<number>): Trace {
	const L = net.W.length;
	const a: Float64Array[] = [Float64Array.from(x)];
	const z: Float64Array[] = [];
	for (let l = 0; l < L; l++) {
		const nIn = net.sizes[l];
		const nOut = net.sizes[l + 1];
		const act = l === L - 1 ? net.output : net.hidden;
		const w = net.W[l];
		const prev = a[l];
		const zl = new Float64Array(nOut);
		const al = new Float64Array(nOut);
		for (let j = 0; j < nOut; j++) {
			let s = net.b[l][j];
			const row = j * nIn;
			for (let i = 0; i < nIn; i++) s += w[row + i] * prev[i];
			zl[j] = s;
			al[j] = actFn(act, s);
		}
		z.push(zl);
		a.push(al);
	}
	return { z, a };
}

export function predict(net: MLP, x: ArrayLike<number>): Float64Array {
	const t = forward(net, x);
	return t.a[t.a.length - 1];
}

const EPS = 1e-7;

export function lossValue(kind: LossKind, out: ArrayLike<number>, target: ArrayLike<number>): number {
	let s = 0;
	for (let i = 0; i < out.length; i++) {
		if (kind === 'mse') {
			const d = out[i] - target[i];
			s += 0.5 * d * d;
		} else {
			const o = Math.min(1 - EPS, Math.max(EPS, out[i]));
			s -= target[i] * Math.log(o) + (1 - target[i]) * Math.log(1 - o);
		}
	}
	return s;
}

export interface Grads {
	W: Float64Array[];
	b: Float64Array[];
	delta: Float64Array[];
	loss: number;
}

export function backward(net: MLP, trace: Trace, target: ArrayLike<number>, kind: LossKind): Grads {
	const L = net.W.length;
	const out = trace.a[L];
	const delta: Float64Array[] = new Array(L);
	const dLast = new Float64Array(out.length);
	for (let j = 0; j < out.length; j++) {
		if (kind === 'bce' && net.output === 'sigmoid') {
			// Stable form of dL/da * sigmoid'(z); matches the clamped loss away from saturation.
			dLast[j] = out[j] - target[j];
		} else {
			let dA: number;
			if (kind === 'mse') dA = out[j] - target[j];
			else if (out[j] < EPS || out[j] > 1 - EPS) dA = 0;
			else dA = -(target[j] / out[j] - (1 - target[j]) / (1 - out[j]));
			dLast[j] = dA * actDeriv(net.output, trace.z[L - 1][j]);
		}
	}
	delta[L - 1] = dLast;
	for (let l = L - 2; l >= 0; l--) {
		const nOut = net.sizes[l + 1];
		const nNext = net.sizes[l + 2];
		const w = net.W[l + 1];
		const d = new Float64Array(nOut);
		for (let i = 0; i < nOut; i++) {
			let s = 0;
			for (let j = 0; j < nNext; j++) s += w[j * nOut + i] * delta[l + 1][j];
			d[i] = s * actDeriv(net.hidden, trace.z[l][i]);
		}
		delta[l] = d;
	}
	const gW: Float64Array[] = [];
	const gb: Float64Array[] = [];
	for (let l = 0; l < L; l++) {
		const nIn = net.sizes[l];
		const nOut = net.sizes[l + 1];
		const g = new Float64Array(nIn * nOut);
		for (let j = 0; j < nOut; j++) {
			for (let i = 0; i < nIn; i++) g[j * nIn + i] = delta[l][j] * trace.a[l][i];
		}
		gW.push(g);
		gb.push(delta[l].slice());
	}
	return { W: gW, b: gb, delta, loss: lossValue(kind, out, target) };
}

export function applyGrads(net: MLP, g: Pick<Grads, 'W' | 'b'>, lr: number): void {
	for (let l = 0; l < net.W.length; l++) {
		const w = net.W[l];
		const gw = g.W[l];
		for (let k = 0; k < w.length; k++) w[k] -= lr * gw[k];
		const b = net.b[l];
		const gb = g.b[l];
		for (let k = 0; k < b.length; k++) b[k] -= lr * gb[k];
	}
}

export function trainStep(
	net: MLP,
	xs: Float64Array,
	ys: Float64Array,
	indices: ArrayLike<number>,
	lr: number,
	kind: LossKind
): number {
	const nIn = net.sizes[0];
	const nOut = net.sizes[net.sizes.length - 1];
	const m = indices.length;
	if (m === 0) return 0;
	const accW = net.W.map((w) => new Float64Array(w.length));
	const accB = net.b.map((v) => new Float64Array(v.length));
	let total = 0;
	for (let r = 0; r < m; r++) {
		const idx = indices[r];
		const x = xs.subarray(idx * nIn, (idx + 1) * nIn);
		const y = ys.subarray(idx * nOut, (idx + 1) * nOut);
		const g = backward(net, forward(net, x), y, kind);
		total += g.loss;
		for (let l = 0; l < accW.length; l++) {
			for (let k = 0; k < accW[l].length; k++) accW[l][k] += g.W[l][k];
			for (let k = 0; k < accB[l].length; k++) accB[l][k] += g.b[l][k];
		}
	}
	for (let l = 0; l < accW.length; l++) {
		for (let k = 0; k < accW[l].length; k++) accW[l][k] /= m;
		for (let k = 0; k < accB[l].length; k++) accB[l][k] /= m;
	}
	applyGrads(net, { W: accW, b: accB }, lr);
	return total / m;
}

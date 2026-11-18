import { gelu, layerNorm } from './block.ts';
import { gaussian, mulberry32 } from './rng.ts';
import type { Mat } from './vec.ts';

export interface GptConfig {
	vocab: number;
	d: number;
	layers: number;
	heads: number;
	ctx: number;
}
export interface Gpt {
	config: GptConfig;
	params: Float32Array | Float64Array;
}

type Params = Float32Array | Float64Array;

const LN_EPS = 1e-5;
const GELU_C = Math.sqrt(2 / Math.PI);

export interface LayerLayout {
	wq: number;
	wk: number;
	wv: number;
	wo: number;
	w1: number;
	b1: number;
	w2: number;
	b2: number;
}
export interface GptLayout {
	tok: number;
	pos: number;
	layers: LayerLayout[];
	total: number;
}

/** Offset of every parameter block in the flat array. */
export function layout(c: GptConfig): GptLayout {
	const d = c.d;
	let at = 0;
	const tok = at;
	at += c.vocab * d;
	const pos = at;
	at += c.ctx * d;
	const layers: LayerLayout[] = [];
	for (let l = 0; l < c.layers; l++) {
		const wq = at;
		at += d * d;
		const wk = at;
		at += d * d;
		const wv = at;
		at += d * d;
		const wo = at;
		at += d * d;
		const w1 = at;
		at += 4 * d * d;
		const b1 = at;
		at += 4 * d;
		const w2 = at;
		at += 4 * d * d;
		const b2 = at;
		at += d;
		layers.push({ wq, wk, wv, wo, w1, b1, w2, b2 });
	}
	return { tok, pos, layers, total: at };
}

export function paramCount(c: GptConfig): number {
	return layout(c).total;
}

/** Seeded Gaussian start: tok and pos sd 0.1; wq, wk, wv, w1 sd 1/sqrt(d); wo sd 1/sqrt(d)/sqrt(2*layers); w2 sd 1/sqrt(4d)/sqrt(2*layers); biases 0. Float32Array params. */
export function initGpt(config: GptConfig, seed: number): Gpt {
	const { d, layers } = config;
	const lay = layout(config);
	const params = new Float32Array(lay.total);
	const rand = mulberry32(seed);
	const fill = (from: number, count: number, sd: number): void => {
		for (let i = 0; i < count; i++) params[from + i] = sd * gaussian(rand);
	};
	fill(lay.tok, config.vocab * d, 0.1);
	fill(lay.pos, config.ctx * d, 0.1);
	const sdIn = 1 / Math.sqrt(d);
	const sdOut = sdIn / Math.sqrt(2 * layers);
	const sdW2 = 1 / Math.sqrt(4 * d) / Math.sqrt(2 * layers);
	for (const L of lay.layers) {
		fill(L.wq, d * d, sdIn);
		fill(L.wk, d * d, sdIn);
		fill(L.wv, d * d, sdIn);
		fill(L.wo, d * d, sdOut);
		fill(L.w1, 4 * d * d, sdIn);
		fill(L.w2, 4 * d * d, sdW2);
	}
	return { config, params };
}

/** Reusable scratch memory for lossAndGrad, sized for sequences of up to config.ctx tokens. */
export interface GptWorkspace {
	config: GptConfig;
	lay: GptLayout;
	/** (layers + 1) x T x d: the stream entering each layer, then the final stream. */
	xs: Float64Array;
	a: Float64Array;
	inv1: Float64Array;
	q: Float64Array;
	k: Float64Array;
	v: Float64Array;
	/** layers x heads x T x T */
	att: Float64Array;
	o: Float64Array;
	mid: Float64Array;
	m: Float64Array;
	inv2: Float64Array;
	pre: Float64Array;
	hid: Float64Array;
	f: Float64Array;
	invF: Float64Array;
	// backward scratch
	dx: Float64Array;
	dmid: Float64Array;
	dq: Float64Array;
	dk: Float64Array;
	dv: Float64Array;
	dO: Float64Array;
	da: Float64Array;
	dm: Float64Array;
	dpre: Float64Array;
	dhid: Float64Array;
	df: Float64Array;
	dw: Float64Array;
	logits: Float64Array;
}

export function createWorkspace(config: GptConfig): GptWorkspace {
	const { d, layers, heads, ctx: T, vocab } = config;
	const f64 = (n: number): Float64Array => new Float64Array(n);
	return {
		config,
		lay: layout(config),
		xs: f64((layers + 1) * T * d),
		a: f64(layers * T * d),
		inv1: f64(layers * T),
		q: f64(layers * T * d),
		k: f64(layers * T * d),
		v: f64(layers * T * d),
		att: f64(layers * heads * T * T),
		o: f64(layers * T * d),
		mid: f64(layers * T * d),
		m: f64(layers * T * d),
		inv2: f64(layers * T),
		pre: f64(layers * T * 4 * d),
		hid: f64(layers * T * 4 * d),
		f: f64(T * d),
		invF: f64(T),
		dx: f64(T * d),
		dmid: f64(T * d),
		dq: f64(T * d),
		dk: f64(T * d),
		dv: f64(T * d),
		dO: f64(T * d),
		da: f64(d),
		dm: f64(d),
		dpre: f64(4 * d),
		dhid: f64(4 * d),
		df: f64(d),
		dw: f64(T),
		logits: f64(vocab)
	};
}

// ---------------------------------------------------------------- kernels (plain functions, no allocation)

/** out[oo + r] = sum_c w[wo + r * cols + c] * x[xo + c] (+ bias[bo + r] when bo >= 0) */
function matVecInto(out: Float64Array, oo: number, w: Params, wo: number, rows: number, cols: number, x: Float64Array, xo: number, bias: Params | null, bo: number): void {
	for (let r = 0; r < rows; r++) {
		let s = bias === null ? 0 : bias[bo + r];
		const base = wo + r * cols;
		for (let c = 0; c < cols; c++) s += w[base + c] * x[xo + c];
		out[oo + r] = s;
	}
}

/** dst[do + c] += sum_r w[wo + r * cols + c] * dy[yo + r] */
function matTVecAdd(dst: Float64Array, dOff: number, w: Params, wo: number, rows: number, cols: number, dy: Float64Array, yo: number): void {
	for (let r = 0; r < rows; r++) {
		const g = dy[yo + r];
		const base = wo + r * cols;
		for (let c = 0; c < cols; c++) dst[dOff + c] += w[base + c] * g;
	}
}

/** grads[go + r * cols + c] += dy[yo + r] * x[xo + c] */
function outerAdd(grads: Float64Array, go: number, rows: number, cols: number, dy: Float64Array, yo: number, x: Float64Array, xo: number): void {
	for (let r = 0; r < rows; r++) {
		const g = dy[yo + r];
		const base = go + r * cols;
		for (let c = 0; c < cols; c++) grads[base + c] += g * x[xo + c];
	}
}

/** dst = (src - mean) * inv, inv stored at invOut[io]. */
function lnForward(src: Float64Array, so: number, dst: Float64Array, dOff: number, d: number, invOut: Float64Array, io: number): void {
	let mean = 0;
	for (let c = 0; c < d; c++) mean += src[so + c];
	mean /= d;
	let variance = 0;
	for (let c = 0; c < d; c++) {
		const e = src[so + c] - mean;
		variance += e * e;
	}
	variance /= d;
	const inv = 1 / Math.sqrt(variance + LN_EPS);
	invOut[io] = inv;
	for (let c = 0; c < d; c++) dst[dOff + c] = (src[so + c] - mean) * inv;
}

/** dst[do + c] += (dy - mean(dy) - y * mean(dy * y)) * inv */
function lnBackAdd(dy: Float64Array, yo: number, y: Float64Array, yOff: number, inv: number, d: number, dst: Float64Array, dOff: number): void {
	let mdy = 0;
	let mdyy = 0;
	for (let c = 0; c < d; c++) {
		mdy += dy[yo + c];
		mdyy += dy[yo + c] * y[yOff + c];
	}
	mdy /= d;
	mdyy /= d;
	for (let c = 0; c < d; c++) dst[dOff + c] += (dy[yo + c] - mdy - y[yOff + c] * mdyy) * inv;
}

// ---------------------------------------------------------------- forward over the workspace

/** Runs the layers and the final LN for the first n tokens, leaving every activation in ws. */
function forward(model: Gpt, ws: GptWorkspace, tokens: ArrayLike<number>, n: number): void {
	const { d, layers, heads, ctx: T } = model.config;
	const p = model.params;
	const lay = ws.lay;
	const dh = d / heads;
	const scale = 1 / Math.sqrt(dh);
	const d4 = 4 * d;
	const { xs, a, inv1, q, k, v, att, o, mid, m, inv2, pre, hid } = ws;

	for (let i = 0; i < n; i++) {
		const t = tokens[i] * d;
		const ps = lay.pos + i * d;
		for (let c = 0; c < d; c++) xs[i * d + c] = p[lay.tok + t + c] + p[ps + c];
	}

	for (let l = 0; l < layers; l++) {
		const L = lay.layers[l];
		const xo = l * T * d;
		const ao = l * T * d;
		const io = l * T;
		for (let i = 0; i < n; i++) {
			lnForward(xs, xo + i * d, a, ao + i * d, d, inv1, io + i);
			matVecInto(q, ao + i * d, p, L.wq, d, d, a, ao + i * d, null, 0);
			matVecInto(k, ao + i * d, p, L.wk, d, d, a, ao + i * d, null, 0);
			matVecInto(v, ao + i * d, p, L.wv, d, d, a, ao + i * d, null, 0);
		}
		for (let h = 0; h < heads; h++) {
			const hc = h * dh;
			const attBase = (l * heads + h) * T * T;
			for (let i = 0; i < n; i++) {
				const row = attBase + i * T;
				const qo = ao + i * d + hc;
				let max = -Infinity;
				for (let j = 0; j <= i; j++) {
					const ko = ao + j * d + hc;
					let s = 0;
					for (let c = 0; c < dh; c++) s += q[qo + c] * k[ko + c];
					s *= scale;
					att[row + j] = s;
					if (s > max) max = s;
				}
				let sum = 0;
				for (let j = 0; j <= i; j++) {
					const e = Math.exp(att[row + j] - max);
					att[row + j] = e;
					sum += e;
				}
				for (let j = 0; j <= i; j++) att[row + j] /= sum;
				for (let j = i + 1; j < T; j++) att[row + j] = 0;
				const oo = ao + i * d + hc;
				for (let c = 0; c < dh; c++) o[oo + c] = 0;
				for (let j = 0; j <= i; j++) {
					const w = att[row + j];
					const vo = ao + j * d + hc;
					for (let c = 0; c < dh; c++) o[oo + c] += w * v[vo + c];
				}
			}
		}
		for (let i = 0; i < n; i++) {
			const mo = ao + i * d;
			// mid = x + wo o
			matVecInto(mid, mo, p, L.wo, d, d, o, mo, null, 0);
			for (let c = 0; c < d; c++) mid[mo + c] += xs[xo + i * d + c];
			lnForward(mid, mo, m, mo, d, inv2, io + i);
			const ho = (l * T + i) * d4;
			matVecInto(pre, ho, p, L.w1, d4, d, m, mo, p, L.b1);
			for (let c = 0; c < d4; c++) hid[ho + c] = gelu(pre[ho + c]);
			// next stream = mid + w2 hid + b2
			const no = (l + 1) * T * d + i * d;
			matVecInto(xs, no, p, L.w2, d, d4, hid, ho, p, L.b2);
			for (let c = 0; c < d; c++) xs[no + c] += mid[mo + c];
		}
	}

	const fo = layers * T * d;
	for (let i = 0; i < n; i++) lnForward(xs, fo + i * d, ws.f, i * d, d, ws.invF, i);
}

/** logits for position i into ws.logits (tok . f[i]); returns nothing. */
function positionLogits(model: Gpt, ws: GptWorkspace, i: number): void {
	const { d, vocab } = model.config;
	matVecInto(ws.logits, 0, model.params, ws.lay.tok, vocab, d, ws.f, i * d, null, 0);
}

/** Cross-entropy of ws.logits against target; overwrites ws.logits with the probabilities. */
function softmaxLogits(ws: GptWorkspace, vocab: number, target: number): number {
	const lg = ws.logits;
	let max = -Infinity;
	for (let t = 0; t < vocab; t++) if (lg[t] > max) max = lg[t];
	const shifted = lg[target] - max;
	let sum = 0;
	for (let t = 0; t < vocab; t++) {
		const e = Math.exp(lg[t] - max);
		lg[t] = e;
		sum += e;
	}
	const ce = Math.log(sum) - shifted;
	for (let t = 0; t < vocab; t++) lg[t] /= sum;
	return ce;
}

/** Mean loss only, no gradient (for validation). Same tokens convention. */
export function loss(model: Gpt, ws: GptWorkspace, tokens: ArrayLike<number>, n: number): number {
	const { vocab } = model.config;
	forward(model, ws, tokens, n);
	let total = 0;
	for (let i = 0; i < n; i++) {
		positionLogits(model, ws, i);
		total += softmaxLogits(ws, vocab, tokens[i + 1]);
	}
	return total / n;
}

/**
 * tokens holds n + 1 ids (n <= ctx): inputs tokens[0..n-1], targets tokens[1..n].
 * ADDS the gradient of the mean loss into grads (length paramCount) and returns the mean loss.
 * Must not allocate arrays per call beyond what the workspace holds.
 */
export function lossAndGrad(model: Gpt, ws: GptWorkspace, tokens: ArrayLike<number>, n: number, grads: Float64Array): number {
	const { d, layers, heads, ctx: T, vocab } = model.config;
	const p = model.params;
	const lay = ws.lay;
	const dh = d / heads;
	const scale = 1 / Math.sqrt(dh);
	const d4 = 4 * d;
	const { a, inv1, q, k, v, att, o, m, inv2, pre, hid, f, invF } = ws;
	const { dx, dmid, dq, dk, dv, dO, da, dm, dpre, dhid, df, dw, logits } = ws;

	forward(model, ws, tokens, n);

	// Output layer: loss, d logits, tied-embedding gradient, d f -> d stream.
	let total = 0;
	dx.fill(0, 0, n * d);
	for (let i = 0; i < n; i++) {
		positionLogits(model, ws, i);
		const target = tokens[i + 1];
		total += softmaxLogits(ws, vocab, target);
		logits[target] -= 1;
		df.fill(0);
		for (let t = 0; t < vocab; t++) {
			const g = logits[t] / n;
			const base = lay.tok + t * d;
			for (let c = 0; c < d; c++) {
				grads[base + c] += g * f[i * d + c];
				df[c] += g * p[base + c];
			}
		}
		lnBackAdd(df, 0, f, i * d, invF[i], d, dx, i * d);
	}

	for (let l = layers - 1; l >= 0; l--) {
		const L = lay.layers[l];
		const ao = l * T * d;
		const io = l * T;

		// Feed-forward branch. dx is the gradient of the stream leaving the layer.
		for (let i = 0; i < n; i++) {
			const mo = ao + i * d;
			const ho = (l * T + i) * d4;
			// x_{l+1} = mid + w2 hid + b2
			for (let c = 0; c < d; c++) {
				grads[L.b2 + c] += dx[i * d + c];
				dmid[i * d + c] = dx[i * d + c];
			}
			outerAdd(grads, L.w2, d, d4, dx, i * d, hid, ho);
			dhid.fill(0);
			matTVecAdd(dhid, 0, p, L.w2, d, d4, dx, i * d);
			for (let c = 0; c < d4; c++) {
				const x = pre[ho + c];
				const t = Math.tanh(GELU_C * (x + 0.044715 * x * x * x));
				const gp = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GELU_C * (1 + 3 * 0.044715 * x * x);
				dpre[c] = dhid[c] * gp;
				grads[L.b1 + c] += dpre[c];
			}
			outerAdd(grads, L.w1, d4, d, dpre, 0, m, mo);
			dm.fill(0);
			matTVecAdd(dm, 0, p, L.w1, d4, d, dpre, 0);
			lnBackAdd(dm, 0, m, mo, inv2[io + i], d, dmid, i * d);
		}

		// Attention branch. mid = x + wo o, so dmid flows to x and to wo o.
		dq.fill(0, 0, n * d);
		dk.fill(0, 0, n * d);
		dv.fill(0, 0, n * d);
		for (let i = 0; i < n; i++) {
			const mo = ao + i * d;
			outerAdd(grads, L.wo, d, d, dmid, i * d, o, mo);
			for (let c = 0; c < d; c++) dO[i * d + c] = 0;
			matTVecAdd(dO, i * d, p, L.wo, d, d, dmid, i * d);
		}
		for (let h = 0; h < heads; h++) {
			const hc = h * dh;
			const attBase = (l * heads + h) * T * T;
			for (let i = 0; i < n; i++) {
				const row = attBase + i * T;
				const oi = i * d + hc;
				let s = 0;
				for (let j = 0; j <= i; j++) {
					const vo = ao + j * d + hc;
					let acc = 0;
					for (let c = 0; c < dh; c++) acc += dO[oi + c] * v[vo + c];
					dw[j] = acc;
					s += acc * att[row + j];
				}
				const qo = ao + i * d + hc;
				for (let j = 0; j <= i; j++) {
					const w = att[row + j];
					const ds = w * (dw[j] - s) * scale;
					const ko = ao + j * d + hc;
					const jo = j * d + hc;
					for (let c = 0; c < dh; c++) {
						dq[oi + c] += ds * k[ko + c];
						dk[jo + c] += ds * q[qo + c];
						dv[jo + c] += w * dO[oi + c];
					}
				}
			}
		}
		for (let i = 0; i < n; i++) {
			const mo = ao + i * d;
			da.fill(0);
			matTVecAdd(da, 0, p, L.wq, d, d, dq, i * d);
			matTVecAdd(da, 0, p, L.wk, d, d, dk, i * d);
			matTVecAdd(da, 0, p, L.wv, d, d, dv, i * d);
			outerAdd(grads, L.wq, d, d, dq, i * d, a, mo);
			outerAdd(grads, L.wk, d, d, dk, i * d, a, mo);
			outerAdd(grads, L.wv, d, d, dv, i * d, a, mo);
			for (let c = 0; c < d; c++) dx[i * d + c] = dmid[i * d + c];
			lnBackAdd(da, 0, a, mo, inv1[io + i], d, dx, i * d);
		}
	}

	// Embeddings: x_0 = tok[t] + pos[i].
	for (let i = 0; i < n; i++) {
		const tb = lay.tok + tokens[i] * d;
		const pb = lay.pos + i * d;
		for (let c = 0; c < d; c++) {
			grads[tb + c] += dx[i * d + c];
			grads[pb + c] += dx[i * d + c];
		}
	}
	return total / n;
}

const inferenceSpaces = new WeakMap<Gpt, GptWorkspace>();

function lastTokens(model: Gpt, tokens: ArrayLike<number>): number[] {
	const from = Math.max(0, tokens.length - model.config.ctx);
	const out: number[] = [];
	for (let i = from; i < tokens.length; i++) out.push(tokens[i]);
	return out;
}

/** Logits for the token after the given ones (length vocab). Uses the last config.ctx tokens if more are given. tokens must not be empty. */
export function nextLogits(model: Gpt, tokens: ArrayLike<number>): Float32Array {
	if (tokens.length === 0) throw new Error('nextLogits needs at least one token');
	let ws = inferenceSpaces.get(model);
	if (ws === undefined) {
		ws = createWorkspace(model.config);
		inferenceSpaces.set(model, ws);
	}
	const t = lastTokens(model, tokens);
	forward(model, ws, t, t.length);
	positionLogits(model, ws, t.length - 1);
	return Float32Array.from(ws.logits);
}

export interface LayerTrace {
	/** n x d: the stream entering the layer. */
	x: Mat;
	/** LN of each row of x. */
	norm1: Mat;
	/** heads[h][i][j]: how much token i looks at token j in head h. Rows sum to 1; 0 for j > i. */
	heads: Mat[];
	/** n x d: wo applied to the gathered values, the update attention adds. */
	attnOut: Mat;
	/** x + attnOut */
	mid: Mat;
	/** LN of each row of mid. */
	norm2: Mat;
	/** n x 4d: gelu(w1 norm2 + b1). */
	hidden: Mat;
	/** n x d: w2 hidden + b2, the update the feed-forward layer adds. */
	mlpOut: Mat;
	/** mid + mlpOut: the stream leaving the layer. */
	out: Mat;
}
export interface GptTrace {
	tokens: number[];
	/** n x d: tok[t[i]] */
	tokenVectors: Mat;
	/** n x d: pos[i] */
	positionVectors: Mat;
	layers: LayerTrace[];
}

function mv(p: Params, off: number, rows: number, cols: number, x: ArrayLike<number>, bias = -1): number[] {
	const out = new Array<number>(rows);
	for (let r = 0; r < rows; r++) {
		let s = bias >= 0 ? p[bias + r] : 0;
		for (let c = 0; c < cols; c++) s += p[off + r * cols + c] * x[c];
		out[r] = s;
	}
	return out;
}

function row(p: Params, off: number, d: number): number[] {
	return Array.from(p.subarray(off, off + d));
}

/** Every intermediate value of a forward pass, as plain arrays, for the lessons to draw. Uses the last config.ctx tokens if more are given. */
export function trace(model: Gpt, tokens: ArrayLike<number>): GptTrace {
	const { d, heads, layers } = model.config;
	const p = model.params;
	const lay = layout(model.config);
	const dh = d / heads;
	const ids = lastTokens(model, tokens);
	const n = ids.length;
	const tokenVectors = ids.map((t) => row(p, lay.tok + t * d, d));
	const positionVectors = ids.map((_, i) => row(p, lay.pos + i * d, d));
	let x: Mat = tokenVectors.map((r, i) => r.map((v, c) => v + positionVectors[i][c]));
	const out: LayerTrace[] = [];
	for (let l = 0; l < layers; l++) {
		const L = lay.layers[l];
		const norm1 = x.map((r) => layerNorm(r));
		const q = norm1.map((r) => mv(p, L.wq, d, d, r));
		const k = norm1.map((r) => mv(p, L.wk, d, d, r));
		const v = norm1.map((r) => mv(p, L.wv, d, d, r));
		const headWeights: Mat[] = [];
		const gathered: Mat = ids.map(() => new Array<number>(d).fill(0));
		for (let h = 0; h < heads; h++) {
			const hc = h * dh;
			const w: Mat = [];
			for (let i = 0; i < n; i++) {
				const scores: number[] = [];
				for (let j = 0; j <= i; j++) {
					let s = 0;
					for (let c = 0; c < dh; c++) s += q[i][hc + c] * k[j][hc + c];
					scores.push(s / Math.sqrt(dh));
				}
				const max = Math.max(...scores);
				const e = scores.map((s) => Math.exp(s - max));
				const sum = e.reduce((acc, z) => acc + z, 0);
				const wi = new Array<number>(n).fill(0);
				for (let j = 0; j <= i; j++) wi[j] = e[j] / sum;
				for (let j = 0; j <= i; j++) for (let c = 0; c < dh; c++) gathered[i][hc + c] += wi[j] * v[j][hc + c];
				w.push(wi);
			}
			headWeights.push(w);
		}
		const attnOut = gathered.map((r) => mv(p, L.wo, d, d, r));
		const mid = x.map((r, i) => r.map((z, c) => z + attnOut[i][c]));
		const norm2 = mid.map((r) => layerNorm(r));
		const hidden = norm2.map((r) => mv(p, L.w1, 4 * d, d, r, L.b1).map(gelu));
		const mlpOut = hidden.map((r) => mv(p, L.w2, d, 4 * d, r, L.b2));
		const next = mid.map((r, i) => r.map((z, c) => z + mlpOut[i][c]));
		out.push({ x, norm1, heads: headWeights, attnOut, mid, norm2, hidden, mlpOut, out: next });
		x = next;
	}
	return { tokens: ids, tokenVectors, positionVectors, layers: out };
}

/** logits = tok LN(vector): reads a stream vector as next-token scores. With the last layer's out[i] this equals the model's real logits at position i. */
export function readout(model: Gpt, vector: ArrayLike<number>): Float32Array {
	const { d, vocab } = model.config;
	const lay = layout(model.config);
	const normed = layerNorm(Array.from(vector));
	return Float32Array.from(mv(model.params, lay.tok, vocab, d, normed));
}

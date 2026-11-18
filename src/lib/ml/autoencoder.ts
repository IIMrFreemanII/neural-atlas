import { actFn, createMLP, type MLP } from './mlp.ts';

export const IMAGE = 256; // 16 x 16 pixels, row-major, values 0..1
export const HIDDEN = 32;
export const LATENT = 2;

const SIZES = [IMAGE, HIDDEN, LATENT, HIDDEN, IMAGE];

/** One MLP with layers [256, HIDDEN, 2, HIDDEN, 256]: tanh on every hidden layer (so the code is in -1..1), sigmoid output. */
export function createAutoencoder(seed: number): MLP {
	return createMLP(SIZES, 'tanh', 'sigmoid', seed);
}

/** One dense layer: out[j] = act(b[j] + sum_i W[j * nIn + i] * input[i]). */
function layer(net: MLP, l: number, input: ArrayLike<number>, out: ArrayLike<number> & { [i: number]: number }): void {
	const nIn = net.sizes[l];
	const nOut = net.sizes[l + 1];
	const act = l === net.W.length - 1 ? net.output : net.hidden;
	const w = net.W[l];
	const b = net.b[l];
	for (let j = 0; j < nOut; j++) {
		let s = b[j];
		const row = j * nIn;
		for (let i = 0; i < nIn; i++) s += w[row + i] * input[i];
		out[j] = actFn(act, s);
	}
}

/** The code of a drawing: the two tanh outputs of layer 2 (runs weight layers 0 and 1 only). */
export function encode(net: MLP, image: ArrayLike<number>): [number, number] {
	const h = new Float64Array(net.sizes[1]);
	const c = new Float64Array(LATENT);
	layer(net, 0, image, h);
	layer(net, 1, h, c);
	return [c[0], c[1]];
}

/** The drawing the decoder makes from a code (runs weight layers 2 and 3 only). Writes into `out` when given, else a new array. Values in 0..1. */
export function decode(net: MLP, code: readonly [number, number], out: Float32Array = new Float32Array(IMAGE)): Float32Array {
	const h = new Float64Array(net.sizes[3]);
	layer(net, 2, code, h);
	layer(net, 3, h, out);
	return out;
}

/** All weights and biases (W0, b0, W1, b1, W2, b2, W3, b3, in that order) as float32, base64-encoded. */
export function packWeights(net: MLP): string {
	let total = 0;
	for (let l = 0; l < net.W.length; l++) total += net.W[l].length + net.b[l].length;
	const flat = new Float32Array(total);
	let o = 0;
	for (let l = 0; l < net.W.length; l++) {
		flat.set(net.W[l], o);
		o += net.W[l].length;
		flat.set(net.b[l], o);
		o += net.b[l].length;
	}
	const bytes = new Uint8Array(flat.buffer);
	let bin = '';
	for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
	return btoa(bin);
}

/** The inverse of packWeights for an autoencoder with the sizes above. Throws if the length is wrong. */
export function unpackWeights(base64: string): MLP {
	const bin = atob(base64);
	const net = createAutoencoder(0);
	let total = 0;
	for (let l = 0; l < net.W.length; l++) total += net.W[l].length + net.b[l].length;
	if (bin.length !== total * 4) throw new Error(`Autoencoder weights need ${total * 4} bytes, got ${bin.length}`);
	const bytes = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
	const flat = new Float32Array(bytes.buffer);
	let o = 0;
	for (let l = 0; l < net.W.length; l++) {
		net.W[l].set(flat.subarray(o, o + net.W[l].length));
		o += net.W[l].length;
		net.b[l].set(flat.subarray(o, o + net.b[l].length));
		o += net.b[l].length;
	}
	return net;
}

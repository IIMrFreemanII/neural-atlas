import { describe, expect, it } from 'vitest';
import { createAutoencoder, decode, encode, IMAGE, packWeights, unpackWeights } from './autoencoder.ts';
import { forward } from './mlp.ts';
import { mulberry32 } from './rng.ts';

function randomNet() {
	const net = createAutoencoder(3);
	const rand = mulberry32(9);
	for (const v of [...net.W, ...net.b]) for (let i = 0; i < v.length; i++) v[i] += (rand() - 0.5) * 0.2;
	return net;
}

describe('autoencoder', () => {
	it('encode and decode agree with forward', () => {
		const net = randomNet();
		const rand = mulberry32(1);
		const x = Float32Array.from({ length: IMAGE }, () => rand());
		const t = forward(net, x);
		const code = encode(net, x);
		expect(Math.abs(code[0] - t.a[2][0])).toBeLessThan(1e-5);
		expect(Math.abs(code[1] - t.a[2][1])).toBeLessThan(1e-5);
		const img = decode(net, code);
		let worst = 0;
		for (let i = 0; i < IMAGE; i++) worst = Math.max(worst, Math.abs(img[i] - t.a[4][i]));
		expect(worst).toBeLessThan(1e-5);
	});
	it('packs and unpacks weights to float32 precision', () => {
		const net = randomNet();
		const back = unpackWeights(packWeights(net));
		for (let l = 0; l < net.W.length; l++) {
			expect(Array.from(back.W[l])).toEqual(Array.from(Float32Array.from(net.W[l])));
			expect(Array.from(back.b[l])).toEqual(Array.from(Float32Array.from(net.b[l])));
		}
		expect(back.hidden).toBe('tanh');
		expect(back.output).toBe('sigmoid');
		expect(() => unpackWeights(btoa('abcd'))).toThrow();
	});
	it('decodes to 256 values in 0..1 and writes into out', () => {
		const net = randomNet();
		const img = decode(net, [0.5, -0.7]);
		expect(img.length).toBe(256);
		for (const v of img) {
			expect(v).toBeGreaterThanOrEqual(0);
			expect(v).toBeLessThanOrEqual(1);
		}
		const out = new Float32Array(256);
		expect(decode(net, [0.5, -0.7], out)).toBe(out);
		expect(Array.from(out)).toEqual(Array.from(img));
	});
});

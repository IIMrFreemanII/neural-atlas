/**
 * 2D convolution on small single-channel images, as deep learning uses the word (cross-correlation:
 * the kernel is not flipped). Pure TypeScript, no DOM, no GPU.
 */

/** A single-channel image or kernel, row-major, row 0 on top. */
export interface Grid {
	w: number;
	h: number;
	data: Float32Array;
}

export function makeGrid(w: number, h: number, data?: ArrayLike<number>): Grid {
	const out = new Float32Array(w * h);
	if (data) {
		const n = Math.min(out.length, data.length);
		for (let i = 0; i < n; i++) out[i] = data[i];
	}
	return { w, h, data: out };
}

/** Output size along one axis: floor((n + 2 * pad - k) / stride) + 1. */
export function outSize(n: number, k: number, stride: number, pad: number): number {
	return Math.max(0, Math.floor((n + 2 * pad - k) / stride) + 1);
}

export interface ConvOptions {
	/** Step between positions. Default 1. */
	stride?: number;
	/** Zero padding on every side, in pixels. Default 0. */
	pad?: number;
}

function pixel(g: Grid, row: number, col: number): number {
	return row < 0 || col < 0 || row >= g.h || col >= g.w ? 0 : g.data[row * g.w + col];
}

/**
 * Cross-correlation (what deep learning calls convolution):
 * out(r, c) = sum over i, j of input(r*stride + i - pad, c*stride + j - pad) * kernel(i, j), with zeros outside the image.
 */
export function conv2d(input: Grid, kernel: Grid, opts: ConvOptions = {}): Grid {
	const stride = opts.stride ?? 1;
	const pad = opts.pad ?? 0;
	const ow = outSize(input.w, kernel.w, stride, pad);
	const oh = outSize(input.h, kernel.h, stride, pad);
	const out = makeGrid(ow, oh);
	for (let r = 0; r < oh; r++) {
		for (let c = 0; c < ow; c++) {
			let sum = 0;
			for (let i = 0; i < kernel.h; i++) {
				for (let j = 0; j < kernel.w; j++) {
					sum += pixel(input, r * stride + i - pad, c * stride + j - pad) * kernel.data[i * kernel.w + j];
				}
			}
			out.data[r * ow + c] = sum;
		}
	}
	return out;
}

/** The input patch under the kernel for output cell (row, col), as a kernel-sized Grid (zeros outside the image). */
export function patchAt(input: Grid, kernel: Grid, row: number, col: number, opts: ConvOptions = {}): Grid {
	const stride = opts.stride ?? 1;
	const pad = opts.pad ?? 0;
	const out = makeGrid(kernel.w, kernel.h);
	for (let i = 0; i < kernel.h; i++) {
		for (let j = 0; j < kernel.w; j++) {
			out.data[i * kernel.w + j] = pixel(input, row * stride + i - pad, col * stride + j - pad);
		}
	}
	return out;
}

export function relu(g: Grid): Grid {
	const out = makeGrid(g.w, g.h);
	for (let i = 0; i < g.data.length; i++) out.data[i] = Math.max(0, g.data[i]);
	return out;
}

/** Non-overlapping max pooling: windows of size x size, stride = size, incomplete windows dropped. */
export function maxPool(g: Grid, size = 2): Grid {
	const ow = Math.floor(g.w / size);
	const oh = Math.floor(g.h / size);
	const out = makeGrid(ow, oh);
	for (let r = 0; r < oh; r++) {
		for (let c = 0; c < ow; c++) {
			let m = -Infinity;
			for (let i = 0; i < size; i++) {
				for (let j = 0; j < size; j++) m = Math.max(m, g.data[(r * size + i) * g.w + c * size + j]);
			}
			out.data[r * ow + c] = m;
		}
	}
	return out;
}

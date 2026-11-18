import type { SceneContext } from '../types.ts';
import { createShader } from '../util.ts';
import src from '../shaders/lines.wgsl?raw';

const FLOATS = 16;

export interface SegmentStyle {
	a: [number, number, number?];
	b: [number, number, number?];
	/** CSS px. */
	width: number;
	color: [number, number, number, number];
	t0?: number;
	t1?: number;
	pulseSpeed?: number;
	pulseCount?: number;
	pulseGain?: number;
}

/** Cumulative length fraction (0..1) at each point of a polyline. All 0 for a degenerate polyline. */
export function lengthFractions(pts: ReadonlyArray<readonly [number, number]>): number[] {
	const out: number[] = [0];
	let total = 0;
	for (let i = 1; i < pts.length; i++) {
		total += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
		out.push(total);
	}
	return total > 0 ? out.map((v) => v / total) : out.map(() => 0);
}

/** Points on a cubic bezier, inclusive of both ends: returns segments + 1 points. */
export function bezier(
	p0: readonly [number, number],
	c0: readonly [number, number],
	c1: readonly [number, number],
	p1: readonly [number, number],
	segments: number
): Array<[number, number]> {
	const n = Math.max(1, Math.floor(segments));
	const out: Array<[number, number]> = [];
	for (let i = 0; i <= n; i++) {
		const t = i / n;
		const u = 1 - t;
		const w0 = u * u * u;
		const w1 = 3 * u * u * t;
		const w2 = 3 * u * t * t;
		const w3 = t * t * t;
		out.push([
			w0 * p0[0] + w1 * c0[0] + w2 * c1[0] + w3 * p1[0],
			w0 * p0[1] + w1 * c0[1] + w2 * c1[1] + w3 * p1[1]
		]);
	}
	return out;
}

export class Lines {
	readonly capacity: number;
	readonly data: Float32Array<ArrayBuffer>;
	count = 0;
	private device: GPUDevice;
	private view: SceneContext['view'];
	private buffer: GPUBuffer;
	private pipeline: GPURenderPipeline;

	constructor(
		ctx: Pick<SceneContext, 'gpu' | 'target' | 'view'>,
		opts: { capacity: number; blend?: 'alpha' | 'additive'; depthTest?: boolean }
	) {
		const device = ctx.gpu.device;
		this.device = device;
		this.view = ctx.view;
		this.capacity = opts.capacity;
		this.data = new Float32Array(opts.capacity * FLOATS);
		this.buffer = device.createBuffer({
			label: 'lines instances',
			size: Math.max(16, this.data.byteLength),
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
		});
		const module = createShader(device, 'lines shader', src);
		const additive = opts.blend === 'additive';
		const dst: GPUBlendFactor = additive ? 'one' : 'one-minus-src-alpha';
		this.pipeline = device.createRenderPipeline({
			label: additive ? 'lines pipeline (additive)' : 'lines pipeline (alpha)',
			layout: device.createPipelineLayout({
				label: 'lines pipeline layout',
				bindGroupLayouts: [ctx.view.bindGroupLayout]
			}),
			vertex: {
				module,
				entryPoint: 'vs',
				buffers: [
					{
						arrayStride: FLOATS * 4,
						stepMode: 'instance',
						attributes: [
							{ shaderLocation: 0, offset: 0, format: 'float32x4' },
							{ shaderLocation: 1, offset: 16, format: 'float32x4' },
							{ shaderLocation: 2, offset: 32, format: 'float32x4' },
							{ shaderLocation: 3, offset: 48, format: 'float32x4' }
						]
					}
				]
			},
			fragment: {
				module,
				entryPoint: 'fs',
				targets: [
					{
						format: ctx.target.format,
						blend: {
							color: { srcFactor: 'one', dstFactor: dst, operation: 'add' },
							alpha: { srcFactor: 'one', dstFactor: dst, operation: 'add' }
						}
					}
				]
			},
			primitive: { topology: 'triangle-list' },
			depthStencil: {
				format: ctx.target.depthFormat,
				depthWriteEnabled: false,
				depthCompare: opts.depthTest ? 'less-equal' : 'always'
			},
			multisample: { count: ctx.target.sampleCount }
		});
	}

	set(i: number, s: SegmentStyle): void {
		const d = this.data;
		const o = i * FLOATS;
		d[o] = s.a[0];
		d[o + 1] = s.a[1];
		d[o + 2] = s.a[2] ?? 0;
		d[o + 3] = s.width;
		d[o + 4] = s.b[0];
		d[o + 5] = s.b[1];
		d[o + 6] = s.b[2] ?? 0;
		d[o + 7] = s.t0 ?? 0;
		d[o + 8] = s.color[0];
		d[o + 9] = s.color[1];
		d[o + 10] = s.color[2];
		d[o + 11] = s.color[3];
		d[o + 12] = s.pulseSpeed ?? 0;
		d[o + 13] = s.pulseCount ?? 0;
		d[o + 14] = s.pulseGain ?? 0;
		d[o + 15] = s.t1 ?? 1;
	}

	setPolyline(
		start: number,
		pts: ReadonlyArray<readonly [number, number]>,
		style: Omit<SegmentStyle, 'a' | 'b' | 't0' | 't1'>
	): number {
		const fr = lengthFractions(pts);
		for (let i = 0; i + 1 < pts.length; i++) {
			this.set(start + i, {
				...style,
				a: [pts[i][0], pts[i][1]],
				b: [pts[i + 1][0], pts[i + 1][1]],
				t0: fr[i],
				t1: fr[i + 1]
			});
		}
		return start + Math.max(0, pts.length - 1);
	}

	upload(): void {
		const n = Math.min(this.count, this.capacity) * FLOATS;
		if (n > 0) this.device.queue.writeBuffer(this.buffer, 0, this.data, 0, n);
	}

	draw(pass: GPURenderPassEncoder): void {
		const n = Math.min(this.count, this.capacity);
		if (n === 0) return;
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.view.bindGroup);
		pass.setVertexBuffer(0, this.buffer);
		pass.draw(6, n);
	}

	destroy(): void {
		this.buffer.destroy();
	}
}

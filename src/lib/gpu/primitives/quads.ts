import type { SceneContext } from '../types.ts';
import { createShader } from '../util.ts';
import src from '../shaders/quads.wgsl?raw';

const FLOATS = 12;

export interface QuadStyle {
	/** World center. */
	x: number;
	y: number;
	z?: number;
	/** World size. */
	w: number;
	h: number;
	color: [number, number, number, number];
	/** Corner radius in CSS px. Default 0. */
	radius?: number;
}

/** Instanced world-space rectangles with antialiased edges and rounded corners. */
export class Quads {
	readonly capacity: number;
	/** capacity * 12 floats: x,y,z,radius, w,h,0,0, r,g,b,a */
	readonly data: Float32Array<ArrayBuffer>;
	count = 0;
	private device: GPUDevice;
	private view: SceneContext['view'];
	private buffer: GPUBuffer;
	private pipeline: GPURenderPipeline;

	constructor(
		ctx: Pick<SceneContext, 'gpu' | 'target' | 'view'>,
		opts: {
			capacity: number;
			/** @deprecated 'additive' draws exactly like 'alpha' since the paper redesign; remove glow layers instead. */
			blend?: 'alpha' | 'additive';
		}
	) {
		const device = ctx.gpu.device;
		this.device = device;
		this.view = ctx.view;
		this.capacity = opts.capacity;
		this.data = new Float32Array(opts.capacity * FLOATS);
		this.buffer = device.createBuffer({
			label: 'quads instances',
			size: Math.max(16, this.data.byteLength),
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
		});
		const module = createShader(device, 'quads shader', src);
		const dst: GPUBlendFactor = 'one-minus-src-alpha';
		this.pipeline = device.createRenderPipeline({
			label: 'quads pipeline (alpha)',
			layout: device.createPipelineLayout({
				label: 'quads pipeline layout',
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
							{ shaderLocation: 2, offset: 32, format: 'float32x4' }
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
				depthCompare: 'always'
			},
			multisample: { count: ctx.target.sampleCount }
		});
	}

	set(i: number, q: QuadStyle): void {
		const d = this.data;
		const o = i * FLOATS;
		d[o] = q.x;
		d[o + 1] = q.y;
		d[o + 2] = q.z ?? 0;
		d[o + 3] = q.radius ?? 0;
		d[o + 4] = q.w;
		d[o + 5] = q.h;
		d[o + 6] = 0;
		d[o + 7] = 0;
		d[o + 8] = q.color[0];
		d[o + 9] = q.color[1];
		d[o + 10] = q.color[2];
		d[o + 11] = q.color[3];
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

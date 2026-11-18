import type { SceneContext } from '../types.ts';
import { createShader } from '../util.ts';
import src from '../shaders/points.wgsl?raw';

const FLOATS = 12;

export interface PointStyle {
	x: number;
	y: number;
	z?: number;
	/** CSS px, screen space. */
	radius: number;
	color: [number, number, number, number];
	/** 0 = filled disc; > 0 = ring stroke width in CSS px. */
	ring?: number;
	/** 0 = crisp edge; 1 = gaussian-like glow sprite. */
	softness?: number;
}

export class Points {
	readonly capacity: number;
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
			depthTest?: boolean;
		}
	) {
		const device = ctx.gpu.device;
		this.device = device;
		this.view = ctx.view;
		this.capacity = opts.capacity;
		this.data = new Float32Array(opts.capacity * FLOATS);
		this.buffer = device.createBuffer({
			label: 'points instances',
			size: Math.max(16, this.data.byteLength),
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
		});
		const module = createShader(device, 'points shader', src);
		const dst: GPUBlendFactor = 'one-minus-src-alpha';
		this.pipeline = device.createRenderPipeline({
			label: 'points pipeline (alpha)',
			layout: device.createPipelineLayout({
				label: 'points pipeline layout',
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
							color: {
								srcFactor: 'one',
								dstFactor: dst,
								operation: 'add'
							},
							alpha: {
								srcFactor: 'one',
								dstFactor: dst,
								operation: 'add'
							}
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

	set(i: number, p: PointStyle): void {
		const d = this.data;
		const o = i * FLOATS;
		d[o] = p.x;
		d[o + 1] = p.y;
		d[o + 2] = p.z ?? 0;
		d[o + 3] = p.radius;
		d[o + 4] = p.color[0];
		d[o + 5] = p.color[1];
		d[o + 6] = p.color[2];
		d[o + 7] = p.color[3];
		d[o + 8] = p.ring ?? 0;
		d[o + 9] = p.softness ?? 0;
		d[o + 10] = 0;
		d[o + 11] = 0;
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

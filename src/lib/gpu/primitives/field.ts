import type { SceneContext } from '../types.ts';
import { createShader } from '../util.ts';
import src from '../shaders/field.wgsl?raw';

const UNIFORM_FLOATS = 24;

/** A scalar grid drawn as a smooth two-color image with an optional iso-line. */
export class Field {
	readonly cols: number;
	readonly rows: number;
	/** cols * rows values, index = row * cols + col. Column 0 is at x0, row 0 is at y0 (the bottom). */
	readonly data: Float32Array<ArrayBuffer>;
	/** World rectangle covered. */
	rect: [x0: number, y0: number, x1: number, y1: number] = [-1, -1, 1, 1];
	colorLo: [number, number, number, number] = [0, 0, 0, 1];
	colorHi: [number, number, number, number] = [1, 1, 1, 1];
	opacity = 1;
	/** Value at which to draw a line; -1 for none. */
	iso = 0.5;
	isoColor: [number, number, number, number] = [1, 1, 1, 0.6];
	/** CSS px. */
	isoWidth = 1.5;

	private device: GPUDevice;
	private view: SceneContext['view'];
	private texture: GPUTexture;
	private uniform: GPUBuffer;
	private uniformData = new Float32Array(UNIFORM_FLOATS);
	private bindGroup: GPUBindGroup;
	private pipeline: GPURenderPipeline;

	constructor(ctx: Pick<SceneContext, 'gpu' | 'target' | 'view'>, opts: { cols: number; rows: number }) {
		const device = ctx.gpu.device;
		this.device = device;
		this.view = ctx.view;
		this.cols = opts.cols;
		this.rows = opts.rows;
		this.data = new Float32Array(opts.cols * opts.rows);
		this.texture = device.createTexture({
			label: 'field texture',
			size: [opts.cols, opts.rows],
			format: 'r32float',
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
		});
		this.uniform = device.createBuffer({
			label: 'field uniforms',
			size: UNIFORM_FLOATS * 4,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		const bgl = device.createBindGroupLayout({
			label: 'field bgl',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }
			]
		});
		this.bindGroup = device.createBindGroup({
			label: 'field bind group',
			layout: bgl,
			entries: [
				{ binding: 0, resource: { buffer: this.uniform } },
				{ binding: 1, resource: this.texture.createView() }
			]
		});
		const module = createShader(device, 'field shader', src);
		this.pipeline = device.createRenderPipeline({
			label: 'field pipeline',
			layout: device.createPipelineLayout({
				label: 'field pipeline layout',
				bindGroupLayouts: [ctx.view.bindGroupLayout, bgl]
			}),
			vertex: { module, entryPoint: 'vs' },
			fragment: {
				module,
				entryPoint: 'fs',
				targets: [
					{
						format: ctx.target.format,
						blend: {
							color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
							alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
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

	upload(): void {
		this.device.queue.writeTexture(
			{ texture: this.texture },
			this.data,
			{ bytesPerRow: this.cols * 4, rowsPerImage: this.rows },
			[this.cols, this.rows]
		);
		const u = this.uniformData;
		u.set(this.rect, 0);
		u.set(this.colorLo, 4);
		u.set(this.colorHi, 8);
		u.set(this.isoColor, 12);
		u[16] = this.opacity;
		u[17] = this.iso;
		u[18] = this.isoWidth;
		u[20] = this.cols;
		u[21] = this.rows;
		this.device.queue.writeBuffer(this.uniform, 0, u);
	}

	draw(pass: GPURenderPassEncoder): void {
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.view.bindGroup);
		pass.setBindGroup(1, this.bindGroup);
		pass.draw(6);
	}

	destroy(): void {
		this.texture.destroy();
		this.uniform.destroy();
	}
}

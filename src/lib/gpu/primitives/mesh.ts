import type { SceneContext } from '../types.ts';
import { createShader } from '../util.ts';
import src from '../shaders/mesh.wgsl?raw';

const UNIFORM_FLOATS = 20;

/** Triangle-list indices for a cols x rows vertex grid: (cols-1)*(rows-1)*6 indices. */
export function buildGridIndices(cols: number, rows: number): Uint32Array {
	const out = new Uint32Array(Math.max(0, cols - 1) * Math.max(0, rows - 1) * 6);
	let n = 0;
	for (let r = 0; r + 1 < rows; r++) {
		for (let c = 0; c + 1 < cols; c++) {
			const i = r * cols + c;
			out[n++] = i;
			out[n++] = i + 1;
			out[n++] = i + cols;
			out[n++] = i + 1;
			out[n++] = i + cols + 1;
			out[n++] = i + cols;
		}
	}
	return out;
}

/** A 3D height surface on a regular XZ grid. */
export class SurfaceMesh {
	readonly cols: number;
	readonly rows: number;
	/** World Y of each vertex, index = row * cols + col. Columns run along X, rows along Z. */
	readonly heights: Float32Array<ArrayBuffer>;
	rect: [x0: number, z0: number, x1: number, z1: number] = [-1, -1, 1, 1];
	/** Heights mapped to t = 0..1 for coloring. */
	range: [lo: number, hi: number] = [0, 1];
	colorLo: [number, number, number] = [0.03, 0.04, 0.16];
	colorHi: [number, number, number] = [0.12, 0.32, 0.45];
	/** Number of iso-height contour lines across range; 0 = none. */
	bands = 12;
	opacity = 1;

	private device: GPUDevice;
	private view: SceneContext['view'];
	private heightBuffer: GPUBuffer;
	private indexBuffer: GPUBuffer;
	private indexCount: number;
	private uniform: GPUBuffer;
	private uniformData = new Float32Array(UNIFORM_FLOATS);
	private bindGroup: GPUBindGroup;
	private pipeline: GPURenderPipeline;

	constructor(ctx: Pick<SceneContext, 'gpu' | 'target' | 'view'>, opts: { cols: number; rows: number }) {
		if (opts.cols < 2 || opts.rows < 2) throw new Error('SurfaceMesh needs cols >= 2 and rows >= 2');
		const device = ctx.gpu.device;
		this.device = device;
		this.view = ctx.view;
		this.cols = opts.cols;
		this.rows = opts.rows;
		this.heights = new Float32Array(opts.cols * opts.rows);
		this.heightBuffer = device.createBuffer({
			label: 'mesh heights',
			size: this.heights.byteLength,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
		});
		const indices = buildGridIndices(opts.cols, opts.rows);
		this.indexCount = indices.length;
		this.indexBuffer = device.createBuffer({
			label: 'mesh indices',
			size: indices.byteLength,
			usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(this.indexBuffer, 0, indices);
		this.uniform = device.createBuffer({
			label: 'mesh uniforms',
			size: UNIFORM_FLOATS * 4,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		const bgl = device.createBindGroupLayout({
			label: 'mesh bgl',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
			]
		});
		this.bindGroup = device.createBindGroup({
			label: 'mesh bind group',
			layout: bgl,
			entries: [{ binding: 0, resource: { buffer: this.uniform } }]
		});
		const module = createShader(device, 'mesh shader', src);
		this.pipeline = device.createRenderPipeline({
			label: 'mesh pipeline',
			layout: device.createPipelineLayout({
				label: 'mesh pipeline layout',
				bindGroupLayouts: [ctx.view.bindGroupLayout, bgl]
			}),
			vertex: {
				module,
				entryPoint: 'vs',
				buffers: [
					{
						arrayStride: 4,
						stepMode: 'vertex',
						attributes: [{ shaderLocation: 0, offset: 0, format: 'float32' }]
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
							color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
							alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
						}
					}
				]
			},
			primitive: { topology: 'triangle-list', cullMode: 'none' },
			depthStencil: {
				format: ctx.target.depthFormat,
				depthWriteEnabled: true,
				depthCompare: 'less'
			},
			multisample: { count: ctx.target.sampleCount }
		});
	}

	upload(): void {
		this.device.queue.writeBuffer(this.heightBuffer, 0, this.heights);
		const u = this.uniformData;
		u.set(this.rect, 0);
		u.set(this.colorLo, 4);
		u.set(this.colorHi, 8);
		u[12] = this.range[0];
		u[13] = this.range[1];
		u[14] = this.bands;
		u[15] = this.opacity;
		u[16] = this.cols;
		u[17] = this.rows;
		this.device.queue.writeBuffer(this.uniform, 0, u);
	}

	draw(pass: GPURenderPassEncoder): void {
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.view.bindGroup);
		pass.setBindGroup(1, this.bindGroup);
		pass.setVertexBuffer(0, this.heightBuffer);
		pass.setIndexBuffer(this.indexBuffer, 'uint32');
		pass.drawIndexed(this.indexCount);
	}

	destroy(): void {
		this.heightBuffer.destroy();
		this.indexBuffer.destroy();
		this.uniform.destroy();
	}
}

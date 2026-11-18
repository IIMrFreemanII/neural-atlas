import type { SceneContext } from '../types.ts';
import { createShader } from '../util.ts';
import type { Vec3 } from '../space.ts';
import src from '../shaders/sheets.wgsl?raw';

const FLOATS = 20;

export type RGBA = [number, number, number, number];

export interface SheetSpec {
	/** Corners are origin, origin + u, origin + v, origin + u + v. */
	origin: Vec3;
	u: Vec3;
	v: Vec3;
	/** Paper colour, linear straight RGBA. */
	fill: RGBA;
	/** Border colour, linear straight RGBA. */
	edge: RGBA;
	/** Border width in CSS px, drawn inside the sheet. Default 1. */
	edgePx?: number;
	texture?: SheetTexture | null;
	/** Multiplies the whole sheet, 0..1. Default 1. */
	opacity?: number;
}

function linearToSrgb(c: number): number {
	const x = Math.max(0, Math.min(1, c));
	return x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
}

/** Encode a texel colour function into 'rgba8unorm-srgb' bytes (rgb sRGB-encoded, alpha linear). x runs along u, y along v; y = 0 is the origin edge. */
export function encodeSheetTexels(width: number, height: number, colors: (x: number, y: number) => RGBA): Uint8Array {
	const out = new Uint8Array(width * height * 4);
	let o = 0;
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			const c = colors(x, y);
			out[o++] = Math.round(linearToSrgb(c[0]) * 255);
			out[o++] = Math.round(linearToSrgb(c[1]) * 255);
			out[o++] = Math.round(linearToSrgb(c[2]) * 255);
			out[o++] = Math.round(Math.max(0, Math.min(1, c[3])) * 255);
		}
	}
	return out;
}

/** A small colour grid laid on a sheet, sampled with nearest filtering. Create with Sheets.createTexture. */
export class SheetTexture {
	readonly width: number;
	readonly height: number;
	/** @internal */
	readonly bindGroup: GPUBindGroup;
	private texture: GPUTexture;
	private device: GPUDevice;

	/** @internal */
	constructor(device: GPUDevice, layout: GPUBindGroupLayout, sampler: GPUSampler, width: number, height: number) {
		this.device = device;
		this.width = width;
		this.height = height;
		this.texture = device.createTexture({
			label: 'sheet texture',
			size: [width, height],
			format: 'rgba8unorm-srgb',
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
		});
		this.bindGroup = device.createBindGroup({
			label: 'sheet texture group',
			layout,
			entries: [
				{ binding: 0, resource: this.texture.createView() },
				{ binding: 1, resource: sampler }
			]
		});
	}

	/** Fill the texture. Colours are linear straight RGBA, e.g. from tone() or diverging(). */
	write(colors: (x: number, y: number) => RGBA): void {
		const bytes = encodeSheetTexels(this.width, this.height, colors);
		this.device.queue.writeTexture(
			{ texture: this.texture },
			bytes as Uint8Array<ArrayBuffer>,
			{ bytesPerRow: this.width * 4 },
			[this.width, this.height]
		);
	}

	/** @internal */
	destroy(): void {
		this.texture.destroy();
	}
}

/** Oriented quads in 3D: sheets of paper with an optional colour grid. Draw order is index order. */
export class Sheets {
	readonly capacity: number;
	readonly data: Float32Array<ArrayBuffer>;
	count = 0;
	private device: GPUDevice;
	private view: SceneContext['view'];
	private buffer: GPUBuffer;
	private pipeline: GPURenderPipeline;
	private layout: GPUBindGroupLayout;
	private sampler: GPUSampler;
	private blank: SheetTexture;
	private textures: SheetTexture[] = [];
	private slots: Array<SheetTexture | null>;
	private dirty = true;

	constructor(ctx: Pick<SceneContext, 'gpu' | 'target' | 'view'>, opts: { capacity: number; depthWrite?: boolean }) {
		const device = ctx.gpu.device;
		this.device = device;
		this.view = ctx.view;
		this.capacity = opts.capacity;
		this.data = new Float32Array(opts.capacity * FLOATS);
		this.slots = new Array(opts.capacity).fill(null);
		this.buffer = device.createBuffer({
			label: 'sheets instances',
			size: Math.max(16, this.data.byteLength),
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
		});
		this.layout = device.createBindGroupLayout({
			label: 'sheets bgl',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } }
			]
		});
		this.sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });
		this.blank = new SheetTexture(device, this.layout, this.sampler, 1, 1);
		this.blank.write(() => [0, 0, 0, 0]);

		const module = createShader(device, 'sheets shader', src);
		const dst: GPUBlendFactor = 'one-minus-src-alpha';
		this.pipeline = device.createRenderPipeline({
			label: 'sheets pipeline',
			layout: device.createPipelineLayout({
				label: 'sheets pipeline layout',
				bindGroupLayouts: [ctx.view.bindGroupLayout, this.layout]
			}),
			vertex: {
				module,
				entryPoint: 'vs',
				buffers: [
					{
						arrayStride: FLOATS * 4,
						stepMode: 'instance',
						attributes: [0, 1, 2, 3, 4].map((i) => ({
							shaderLocation: i,
							offset: i * 16,
							format: 'float32x4' as GPUVertexFormat
						}))
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
			primitive: { topology: 'triangle-list', cullMode: 'none' },
			depthStencil: {
				format: ctx.target.depthFormat,
				depthWriteEnabled: opts.depthWrite ?? true,
				depthCompare: 'less-equal'
			},
			multisample: { count: ctx.target.sampleCount }
		});
	}

	createTexture(width: number, height: number): SheetTexture {
		const t = new SheetTexture(this.device, this.layout, this.sampler, width, height);
		this.textures.push(t);
		return t;
	}

	set(i: number, s: SheetSpec): void {
		const d = this.data;
		const o = i * FLOATS;
		d[o] = s.origin[0];
		d[o + 1] = s.origin[1];
		d[o + 2] = s.origin[2];
		d[o + 3] = s.edgePx ?? 1;
		d[o + 4] = s.u[0];
		d[o + 5] = s.u[1];
		d[o + 6] = s.u[2];
		d[o + 7] = s.opacity ?? 1;
		d[o + 8] = s.v[0];
		d[o + 9] = s.v[1];
		d[o + 10] = s.v[2];
		d[o + 11] = 0;
		d.set(s.fill, o + 12);
		d.set(s.edge, o + 16);
		this.slots[i] = s.texture ?? null;
		this.dirty = true;
	}

	/** Push instance data to the GPU. draw() does this itself when anything changed. */
	upload(): void {
		this.dirty = false;
		const n = Math.min(this.count, this.capacity) * FLOATS;
		if (n > 0) this.device.queue.writeBuffer(this.buffer, 0, this.data, 0, n);
	}

	draw(pass: GPURenderPassEncoder): void {
		const n = Math.min(this.count, this.capacity);
		if (n === 0) return;
		if (this.dirty) this.upload();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.view.bindGroup);
		pass.setVertexBuffer(0, this.buffer);
		for (let i = 0; i < n; i++) {
			pass.setBindGroup(1, (this.slots[i] ?? this.blank).bindGroup);
			pass.draw(6, 1, 0, i);
		}
	}

	dispose(): void {
		this.buffer.destroy();
		this.blank.destroy();
		for (const t of this.textures) t.destroy();
		this.textures = [];
	}
}

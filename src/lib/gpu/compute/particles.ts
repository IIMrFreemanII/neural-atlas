import type { SceneContext } from '../types.ts';
import { createShader } from '../util.ts';
import { mulberry32 } from '../../ml/rng.ts';
import src from '../shaders/particles.wgsl?raw';

export const MAX_GROUPS = 8;

export interface ParticleCloudOptions {
	capacity: number;
	/** Default 'alpha'. 'additive' is @deprecated: it draws exactly like 'alpha' since the paper redesign; remove glow layers instead. */
	blend?: 'alpha' | 'additive';
}

const STATE_FLOATS = 8; // pos.xy, group, seed, color.rgba
const SIM_FLOATS = 4 * 3 + MAX_GROUPS * 2 + MAX_GROUPS * 4; // rates, misc, neutral, centers, palette
const WG = 64;

/**
 * A GPU particle cloud. Every particle has a fixed target; a compute pass (run once per frame from step())
 * eases its drawn position toward the target, finds the nearest active group center to the target and eases
 * its color toward that group's palette color. A render pipeline draws every particle as a small disc.
 */
export class ParticleCloud {
	readonly capacity: number;
	/** Number of particles simulated and drawn (<= capacity). */
	count: number;
	/** Target xy pairs in world units. Call uploadTargets() after writing. */
	readonly targets: Float32Array<ArrayBuffer>;
	/** Group centers, xy pairs. */
	readonly centers = new Float32Array(MAX_GROUPS * 2);
	/** Active groups, 0..MAX_GROUPS. 0 = every particle eases to neutralColor. */
	groupCount = 0;
	/** Linear RGBA per group. */
	readonly palette = new Float32Array(MAX_GROUPS * 4);
	neutralColor: [number, number, number, number] = [0.3, 0.35, 0.5, 0.5];
	/** Disc radius in CSS px. */
	radius = 1.6;
	/** 0 = crisp, 1 = soft glow sprite. */
	softness = 0;
	/** Position easing rate in 1/s. */
	follow = 5;
	/** Color easing rate in 1/s. */
	colorFollow = 8;
	/** Amplitude in world units of a slow per-particle drift around the target. */
	jitter = 0;

	private device: GPUDevice;
	private view: SceneContext['view'];
	private stateBuf: GPUBuffer;
	private targetBuf: GPUBuffer;
	private teleportBuf: GPUBuffer;
	private simBuf: GPUBuffer;
	private styleBuf: GPUBuffer;
	private computePipeline: GPUComputePipeline;
	private computeGroup: GPUBindGroup;
	private renderPipeline: GPURenderPipeline;
	private styleGroup: GPUBindGroup;
	private simData = new Float32Array(SIM_FLOATS);
	private styleData = new Float32Array(4);
	private teleport: Float32Array<ArrayBuffer>;
	private teleportCount = 0;

	constructor(
		ctx: Pick<SceneContext, 'gpu' | 'target' | 'view'>,
		opts: ParticleCloudOptions
	) {
		const device = ctx.gpu.device;
		this.device = device;
		this.view = ctx.view;
		this.capacity = opts.capacity;
		this.count = opts.capacity;
		this.targets = new Float32Array(opts.capacity * 2);
		this.teleport = new Float32Array(opts.capacity * 2);

		const bytes = (floats: number) => Math.max(16, floats * 4);
		this.stateBuf = device.createBuffer({
			label: 'particles state',
			size: bytes(opts.capacity * STATE_FLOATS),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
		});
		this.targetBuf = device.createBuffer({
			label: 'particles targets',
			size: bytes(opts.capacity * 2),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		this.teleportBuf = device.createBuffer({
			label: 'particles teleport',
			size: bytes(opts.capacity * 2),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		this.simBuf = device.createBuffer({
			label: 'particles sim uniforms',
			size: SIM_FLOATS * 4,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.styleBuf = device.createBuffer({
			label: 'particles style uniforms',
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		// Initial state: deterministic seeds, neutral color, position at the origin.
		const init = new Float32Array(opts.capacity * STATE_FLOATS);
		const rand = mulberry32(0x9e3779b1);
		for (let i = 0; i < opts.capacity; i++) {
			const o = i * STATE_FLOATS;
			init[o + 2] = -1;
			init[o + 3] = rand();
			init[o + 4] = this.neutralColor[0];
			init[o + 5] = this.neutralColor[1];
			init[o + 6] = this.neutralColor[2];
			init[o + 7] = this.neutralColor[3];
		}
		device.queue.writeBuffer(this.stateBuf, 0, init);

		const module = createShader(device, 'particles shader', src);

		const computeLayout = device.createBindGroupLayout({
			label: 'particles compute layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
				{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
			]
		});
		this.computePipeline = device.createComputePipeline({
			label: 'particles compute pipeline',
			layout: device.createPipelineLayout({ bindGroupLayouts: [computeLayout] }),
			compute: { module, entryPoint: 'simulate' }
		});
		this.computeGroup = device.createBindGroup({
			label: 'particles compute group',
			layout: computeLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.simBuf } },
				{ binding: 1, resource: { buffer: this.targetBuf } },
				{ binding: 2, resource: { buffer: this.stateBuf } },
				{ binding: 3, resource: { buffer: this.teleportBuf } }
			]
		});

		const styleLayout = device.createBindGroupLayout({
			label: 'particles style layout',
			entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }]
		});
		this.styleGroup = device.createBindGroup({
			layout: styleLayout,
			entries: [{ binding: 0, resource: { buffer: this.styleBuf } }]
		});
		const dst: GPUBlendFactor = 'one-minus-src-alpha';
		this.renderPipeline = device.createRenderPipeline({
			label: 'particles pipeline (alpha)',
			layout: device.createPipelineLayout({
				bindGroupLayouts: [ctx.view.bindGroupLayout, styleLayout]
			}),
			vertex: {
				module,
				entryPoint: 'vs',
				buffers: [
					{
						arrayStride: STATE_FLOATS * 4,
						stepMode: 'instance',
						attributes: [
							{ shaderLocation: 0, offset: 0, format: 'float32x4' },
							{ shaderLocation: 1, offset: 16, format: 'float32x4' }
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

	uploadTargets(): void {
		this.device.queue.writeBuffer(this.targetBuf, 0, this.targets, 0, this.capacity * 2);
	}

	/** Teleport particles: xy pairs for the first xy.length / 2 particles. Colors are kept. Applied on the next step(). */
	setPositions(xy: Float32Array): void {
		const n = Math.min(this.capacity, Math.floor(xy.length / 2));
		this.teleport.set(xy.subarray(0, n * 2));
		this.device.queue.writeBuffer(this.teleportBuf, 0, this.teleport, 0, n * 2);
		this.teleportCount = n;
	}

	/**
	 * Run the compute pass now (own encoder, submitted immediately). Call once per frame.
	 * With dt <= 0 and no pending teleport it returns without doing anything (a paused scene stays as it is).
	 */
	step(dt: number, time: number): void {
		if (dt <= 0 && this.teleportCount === 0) return;
		const n = Math.min(this.count, this.capacity);
		if (n <= 0) return;
		const d = this.simData;
		d[0] = Math.max(0, dt);
		d[1] = time;
		d[2] = this.follow;
		d[3] = this.colorFollow;
		d[4] = this.jitter;
		d[5] = Math.max(0, Math.min(MAX_GROUPS, this.groupCount));
		d[6] = n;
		d[7] = this.teleportCount;
		d.set(this.neutralColor, 8);
		d.set(this.centers, 12);
		d.set(this.palette, 12 + MAX_GROUPS * 2);
		this.device.queue.writeBuffer(this.simBuf, 0, d);
		this.teleportCount = 0;

		const encoder = this.device.createCommandEncoder({ label: 'particles step' });
		const pass = encoder.beginComputePass({ label: 'particles compute' });
		pass.setPipeline(this.computePipeline);
		pass.setBindGroup(0, this.computeGroup);
		pass.dispatchWorkgroups(Math.ceil(n / WG));
		pass.end();
		this.device.queue.submit([encoder.finish()]);
	}

	draw(pass: GPURenderPassEncoder): void {
		const n = Math.min(this.count, this.capacity);
		if (n === 0) return;
		this.styleData[0] = this.radius;
		this.styleData[1] = this.softness;
		this.device.queue.writeBuffer(this.styleBuf, 0, this.styleData);
		pass.setPipeline(this.renderPipeline);
		pass.setBindGroup(0, this.view.bindGroup);
		pass.setBindGroup(1, this.styleGroup);
		pass.setVertexBuffer(0, this.stateBuf);
		pass.draw(6, n);
	}

	destroy(): void {
		this.stateBuf.destroy();
		this.targetBuf.destroy();
		this.teleportBuf.destroy();
		this.simBuf.destroy();
		this.styleBuf.destroy();
	}
}

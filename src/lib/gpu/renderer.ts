import type { Gpu, TargetInfo } from './types.ts';
import { Bloom } from './post/bloom.ts';
import { VOID } from './colors.ts';
import { createShader } from './util.ts';
import compositeSrc from './shaders/composite.wgsl?raw';

const HDR: GPUTextureFormat = 'rgba16float';
const DEPTH: GPUTextureFormat = 'depth24plus';
const SAMPLES = 4;

export class Renderer {
	readonly target: TargetInfo = { format: HDR, depthFormat: DEPTH, sampleCount: SAMPLES };
	bloomStrength = 0.9;

	private gpu: Gpu;
	private canvas: HTMLCanvasElement;
	private context: GPUCanvasContext;
	private bloom: Bloom;
	private msaa: GPUTexture | null = null;
	private resolve: GPUTexture | null = null;
	private depth: GPUTexture | null = null;
	private msaaView: GPUTextureView | null = null;
	private resolveView: GPUTextureView | null = null;
	private depthView: GPUTextureView | null = null;
	private compositeBgl: GPUBindGroupLayout;
	private compositePipeline: GPURenderPipeline;
	private compositeGroup: GPUBindGroup | null = null;
	private paramsBuffer: GPUBuffer;
	private paramsData = new Float32Array(4);
	private sampler: GPUSampler;
	private pw = 0;
	private ph = 0;

	constructor(gpu: Gpu, canvas: HTMLCanvasElement) {
		this.gpu = gpu;
		this.canvas = canvas;
		const ctx = canvas.getContext('webgpu');
		if (!ctx) throw new Error('Could not create a WebGPU canvas context');
		this.context = ctx;
		ctx.configure({ device: gpu.device, format: gpu.format, alphaMode: 'opaque' });

		const device = gpu.device;
		this.bloom = new Bloom(gpu);
		this.sampler = device.createSampler({
			label: 'composite sampler',
			magFilter: 'linear',
			minFilter: 'linear',
			addressModeU: 'clamp-to-edge',
			addressModeV: 'clamp-to-edge'
		});
		this.paramsBuffer = device.createBuffer({
			label: 'composite params',
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.compositeBgl = device.createBindGroupLayout({
			label: 'composite bgl',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
				{ binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
			]
		});
		const module = createShader(device, 'composite shader', compositeSrc);
		this.compositePipeline = device.createRenderPipeline({
			label: 'composite pipeline',
			layout: device.createPipelineLayout({
				label: 'composite pipeline layout',
				bindGroupLayouts: [this.compositeBgl]
			}),
			vertex: { module, entryPoint: 'vs' },
			fragment: { module, entryPoint: 'fs', targets: [{ format: gpu.format }] },
			primitive: { topology: 'triangle-list' }
		});
	}

	resize(cssWidth: number, cssHeight: number, dpr: number): void {
		const max = this.gpu.device.limits.maxTextureDimension2D;
		const w = Math.min(max, Math.max(0, Math.round(cssWidth * dpr)));
		const h = Math.min(max, Math.max(0, Math.round(cssHeight * dpr)));
		if (w === this.pw && h === this.ph) return;
		this.pw = w;
		this.ph = h;
		this.canvas.width = w;
		this.canvas.height = h;
		this.releaseTargets();
		if (w === 0 || h === 0) return;

		const device = this.gpu.device;
		this.msaa = device.createTexture({
			label: 'scene msaa color',
			size: [w, h],
			format: HDR,
			sampleCount: SAMPLES,
			usage: GPUTextureUsage.RENDER_ATTACHMENT
		});
		this.resolve = device.createTexture({
			label: 'scene resolved color',
			size: [w, h],
			format: HDR,
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
		});
		this.depth = device.createTexture({
			label: 'scene depth',
			size: [w, h],
			format: DEPTH,
			sampleCount: SAMPLES,
			usage: GPUTextureUsage.RENDER_ATTACHMENT
		});
		this.msaaView = this.msaa.createView();
		this.resolveView = this.resolve.createView();
		this.depthView = this.depth.createView();
		this.bloom.resize(w, h, this.resolveView);
		this.compositeGroup = device.createBindGroup({
			label: 'composite bind group',
			layout: this.compositeBgl,
			entries: [
				{ binding: 0, resource: this.sampler },
				{ binding: 1, resource: this.resolveView },
				{ binding: 2, resource: this.bloom.output },
				{ binding: 3, resource: { buffer: this.paramsBuffer } }
			]
		});
	}

	render(draw: (pass: GPURenderPassEncoder) => void): void {
		if (!this.msaaView || !this.resolveView || !this.depthView || !this.compositeGroup) return;
		const device = this.gpu.device;
		this.paramsData[0] = this.bloomStrength;
		device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData);

		const encoder = device.createCommandEncoder({ label: 'frame' });
		const scene = encoder.beginRenderPass({
			label: 'scene pass',
			colorAttachments: [
				{
					view: this.msaaView,
					resolveTarget: this.resolveView,
					loadOp: 'clear',
					storeOp: 'discard',
					clearValue: { r: VOID[0], g: VOID[1], b: VOID[2], a: 1 }
				}
			],
			depthStencilAttachment: {
				view: this.depthView,
				depthLoadOp: 'clear',
				depthStoreOp: 'discard',
				depthClearValue: 1
			}
		});
		draw(scene);
		scene.end();

		this.bloom.encode(encoder);

		const composite = encoder.beginRenderPass({
			label: 'composite pass',
			colorAttachments: [
				{
					view: this.context.getCurrentTexture().createView(),
					loadOp: 'clear',
					storeOp: 'store',
					clearValue: [0, 0, 0, 1]
				}
			]
		});
		composite.setPipeline(this.compositePipeline);
		composite.setBindGroup(0, this.compositeGroup);
		composite.draw(3);
		composite.end();

		device.queue.submit([encoder.finish()]);
	}

	private releaseTargets(): void {
		this.msaa?.destroy();
		this.resolve?.destroy();
		this.depth?.destroy();
		this.msaa = this.resolve = this.depth = null;
		this.msaaView = this.resolveView = this.depthView = null;
		this.compositeGroup = null;
	}

	destroy(): void {
		this.releaseTargets();
		this.bloom.destroy();
		this.paramsBuffer.destroy();
		this.context.unconfigure();
	}
}

import type { Gpu, TargetInfo } from './types.ts';
import { createShader } from './util.ts';
import compositeSrc from './shaders/composite.wgsl?raw';

const HDR: GPUTextureFormat = 'rgba16float';
const DEPTH: GPUTextureFormat = 'depth24plus';
const SAMPLES = 4;

export class Renderer {
	readonly target: TargetInfo = { format: HDR, depthFormat: DEPTH, sampleCount: SAMPLES };

	private gpu: Gpu;
	private canvas: HTMLCanvasElement;
	private context: GPUCanvasContext;
	private msaa: GPUTexture | null = null;
	private resolve: GPUTexture | null = null;
	private depth: GPUTexture | null = null;
	private msaaView: GPUTextureView | null = null;
	private resolveView: GPUTextureView | null = null;
	private depthView: GPUTextureView | null = null;
	private compositeBgl: GPUBindGroupLayout;
	private compositePipeline: GPURenderPipeline;
	private compositeGroup: GPUBindGroup | null = null;
	private sampler: GPUSampler;
	private pw = 0;
	private ph = 0;

	constructor(gpu: Gpu, canvas: HTMLCanvasElement) {
		this.gpu = gpu;
		this.canvas = canvas;
		const ctx = canvas.getContext('webgpu');
		if (!ctx) throw new Error('Could not create a WebGPU canvas context');
		this.context = ctx;
		ctx.configure({ device: gpu.device, format: gpu.format, alphaMode: 'premultiplied' });

		const device = gpu.device;
		this.sampler = device.createSampler({
			label: 'composite sampler',
			magFilter: 'linear',
			minFilter: 'linear',
			addressModeU: 'clamp-to-edge',
			addressModeV: 'clamp-to-edge'
		});
		this.compositeBgl = device.createBindGroupLayout({
			label: 'composite bgl',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }
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
		this.compositeGroup = device.createBindGroup({
			label: 'composite bind group',
			layout: this.compositeBgl,
			entries: [
				{ binding: 0, resource: this.sampler },
				{ binding: 1, resource: this.resolveView }
			]
		});
	}

	render(draw: (pass: GPURenderPassEncoder) => void): void {
		if (!this.msaaView || !this.resolveView || !this.depthView || !this.compositeGroup) return;
		const device = this.gpu.device;

		const encoder = device.createCommandEncoder({ label: 'frame' });
		const scene = encoder.beginRenderPass({
			label: 'scene pass',
			colorAttachments: [
				{
					view: this.msaaView,
					resolveTarget: this.resolveView,
					loadOp: 'clear',
					storeOp: 'discard',
					clearValue: { r: 0, g: 0, b: 0, a: 0 }
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

		const composite = encoder.beginRenderPass({
			label: 'composite pass',
			colorAttachments: [
				{
					view: this.context.getCurrentTexture().createView(),
					loadOp: 'clear',
					storeOp: 'store',
					clearValue: [0, 0, 0, 0]
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
		this.context.unconfigure();
	}
}

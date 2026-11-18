import type { Gpu } from '../types.ts';
import { createShader } from '../util.ts';
import src from '../shaders/bloom.wgsl?raw';

const LEVELS = 5;
const FORMAT: GPUTextureFormat = 'rgba16float';

/** Soft-knee threshold + 5-level downsample chain + tent upsample. Result is level 0 (half resolution). */
export class Bloom {
	private device: GPUDevice;
	private bgl: GPUBindGroupLayout;
	private sampler: GPUSampler;
	private prefilter: GPURenderPipeline;
	private down: GPURenderPipeline;
	private up: GPURenderPipeline;
	private textures: GPUTexture[] = [];
	private views: GPUTextureView[] = [];
	private downGroups: GPUBindGroup[] = [];
	private upGroups: GPUBindGroup[] = [];

	constructor(gpu: Gpu) {
		const device = gpu.device;
		this.device = device;
		this.sampler = device.createSampler({
			label: 'bloom sampler',
			magFilter: 'linear',
			minFilter: 'linear',
			addressModeU: 'clamp-to-edge',
			addressModeV: 'clamp-to-edge'
		});
		this.bgl = device.createBindGroupLayout({
			label: 'bloom bgl',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }
			]
		});
		const layout = device.createPipelineLayout({
			label: 'bloom pipeline layout',
			bindGroupLayouts: [this.bgl]
		});
		const module = createShader(device, 'bloom shader', src);
		const make = (
			label: string,
			entry: string,
			blend?: GPUBlendState
		): GPURenderPipeline =>
			device.createRenderPipeline({
				label,
				layout,
				vertex: { module, entryPoint: 'vs' },
				fragment: { module, entryPoint: entry, targets: [{ format: FORMAT, blend }] },
				primitive: { topology: 'triangle-list' }
			});
		const add: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' };
		this.prefilter = make('bloom prefilter', 'fs_prefilter');
		this.down = make('bloom downsample', 'fs_down');
		this.up = make('bloom upsample', 'fs_up', { color: add, alpha: add });
	}

	get output(): GPUTextureView {
		return this.views[0];
	}

	resize(width: number, height: number, source: GPUTextureView): void {
		this.release();
		for (let i = 0; i < LEVELS; i++) {
			const w = Math.max(1, Math.floor(width / 2 ** (i + 1)));
			const h = Math.max(1, Math.floor(height / 2 ** (i + 1)));
			const tex = this.device.createTexture({
				label: `bloom level ${i}`,
				size: [w, h],
				format: FORMAT,
				usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
			});
			this.textures.push(tex);
			this.views.push(tex.createView({ label: `bloom level ${i} view` }));
		}
		const group = (label: string, view: GPUTextureView) =>
			this.device.createBindGroup({
				label,
				layout: this.bgl,
				entries: [
					{ binding: 0, resource: this.sampler },
					{ binding: 1, resource: view }
				]
			});
		// down[i] writes level i. down[0] is the prefilter reading the scene.
		this.downGroups = [group('bloom down 0', source)];
		for (let i = 1; i < LEVELS; i++) this.downGroups.push(group(`bloom down ${i}`, this.views[i - 1]));
		// up[i] writes level i reading level i + 1.
		this.upGroups = [];
		for (let i = 0; i < LEVELS - 1; i++) this.upGroups.push(group(`bloom up ${i}`, this.views[i + 1]));
	}

	encode(encoder: GPUCommandEncoder): void {
		if (this.views.length === 0) return;
		for (let i = 0; i < LEVELS; i++) {
			const pass = encoder.beginRenderPass({
				label: `bloom down pass ${i}`,
				colorAttachments: [
					{ view: this.views[i], loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }
				]
			});
			pass.setPipeline(i === 0 ? this.prefilter : this.down);
			pass.setBindGroup(0, this.downGroups[i]);
			pass.draw(3);
			pass.end();
		}
		for (let i = LEVELS - 2; i >= 0; i--) {
			const pass = encoder.beginRenderPass({
				label: `bloom up pass ${i}`,
				colorAttachments: [{ view: this.views[i], loadOp: 'load', storeOp: 'store' }]
			});
			pass.setPipeline(this.up);
			pass.setBindGroup(0, this.upGroups[i]);
			pass.draw(3);
			pass.end();
		}
	}

	private release(): void {
		for (const t of this.textures) t.destroy();
		this.textures = [];
		this.views = [];
		this.downGroups = [];
		this.upGroups = [];
	}

	destroy(): void {
		this.release();
	}
}

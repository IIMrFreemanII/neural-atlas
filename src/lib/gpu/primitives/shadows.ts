import { hexToLinear } from '../colors.ts';
import { LIGHT, type Vec3 } from '../space.ts';
import { palette } from '../theme.ts';
import type { SceneContext } from '../types.ts';
import { createShader } from '../util.ts';
import src from '../shaders/shadows.wgsl?raw';

const FLOATS = 16;

/** Horizontal direction a shadow falls along (away from the light), unit length. */
function shadowDir(): [number, number] {
	const h = Math.hypot(LIGHT[0], LIGHT[2]) || 1;
	return [-LIGHT[0] / h, -LIGHT[2] / h];
}

/** Four floor corners in ring order, whatever order they were given in (sorted by angle around the centroid). */
export function ringOrder(corners: ReadonlyArray<Readonly<Vec3>>): Vec3[] {
	const cx = corners.reduce((s, c) => s + c[0], 0) / corners.length;
	const cz = corners.reduce((s, c) => s + c[2], 0) / corners.length;
	return corners
		.map((c) => [...c] as Vec3)
		.sort((p, q) => Math.atan2(p[2] - cz, p[0] - cx) - Math.atan2(q[2] - cz, q[0] - cx));
}

/** Soft cast shadows on a floor plane. Colour and strength come from the palette at construction. */
export class Shadows {
	readonly capacity: number;
	readonly data: Float32Array<ArrayBuffer>;
	count = 0;
	private device: GPUDevice;
	private view: SceneContext['view'];
	private buffer: GPUBuffer;
	private uniform: GPUBuffer;
	private group: GPUBindGroup;
	private pipeline: GPURenderPipeline;
	private dirty = true;

	constructor(ctx: Pick<SceneContext, 'gpu' | 'target' | 'view'>, opts: { capacity: number }) {
		const device = ctx.gpu.device;
		this.device = device;
		this.view = ctx.view;
		this.capacity = opts.capacity;
		this.data = new Float32Array(opts.capacity * FLOATS);
		this.buffer = device.createBuffer({
			label: 'shadows instances',
			size: Math.max(16, this.data.byteLength),
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
		});
		const pal = palette();
		const [r, g, b] = hexToLinear(pal.cast);
		this.uniform = device.createBuffer({
			label: 'shadows color',
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(this.uniform, 0, new Float32Array([r, g, b, pal.castAlpha]));
		const layout = device.createBindGroupLayout({
			label: 'shadows bgl',
			entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }]
		});
		this.group = device.createBindGroup({
			label: 'shadows group',
			layout,
			entries: [{ binding: 0, resource: { buffer: this.uniform } }]
		});
		const module = createShader(device, 'shadows shader', src);
		const dst: GPUBlendFactor = 'one-minus-src-alpha';
		this.pipeline = device.createRenderPipeline({
			label: 'shadows pipeline',
			layout: device.createPipelineLayout({
				label: 'shadows pipeline layout',
				bindGroupLayouts: [ctx.view.bindGroupLayout, layout]
			}),
			vertex: {
				module,
				entryPoint: 'vs',
				buffers: [
					{
						arrayStride: FLOATS * 4,
						stepMode: 'instance',
						attributes: [0, 1, 2, 3].map((i) => ({
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
				depthWriteEnabled: false,
				depthCompare: 'less-equal',
				// Pull toward the camera so a shadow on the floor plane never z-fights the floor drawn there.
				depthBias: -4,
				depthBiasSlopeScale: -2
			},
			multisample: { count: ctx.target.sampleCount }
		});
	}

	/**
	 * A soft elliptical shadow under a sphere or point. `center` is on the floor (use projectToFloor).
	 * `radius` is the semi-minor radius at which the shadow has faded to nothing; the ellipse is stretched
	 * along the light's horizontal direction by 1 / squash (default: the light's elevation, LIGHT.y).
	 */
	blob(i: number, s: { center: Vec3; radius: number; squash?: number; alpha?: number }): void {
		const squash = Math.max(0.05, Math.min(1, s.squash ?? LIGHT[1]));
		const major = s.radius / squash;
		const [dx, dz] = shadowDir();
		this.write(i, [
			s.center[0], s.center[1], s.center[2], 0,
			dx, dz, major, s.radius,
			0, 0, 0, 0,
			s.alpha ?? 1, 0, major, 0
		]);
	}

	/**
	 * The shadow of a sheet. `corners` are the four floor-plane corners (e.g. via projectToFloor), in any order;
	 * `blur` is the edge softness in world units.
	 */
	quad(i: number, s: { corners: [Vec3, Vec3, Vec3, Vec3]; blur: number; alpha?: number }): void {
		const c = ringOrder(s.corners);
		const cx = (c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4;
		const cy = (c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4;
		const cz = (c[0][2] + c[1][2] + c[2][2] + c[3][2]) / 4;
		let ext = 0;
		for (const p of c) ext = Math.max(ext, Math.abs(p[0] - cx), Math.abs(p[2] - cz));
		const blur = Math.max(s.blur, 1e-4);
		this.write(i, [
			cx, cy, cz, 1,
			c[0][0], c[0][2], c[1][0], c[1][2],
			c[2][0], c[2][2], c[3][0], c[3][2],
			s.alpha ?? 1, blur, ext + 2 * blur, 0
		]);
	}

	private write(i: number, f: number[]): void {
		this.data.set(f, i * FLOATS);
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
		pass.setBindGroup(1, this.group);
		pass.setVertexBuffer(0, this.buffer);
		pass.draw(6, n);
	}

	dispose(): void {
		this.buffer.destroy();
		this.uniform.destroy();
	}
}

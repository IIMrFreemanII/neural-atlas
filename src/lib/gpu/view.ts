import type { Gpu, View } from './types.ts';

const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/** World -> CSS pixels (top-left origin) using a column-major viewProj. */
export function projectToCss(
	m: ArrayLike<number>,
	width: number,
	height: number,
	x: number,
	y: number,
	z = 0
): [number, number] {
	const cx = m[0] * x + m[4] * y + m[8] * z + m[12];
	const cy = m[1] * x + m[5] * y + m[9] * z + m[13];
	const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
	const iw = cw !== 0 ? 1 / cw : 0;
	return [(cx * iw * 0.5 + 0.5) * width, (1 - (cy * iw * 0.5 + 0.5)) * height];
}

/** CSS pixels -> world position on the z = 0 plane. */
export function unprojectFromCss(
	m: ArrayLike<number>,
	width: number,
	height: number,
	cssX: number,
	cssY: number
): [number, number] {
	const nx = (cssX / width) * 2 - 1;
	const ny = 1 - (cssY / height) * 2;
	// clip.x - nx*clip.w = 0 and clip.y - ny*clip.w = 0, with z = 0.
	const a = m[0] - nx * m[3];
	const b = m[4] - nx * m[7];
	const c = nx * m[15] - m[12];
	const d = m[1] - ny * m[3];
	const e = m[5] - ny * m[7];
	const f = ny * m[15] - m[13];
	const det = a * e - b * d;
	if (Math.abs(det) < 1e-12) return [0, 0];
	return [(c * e - b * f) / det, (a * f - c * d) / det];
}

export class ViewUniforms implements View {
	readonly bindGroupLayout: GPUBindGroupLayout;
	readonly bindGroup: GPUBindGroup;
	width = 1;
	height = 1;
	dpr = 1;
	private buffer: GPUBuffer;
	private device: GPUDevice;
	private f32 = new Float32Array(20);
	private viewProj = new Float32Array(IDENTITY);
	private time = 0;
	private dirty = true;

	constructor(gpu: Gpu) {
		this.device = gpu.device;
		this.bindGroupLayout = gpu.device.createBindGroupLayout({
			label: 'view bgl',
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
					buffer: { type: 'uniform' }
				}
			]
		});
		this.buffer = gpu.device.createBuffer({
			label: 'view uniforms',
			size: 80,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.bindGroup = gpu.device.createBindGroup({
			label: 'view bind group',
			layout: this.bindGroupLayout,
			entries: [{ binding: 0, resource: { buffer: this.buffer } }]
		});
	}

	get aspect(): number {
		return this.height > 0 ? this.width / this.height : 1;
	}

	setSize(cssWidth: number, cssHeight: number, dpr: number): void {
		this.width = cssWidth;
		this.height = cssHeight;
		this.dpr = dpr;
		this.dirty = true;
	}

	setTime(time: number): void {
		if (time !== this.time) {
			this.time = time;
			this.dirty = true;
		}
	}

	setViewProj(m: Float32Array): void {
		this.viewProj.set(m);
		this.dirty = true;
	}

	project(x: number, y: number, z = 0): [number, number] {
		return projectToCss(this.viewProj, this.width, this.height, x, y, z);
	}

	unproject(cssX: number, cssY: number): [number, number] {
		return unprojectFromCss(this.viewProj, this.width, this.height, cssX, cssY);
	}

	flush(): void {
		if (!this.dirty) return;
		this.dirty = false;
		const f = this.f32;
		f.set(this.viewProj, 0);
		f[16] = Math.round(this.width * this.dpr);
		f[17] = Math.round(this.height * this.dpr);
		f[18] = this.dpr;
		f[19] = this.time;
		this.device.queue.writeBuffer(this.buffer, 0, f);
	}

	destroy(): void {
		this.buffer.destroy();
	}
}

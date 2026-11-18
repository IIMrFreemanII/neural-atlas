import { mat4 } from 'wgpu-matrix';

export class Camera2D {
	center: [number, number] = [0, 0];
	/** World units visible across the stage height. */
	viewHeight = 2;

	matrix(aspect: number): Float32Array {
		const hh = this.viewHeight / 2;
		const hw = hh * aspect;
		const [cx, cy] = this.center;
		return mat4.ortho(cx - hw, cx + hw, cy - hh, cy + hh, -1, 1) as Float32Array;
	}
}

export class OrbitCamera {
	target: [number, number, number] = [0, 0, 0];
	distance = 5;
	azimuth = 0.6;
	elevation = 0.5;
	fovY = Math.PI / 4;

	matrix(aspect: number): Float32Array {
		const ce = Math.cos(this.elevation);
		const eye: [number, number, number] = [
			this.target[0] + this.distance * ce * Math.sin(this.azimuth),
			this.target[1] + this.distance * Math.sin(this.elevation),
			this.target[2] + this.distance * ce * Math.cos(this.azimuth)
		];
		const proj = mat4.perspective(this.fovY, aspect, 0.05, 100);
		const view = mat4.lookAt(eye, this.target, [0, 1, 0]);
		return mat4.multiply(proj, view) as Float32Array;
	}
}

import { mat4 } from 'wgpu-matrix';
import { angleDelta, easeInOut, lerpPose, type Vec3 } from './space.ts';

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

/** A camera placement for one lesson step. */
export interface Pose {
	target: Vec3;
	/** Radians around the y axis. */
	azimuth: number;
	/** Radians above the XZ plane. */
	elevation: number;
	/** Eye distance from the target; for an ortho pose, the world units visible across the stage height. */
	distance: number;
	ortho?: boolean;
}

const clonePose = (p: Pose): Pose => ({ ...p, target: [p.target[0], p.target[1], p.target[2]] });

/** Eye distance used for ortho poses (only the near/far range depends on it). */
const ORTHO_EYE_DISTANCE = 30;

/**
 * An orbit camera with a home pose per lesson step, a glide between poses, and a drag that lets the learner
 * turn the figure. `home` is the step's pose; `current` is what is on screen.
 */
export class OrbitRig {
	home: Pose = { target: [0, 0, 0], azimuth: 0.6, elevation: 0.5, distance: 5 };
	current: Pose = clonePose(this.home);
	moved = false;

	private fovY: number;
	private minElevation: number;
	private maxElevation: number;
	private maxDrag: number;
	private dragSpeed: number;
	private tween: { from: Pose; to: Pose; t: number; duration: number } | null = null;

	constructor(
		opts: { fovY?: number; minElevation?: number; maxElevation?: number; maxDrag?: number; dragSpeed?: number } = {}
	) {
		this.fovY = opts.fovY ?? Math.PI / 4;
		this.minElevation = opts.minElevation ?? 0.05;
		this.maxElevation = opts.maxElevation ?? 1.5;
		this.maxDrag = opts.maxDrag ?? Math.PI;
		this.dragSpeed = opts.dragSpeed ?? 0.008;
	}

	/** Jump: home = current = p. */
	setPose(p: Pose): void {
		this.home = clonePose(p);
		this.current = clonePose(p);
		this.tween = null;
		this.moved = false;
	}

	/** Make p the new home and glide the current pose to it (cubic ease in/out, azimuth the short way round). */
	glideTo(p: Pose, opts: { duration?: number; instant?: boolean } = {}): void {
		const duration = opts.duration ?? 0.8;
		if (opts.instant || duration <= 0) {
			this.setPose(p);
			return;
		}
		this.home = clonePose(p);
		this.tween = { from: clonePose(this.current), to: clonePose(p), t: 0, duration };
		this.moved = false;
	}

	/** Turn the figure by a pointer movement in CSS px. Cancels a running glide. */
	drag(dxCss: number, dyCss: number): void {
		this.tween = null;
		const c = this.current;
		const az = c.azimuth - dxCss * this.dragSpeed;
		// Keep the azimuth within maxDrag of home, measured the short way round.
		const off = Math.max(-this.maxDrag, Math.min(this.maxDrag, angleDelta(this.home.azimuth, az)));
		c.azimuth = this.home.azimuth + off;
		c.elevation = Math.max(this.minElevation, Math.min(this.maxElevation, c.elevation + dyCss * this.dragSpeed));
		this.moved = true;
	}

	/** Glide back to the home pose. */
	reset(instant = false): void {
		this.glideTo(this.home, { instant });
	}

	/** Advance the glide by dt seconds. True while still animating. */
	update(dt: number): boolean {
		const tw = this.tween;
		if (!tw) return false;
		tw.t += dt / tw.duration;
		if (tw.t >= 1) {
			this.current = clonePose(tw.to);
			this.tween = null;
			return false;
		}
		this.current = lerpPose(tw.from, tw.to, easeInOut(tw.t));
		return true;
	}

	eye(): Vec3 {
		const c = this.current;
		const d = c.ortho ? ORTHO_EYE_DISTANCE : c.distance;
		const ce = Math.cos(c.elevation);
		return [
			c.target[0] + d * ce * Math.sin(c.azimuth),
			c.target[1] + d * Math.sin(c.elevation),
			c.target[2] + d * ce * Math.cos(c.azimuth)
		];
	}

	matrix(aspect: number): Float32Array {
		const c = this.current;
		const view = mat4.lookAt(this.eye(), c.target, [0, 1, 0]);
		let proj;
		if (c.ortho) {
			const hh = c.distance / 2;
			const hw = hh * aspect;
			proj = mat4.ortho(-hw, hw, -hh, hh, 0.05, 100);
		} else {
			proj = mat4.perspective(this.fovY, aspect, 0.05, 100);
		}
		return mat4.multiply(proj, view) as Float32Array;
	}
}

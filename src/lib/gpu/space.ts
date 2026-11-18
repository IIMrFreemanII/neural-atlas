/**
 * Pure 3D helpers shared by the orbit rig, the scene toolkit and their tests.
 * World convention: y is up; floors lie in XZ.
 */
import { mat4 } from 'wgpu-matrix';
import type { Pose } from './camera.ts';

export type Vec3 = [number, number, number];

export const add = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: Readonly<Vec3>, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: Readonly<Vec3>, b: Readonly<Vec3>): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [
	a[1] * b[2] - a[2] * b[1],
	a[2] * b[0] - a[0] * b[2],
	a[0] * b[1] - a[1] * b[0]
];
export const length = (a: Readonly<Vec3>): number => Math.hypot(a[0], a[1], a[2]);
export function normalize(a: Readonly<Vec3>): Vec3 {
	const l = length(a);
	return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
}
export const lerp3 = (a: Readonly<Vec3>, b: Readonly<Vec3>, t: number): Vec3 => [
	a[0] + (b[0] - a[0]) * t,
	a[1] + (b[1] - a[1]) * t,
	a[2] + (b[2] - a[2]) * t
];

/** Direction pointing toward the light. Same as the constant in mesh.wgsl and points.wgsl. */
export const LIGHT: Vec3 = normalize([-0.5, 0.8, 0.35]);

/** Cubic ease in/out, t clamped to 0..1. */
export function easeInOut(t: number): number {
	const x = Math.max(0, Math.min(1, t));
	return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
}

/** Signed shortest angular difference b - a, in (-PI, PI]. */
export function angleDelta(a: number, b: number): number {
	const tau = Math.PI * 2;
	let d = (b - a) % tau;
	if (d > Math.PI) d -= tau;
	else if (d <= -Math.PI) d += tau;
	return d;
}

/**
 * Interpolate two camera poses. Azimuth goes the short way round. The ortho flag cannot be blended:
 * it keeps a's value until t reaches 1, then switches to b's.
 */
export function lerpPose(a: Pose, b: Pose, t: number): Pose {
	return {
		target: lerp3(a.target, b.target, t),
		azimuth: a.azimuth + angleDelta(a.azimuth, b.azimuth) * t,
		elevation: a.elevation + (b.elevation - a.elevation) * t,
		distance: a.distance + (b.distance - a.distance) * t,
		ortho: t >= 1 ? b.ortho : a.ortho
	};
}

/** Slide p along -light until it reaches y = floorY: the point of the floor that p casts its shadow on. */
export function projectToFloor(p: Readonly<Vec3>, floorY: number, light: Readonly<Vec3> = LIGHT): Vec3 {
	const k = light[1] === 0 ? 0 : (p[1] - floorY) / light[1];
	return [p[0] - light[0] * k, floorY, p[2] - light[2] * k];
}

export interface Ray {
	origin: Vec3;
	/** Unit length. */
	dir: Vec3;
}

/** CSS pixels (top-left origin) -> a world ray, by inverting a column-major viewProj. Works for perspective and ortho. */
export function rayFromCss(
	viewProj: ArrayLike<number>,
	width: number,
	height: number,
	cssX: number,
	cssY: number
): Ray {
	const inv = mat4.inverse(viewProj as Float32Array);
	const nx = (cssX / width) * 2 - 1;
	const ny = 1 - (cssY / height) * 2;
	const unproject = (z: number): Vec3 => {
		const x = inv[0] * nx + inv[4] * ny + inv[8] * z + inv[12];
		const y = inv[1] * nx + inv[5] * ny + inv[9] * z + inv[13];
		const zz = inv[2] * nx + inv[6] * ny + inv[10] * z + inv[14];
		const w = inv[3] * nx + inv[7] * ny + inv[11] * z + inv[15];
		const iw = w !== 0 ? 1 / w : 0;
		return [x * iw, y * iw, zz * iw];
	};
	// WebGPU clip depth runs 0 (near) .. 1 (far).
	const near = unproject(0);
	const far = unproject(1);
	return { origin: near, dir: normalize(sub(far, near)) };
}

/** Where a ray meets the plane through `point` with `normal`; null if parallel or behind the origin. */
export function intersectPlane(ray: Ray, point: Readonly<Vec3>, normal: Readonly<Vec3>): Vec3 | null {
	const denom = dot(normal, ray.dir);
	if (Math.abs(denom) < 1e-9) return null;
	const t = dot(normal, sub(point, ray.origin)) / denom;
	if (t < 0) return null;
	return add(ray.origin, scale(ray.dir, t));
}

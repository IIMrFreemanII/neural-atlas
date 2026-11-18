import { describe, expect, it } from 'vitest';
import { OrbitCamera } from './camera.ts';
import {
	LIGHT,
	angleDelta,
	dot,
	easeInOut,
	intersectPlane,
	length,
	lerpPose,
	projectToFloor,
	rayFromCss
} from './space.ts';
import { projectToCss } from './view.ts';

describe('LIGHT', () => {
	it('is a unit vector pointing up', () => {
		expect(length(LIGHT)).toBeCloseTo(1, 12);
		expect(LIGHT[1]).toBeGreaterThan(0);
	});
});

describe('easeInOut', () => {
	it('hits the ends and the midpoint, is monotone and clamps', () => {
		expect(easeInOut(0)).toBe(0);
		expect(easeInOut(1)).toBe(1);
		expect(easeInOut(0.5)).toBeCloseTo(0.5, 12);
		expect(easeInOut(-1)).toBe(0);
		expect(easeInOut(2)).toBe(1);
		let prev = 0;
		for (let i = 1; i <= 20; i++) {
			const v = easeInOut(i / 20);
			expect(v).toBeGreaterThanOrEqual(prev);
			prev = v;
		}
		expect(easeInOut(0.25)).toBeCloseTo(4 * 0.25 ** 3, 12);
	});
});

describe('angleDelta + lerpPose', () => {
	it('takes the short way round', () => {
		expect(angleDelta(3, -3)).toBeCloseTo(2 * Math.PI - 6, 12);
		expect(angleDelta(0, 1)).toBe(1);
	});

	it('interpolates every field, azimuth across the wrap, and flips ortho at the end', () => {
		const a = { target: [0, 0, 0] as [number, number, number], azimuth: 3, elevation: 0.2, distance: 4 };
		const b = { target: [2, 4, 6] as [number, number, number], azimuth: -3, elevation: 0.6, distance: 8, ortho: true };
		const m = lerpPose(a, b, 0.5);
		expect(m.target).toEqual([1, 2, 3]);
		expect(m.elevation).toBeCloseTo(0.4, 12);
		expect(m.distance).toBeCloseTo(6, 12);
		// 3 -> -3 the short way crosses PI: halfway is PI, not 0.
		expect(Math.cos(m.azimuth)).toBeCloseTo(-1, 6);
		expect(m.ortho).toBeUndefined();
		expect(lerpPose(a, b, 1).ortho).toBe(true);
	});
});

describe('projectToFloor', () => {
	it('lands on the floor along -light', () => {
		const p: [number, number, number] = [1, 2.5, -0.5];
		const q = projectToFloor(p, 0.5);
		expect(q[1]).toBe(0.5);
		// the offset is parallel to the light
		const off = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
		const k = off[1] / LIGHT[1];
		expect(off[0]).toBeCloseTo(LIGHT[0] * k, 10);
		expect(off[2]).toBeCloseTo(LIGHT[2] * k, 10);
	});

	it('leaves a point on the floor where it is', () => {
		expect(projectToFloor([1, 0, 2], 0)).toEqual([1, 0, 2]);
	});

	it('honours a custom light', () => {
		expect(projectToFloor([0, 2, 0], 0, [0, 1, 0])).toEqual([0, 0, 0]);
	});
});

describe('rayFromCss + intersectPlane', () => {
	const W = 800;
	const H = 400;

	it('a perspective ray through a projected point passes through that point', () => {
		const cam = new OrbitCamera();
		const m = cam.matrix(W / H);
		const p: [number, number, number] = [0.4, 0.3, -0.2];
		const [cx, cy] = projectToCss(m, W, H, ...p);
		const ray = rayFromCss(m, W, H, cx, cy);
		expect(length(ray.dir)).toBeCloseTo(1, 5);
		// the plane through p facing the ray is hit at p
		const hit = intersectPlane(ray, p, ray.dir)!;
		expect(hit[0]).toBeCloseTo(p[0], 3);
		expect(hit[1]).toBeCloseTo(p[1], 3);
		expect(hit[2]).toBeCloseTo(p[2], 3);
	});

	it('hits the floor plane at the right place', () => {
		const cam = new OrbitCamera();
		const m = cam.matrix(W / H);
		const target: [number, number, number] = [0.7, 0, -0.4];
		const [cx, cy] = projectToCss(m, W, H, ...target);
		const hit = intersectPlane(rayFromCss(m, W, H, cx, cy), [0, 0, 0], [0, 1, 0])!;
		expect(hit[0]).toBeCloseTo(0.7, 3);
		expect(hit[1]).toBeCloseTo(0, 3);
		expect(hit[2]).toBeCloseTo(-0.4, 3);
	});

	it('works for an orthographic view-projection: parallel rays', async () => {
		const { OrbitRig } = await import('./camera.ts');
		const rig = new OrbitRig();
		rig.setPose({ target: [0, 0, 0], azimuth: 0.5, elevation: 0.7, distance: 4, ortho: true });
		const m = rig.matrix(W / H);
		const r1 = rayFromCss(m, W, H, 100, 100);
		const r2 = rayFromCss(m, W, H, 700, 300);
		expect(dot(r1.dir, r2.dir)).toBeCloseTo(1, 4);
		const p: [number, number, number] = [0.3, 0.2, 0.5];
		const [cx, cy] = projectToCss(m, W, H, ...p);
		const hit = intersectPlane(rayFromCss(m, W, H, cx, cy), p, [0, 1, 0])!;
		expect(hit[0]).toBeCloseTo(p[0], 3);
		expect(hit[2]).toBeCloseTo(p[2], 3);
	});

	it('returns null for a parallel plane and for one behind the ray', () => {
		const ray = { origin: [0, 1, 0] as [number, number, number], dir: [1, 0, 0] as [number, number, number] };
		expect(intersectPlane(ray, [0, 0, 0], [0, 1, 0])).toBeNull();
		expect(intersectPlane(ray, [-5, 0, 0], [1, 0, 0])).toBeNull();
		expect(intersectPlane(ray, [5, 0, 0], [1, 0, 0])).toEqual([5, 1, 0]);
	});
});

import { describe, expect, it } from 'vitest';
import { Camera2D, OrbitCamera, OrbitRig } from './camera.ts';
import { projectToCss, unprojectFromCss } from './view.ts';

describe('Camera2D + projection', () => {
	const W = 800;
	const H = 400;

	it('projects the center to the middle of the stage', () => {
		const cam = new Camera2D();
		const m = cam.matrix(W / H);
		const [x, y] = projectToCss(m, W, H, 0, 0);
		expect(x).toBeCloseTo(400, 4);
		expect(y).toBeCloseTo(200, 4);
	});

	it('maps world units to CSS pixels with +y up', () => {
		const cam = new Camera2D();
		cam.viewHeight = 2; // 400px tall => 200 px per unit
		const m = cam.matrix(W / H);
		const [x, y] = projectToCss(m, W, H, 1, 0.5);
		expect(x).toBeCloseTo(600, 3);
		expect(y).toBeCloseTo(100, 3);
	});

	it('round-trips with an offset center', () => {
		const cam = new Camera2D();
		cam.center = [0.3, -0.2];
		cam.viewHeight = 3;
		const m = cam.matrix(W / H);
		const [cx, cy] = projectToCss(m, W, H, 0.9, 0.4);
		const [wx, wy] = unprojectFromCss(m, W, H, cx, cy);
		expect(wx).toBeCloseTo(0.9, 5);
		expect(wy).toBeCloseTo(0.4, 5);
	});

	it('unprojects on the z = 0 plane for an orbit camera', () => {
		const cam = new OrbitCamera();
		const m = cam.matrix(2);
		const [cx, cy] = projectToCss(m, W, H, 0.5, 0.25, 0);
		const [wx, wy] = unprojectFromCss(m, W, H, cx, cy);
		expect(wx).toBeCloseTo(0.5, 4);
		expect(wy).toBeCloseTo(0.25, 4);
	});
});

describe('OrbitRig projection', () => {
	it('matches OrbitCamera for the same pose and puts the target mid-stage, perspective and ortho', () => {
		const cam = new OrbitCamera();
		cam.target = [0.2, 0.1, -0.3];
		const rig = new OrbitRig();
		rig.setPose({ target: [0.2, 0.1, -0.3], azimuth: cam.azimuth, elevation: cam.elevation, distance: cam.distance });
		const a = cam.matrix(2);
		const b = rig.matrix(2);
		for (let i = 0; i < 16; i++) expect(b[i]).toBeCloseTo(a[i], 5);
		for (const ortho of [false, true]) {
			rig.setPose({ target: [0.2, 0.1, -0.3], azimuth: 1, elevation: 0.4, distance: 4, ortho });
			const [x, y] = projectToCss(rig.matrix(2), 800, 400, 0.2, 0.1, -0.3);
			expect(x).toBeCloseTo(400, 3);
			expect(y).toBeCloseTo(200, 3);
		}
	});
});

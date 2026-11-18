import { describe, expect, it } from 'vitest';
import { Camera2D, OrbitCamera } from './camera.ts';
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

import { describe, expect, it } from 'vitest';
import { OrbitRig, type Pose } from '../../gpu/camera.ts';
import { SceneOrbit } from './orbit.ts';

const poseA: Pose = { target: [0, 0, 0], azimuth: 0.5, elevation: 0.5, distance: 5 };
const poseB: Pose = { target: [1, 0, 0], azimuth: 1.5, elevation: 0.9, distance: 7 };

function fakeCtx(reducedMotion = false) {
	const emits: Array<[string, number | string]> = [];
	return {
		emits,
		ctx: { emit: (k: string, v: number | string) => void emits.push([k, v]), reducedMotion }
	};
}

describe('OrbitRig', () => {
	it('setPose jumps and sets home', () => {
		const r = new OrbitRig();
		r.setPose(poseB);
		expect(r.current).toEqual(poseB);
		expect(r.home).toEqual(poseB);
		expect(r.moved).toBe(false);
	});

	it('glides from current to the new home with easing and finishes exactly', () => {
		const r = new OrbitRig();
		r.setPose(poseA);
		r.glideTo(poseB);
		expect(r.home).toEqual(poseB);
		expect(r.update(0.4)).toBe(true); // halfway in time = halfway in progress
		expect(r.current.distance).toBeCloseTo(6, 6);
		expect(r.current.target[0]).toBeCloseTo(0.5, 6);
		// an eased tween is slower than linear at the start
		const r2 = new OrbitRig();
		r2.setPose(poseA);
		r2.glideTo(poseB);
		r2.update(0.2);
		expect(r2.current.distance).toBeLessThan(5 + 2 * 0.25);
		expect(r.update(0.5)).toBe(false);
		expect(r.current).toEqual(poseB);
		expect(r.update(0.1)).toBe(false);
	});

	it('honours duration and instant', () => {
		const r = new OrbitRig();
		r.setPose(poseA);
		r.glideTo(poseB, { duration: 2 });
		r.update(1);
		expect(r.current.distance).toBeCloseTo(6, 6);
		r.glideTo(poseA, { instant: true });
		expect(r.current).toEqual(poseA);
		expect(r.update(1)).toBe(false);
	});

	it('goes the short way round in azimuth', () => {
		const r = new OrbitRig();
		r.setPose({ ...poseA, azimuth: 3 });
		r.glideTo({ ...poseA, azimuth: -3 }, { duration: 1 });
		r.update(0.5);
		expect(Math.cos(r.current.azimuth)).toBeCloseTo(-1, 6);
	});

	it('drag turns the view, sets moved, and cancels a glide', () => {
		const r = new OrbitRig();
		r.setPose(poseA);
		r.glideTo(poseB);
		r.update(0.2);
		r.drag(10, 5);
		expect(r.moved).toBe(true);
		const snapshot = { ...r.current };
		expect(r.update(0.3)).toBe(false);
		expect(r.current.azimuth).toBe(snapshot.azimuth);
		expect(r.home).toEqual(poseB);
	});

	it('drag direction matches OrbitCamera conventions', () => {
		const r = new OrbitRig();
		r.setPose(poseA);
		r.drag(100, 0);
		expect(r.current.azimuth).toBeCloseTo(0.5 - 100 * 0.008, 10);
		r.drag(0, 10);
		expect(r.current.elevation).toBeCloseTo(0.5 + 10 * 0.008, 10);
	});

	it('clamps elevation and azimuth offset', () => {
		const r = new OrbitRig({ minElevation: 0.2, maxElevation: 1.0, maxDrag: 0.5 });
		r.setPose(poseA);
		r.drag(0, 10000);
		expect(r.current.elevation).toBe(1.0);
		r.drag(0, -100000);
		expect(r.current.elevation).toBe(0.2);
		r.drag(100000, 0);
		expect(r.current.azimuth).toBeCloseTo(poseA.azimuth - 0.5, 10);
		r.drag(-100000, 0);
		expect(r.current.azimuth).toBeCloseTo(poseA.azimuth + 0.5, 10);
	});

	it('reset glides home and clears moved', () => {
		const r = new OrbitRig();
		r.setPose(poseA);
		r.drag(50, 20);
		r.reset();
		expect(r.moved).toBe(false);
		expect(r.update(0.4)).toBe(true);
		expect(r.update(0.5)).toBe(false);
		expect(r.current).toEqual(poseA);
		r.drag(50, 20);
		r.reset(true);
		expect(r.current).toEqual(poseA);
	});

	it('builds perspective and ortho matrices; eye matches OrbitCamera', () => {
		const r = new OrbitRig();
		r.setPose({ target: [0, 0, 0], azimuth: 0, elevation: 0, distance: 5 });
		expect(r.eye()[0]).toBeCloseTo(0, 10);
		expect(r.eye()[2]).toBeCloseTo(5, 10);
		const mp = r.matrix(2);
		expect(mp).toHaveLength(16);
		r.setPose({ target: [0, 0, 0], azimuth: 0, elevation: 0, distance: 4, ortho: true });
		const mo = r.matrix(2);
		// ortho: 4 world units across the height -> y scale 2/4 ; x scale halved by aspect 2
		expect(mo[5]).toBeCloseTo(0.5, 6);
		expect(mo[0]).toBeCloseTo(0.25, 6);
		expect(mo[3]).toBe(0); // no perspective divide
	});
});

describe('SceneOrbit', () => {
	it('announces drag and moved=0 on construction', () => {
		const { ctx, emits } = fakeCtx();
		new SceneOrbit(ctx, new OrbitRig());
		expect(emits).toEqual([
			['view:drag', 1],
			['view:moved', 0]
		]);
		const f = fakeCtx();
		new SceneOrbit(f.ctx, new OrbitRig(), { draggable: false });
		expect(f.emits[0]).toEqual(['view:drag', 0]);
	});

	it('first stepPose jumps, later ones glide, reduced motion always jumps', () => {
		const { ctx } = fakeCtx();
		const o = new SceneOrbit(ctx, new OrbitRig());
		o.stepPose(poseA);
		expect(o.rig.current).toEqual(poseA);
		o.stepPose(poseB);
		expect(o.rig.current).toEqual(poseA);
		o.update(0.4);
		expect(o.rig.current.distance).toBeCloseTo(6, 6);

		const rm = fakeCtx(true);
		const o2 = new SceneOrbit(rm.ctx, new OrbitRig());
		o2.stepPose(poseA);
		o2.stepPose(poseB);
		expect(o2.rig.current).toEqual(poseB);
	});

	it('setDraggable emits view:drag', () => {
		const { ctx, emits } = fakeCtx();
		const o = new SceneOrbit(ctx, new OrbitRig());
		o.setDraggable(false);
		expect(emits.at(-1)).toEqual(['view:drag', 0]);
		o.setDraggable(true);
		expect(emits.at(-1)).toEqual(['view:drag', 1]);
	});

	it('a short press is a click; a longer one drags and reports it', () => {
		const { ctx } = fakeCtx();
		const o = new SceneOrbit(ctx, new OrbitRig());
		o.stepPose(poseA);
		expect(o.pointer({ type: 'down', x: 100, y: 100 })).toBeNull();
		expect(o.pointer({ type: 'move', x: 102, y: 101 })).toBeNull();
		expect(o.pointer({ type: 'up', x: 102, y: 101 })).toBe('click');
		expect(o.rig.moved).toBe(false);

		o.pointer({ type: 'down', x: 100, y: 100 });
		expect(o.pointer({ type: 'move', x: 130, y: 100 })).toBe('drag');
		// no jump: the first drag move includes the whole 30 px
		expect(o.rig.current.azimuth).toBeCloseTo(0.5 - 30 * 0.008, 10);
		expect(o.pointer({ type: 'move', x: 140, y: 100 })).toBe('drag');
		expect(o.pointer({ type: 'up', x: 140, y: 100 })).toBe('drag');
		expect(o.rig.moved).toBe(true);
		expect(o.pointer({ type: 'move', x: 200, y: 100 })).toBeNull(); // not pressed
	});

	it('leave ends a press without a click', () => {
		const { ctx } = fakeCtx();
		const o = new SceneOrbit(ctx, new OrbitRig());
		o.pointer({ type: 'down', x: 0, y: 0 });
		expect(o.pointer({ type: 'leave', x: 0, y: 0 })).toBeNull();
	});

	it('a non-draggable orbit never turns', () => {
		const { ctx } = fakeCtx();
		const o = new SceneOrbit(ctx, new OrbitRig(), { draggable: false });
		o.stepPose(poseA);
		o.pointer({ type: 'down', x: 0, y: 0 });
		expect(o.pointer({ type: 'move', x: 50, y: 0 })).toBeNull();
		expect(o.rig.moved).toBe(false);
	});

	it('emits view:moved on change only, and view:reset glides home', () => {
		const { ctx, emits } = fakeCtx();
		const o = new SceneOrbit(ctx, new OrbitRig());
		o.stepPose(poseA);
		o.update(0.016);
		expect(emits.filter(([k]) => k === 'view:moved')).toHaveLength(1); // the construction 0
		o.pointer({ type: 'down', x: 0, y: 0 });
		o.pointer({ type: 'move', x: 20, y: 0 });
		o.update(0.016);
		o.update(0.016);
		expect(emits.filter(([k, v]) => k === 'view:moved' && v === 1)).toHaveLength(1);
		expect(o.command('nope')).toBe(false);
		expect(o.command('view:reset')).toBe(true);
		o.update(0.016);
		expect(emits.at(-1)).toEqual(['view:moved', 0]);
		o.update(1);
		expect(o.rig.current).toEqual(poseA);
	});

	it('view:reset is instant under reduced motion', () => {
		const { ctx } = fakeCtx(true);
		const o = new SceneOrbit(ctx, new OrbitRig());
		o.stepPose(poseA);
		o.rig.drag(50, 0);
		o.command('view:reset');
		expect(o.rig.current).toEqual(poseA);
	});
});

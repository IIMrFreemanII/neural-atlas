import { OrbitRig, type Pose } from './camera.ts';
import { diverging, sequential, sheetFill, tone } from './colors.ts';
import { Lines } from './primitives/lines.ts';
import { SurfaceMesh } from './primitives/mesh.ts';
import { Points } from './primitives/points.ts';
import { Shadows } from './primitives/shadows.ts';
import { Sheets, type SheetSpec, type SheetTexture } from './primitives/sheets.ts';
import { add, dot, intersectPlane, length, projectToFloor, rayFromCss, sub, type Vec3 } from './space.ts';
import type { SceneContext, SceneFactory } from './types.ts';
import { Pulses } from '../lessons/shared/pulses.ts';
import { SceneOrbit } from '../lessons/shared/orbit.ts';

const MESH_N = 48;
const RECT: [number, number, number, number] = [-4, -1.6, -0.8, 1.6];
const PATH_SEGMENTS = 80;
const REVEAL_SECONDS = 3.2;
const HOLD_SECONDS = 1.2;
const PULSE_EVERY = 4;
const SPHERE_R = 0.3;
const FLOOR_HALF: [number, number] = [4.6, 3.4];

const OVERVIEW: Pose = { target: [-0.2, 0.5, 0], azimuth: 0.85, elevation: 0.55, distance: 10 };
const TOP: Pose = { target: [0, 0, 0], azimuth: 0, elevation: 1.5, distance: 8.5, ortho: true };

function surfaceHeight(x: number, z: number): number {
	const cx = (RECT[0] + RECT[2]) / 2;
	const dx = x - cx;
	const bowl = 0.12 * (dx * dx + z * z);
	const hill = 0.7 * Math.exp(-((dx + 0.5) ** 2 + (z - 0.4) ** 2) / 0.5);
	const ripple = 0.04 * Math.cos(5 * Math.hypot(dx, z));
	return bowl + hill + ripple;
}

function pathPoint(s: number): Vec3 {
	const x = RECT[0] + 0.25 + (RECT[2] - RECT[0] - 0.5) * s;
	const z = 1.1 * Math.sin(2.4 * s * Math.PI);
	return [x, surfaceHeight(x, z) + 0.05, z];
}

/** Dev scene for the 3D toolkit: orbit rig, pen reveal, matte sphere, sheets, shadows, ink surface, pulses. */
export const createToolkit3dDemoScene: SceneFactory = () => {
	let ctx: SceneContext;
	let rig: OrbitRig;
	let orbit: SceneOrbit;
	let floor: Sheets;
	let shadows: Shadows;
	let mesh: SurfaceMesh;
	let path: Lines;
	let sheets: Sheets;
	let ball: Points;
	let pulses: Pulses;
	let texA: SheetTexture;
	let texB: SheetTexture;
	let vp: Float32Array = new Float32Array(16);
	let poseIndex = 0;
	let pulseClock = PULSE_EVERY - 1;
	let revealClock = 0;
	let time = 0;
	const sphereXZ: [number, number] = [0.4, 0.9];
	let sphere: Vec3 = [0.4, 0.8, 0.9];

	const sheetA = (): SheetSpec => ({
		origin: [1.1, 0.55, -1.5],
		u: [1.7, 0, 0],
		v: [0, 0.12, 1.5],
		fill: sheetFill(),
		edge: tone('text', 0.7),
		texture: texA,
		opacity: 0.96
	});
	const sheetB = (): SheetSpec => ({
		origin: [1.5, 1.25, -1.2],
		u: [1.3, 0, 0],
		v: [0, 0.1, 1.1],
		fill: sheetFill(),
		edge: tone('text', 0.7),
		texture: texB,
		opacity: 0.96
	});

	const poses = [OVERVIEW, TOP];
	const poseFor = (p: Pose): Pose => {
		const a = ctx.view.aspect;
		// Keep the figure inside narrow stages.
		return { ...p, distance: p.distance * Math.max(1, 1.8 / a) };
	};

	const scene: ReturnType<SceneFactory> = {
		init(c) {
			ctx = c;
			rig = new OrbitRig();
			orbit = new SceneOrbit(c, rig);
			orbit.stepPose(poseFor(poses[poseIndex]));

			floor = new Sheets(c, { capacity: 1 });
			floor.set(0, {
				origin: [-FLOOR_HALF[0], 0, FLOOR_HALF[1]],
				u: [2 * FLOOR_HALF[0], 0, 0],
				v: [0, 0, -2 * FLOOR_HALF[1]],
				fill: sheetFill(),
				edge: tone('muted', 0.5),
				edgePx: 1
			});
			floor.count = 1;

			shadows = new Shadows(c, { capacity: 4 });

			mesh = new SurfaceMesh(c, { cols: MESH_N, rows: MESH_N, style: 'ink' });
			mesh.rect = RECT;
			mesh.range = [0, 0.95];
			mesh.bands = 9;
			mesh.setInk({
				fill: sheetFill(),
				ink: tone('text', 0.55),
				gridEvery: 4,
				hatch: true,
				contours: true,
				contourInk: tone('data', 0.85)
			});
			for (let r = 0; r < MESH_N; r++) {
				for (let col = 0; col < MESH_N; col++) {
					const x = RECT[0] + ((RECT[2] - RECT[0]) * col) / (MESH_N - 1);
					const z = RECT[1] + ((RECT[3] - RECT[1]) * r) / (MESH_N - 1);
					mesh.heights[r * MESH_N + col] = surfaceHeight(x, z);
				}
			}
			mesh.upload();

			path = new Lines(c, { capacity: PATH_SEGMENTS, depthTest: true, depthWrite: true });
			const pts: Vec3[] = [];
			for (let i = 0; i <= PATH_SEGMENTS; i++) pts.push(pathPoint(i / PATH_SEGMENTS));
			path.count = path.setPolyline(0, pts, { width: 2.4, color: tone('grad', 1) });
			path.upload();

			sheets = new Sheets(c, { capacity: 2 });
			texA = sheets.createTexture(8, 6);
			texA.write((x, y) => diverging(Math.sin(x * 0.9) * Math.cos(y * 1.1), 0.9));
			texB = sheets.createTexture(6, 6);
			texB.write((x, y) => sequential((x + y) / 10, 1));
			sheets.count = 2;

			ball = new Points(c, { capacity: 1, shade: 'sphere', size: 'world', depthTest: true, depthWrite: true });
			ball.count = 1;

			pulses = new Pulses(c, { capacity: 2 });
		},
		setParams() {},
		update(frame) {
			time += frame.dt;
			orbit.update(frame.dt);
			vp = rig.matrix(ctx.view.aspect);
			ctx.view.setViewProj(vp);

			// Pen reveal loops; reduced motion shows the finished stroke.
			if (ctx.reducedMotion) {
				path.setReveal(1);
			} else {
				revealClock = (revealClock + frame.dt) % (REVEAL_SECONDS + HOLD_SECONDS);
				path.setReveal(Math.min(1, revealClock / REVEAL_SECONDS));
			}

			const bob = ctx.reducedMotion ? 0 : 0.1 * Math.sin(time * 1.6);
			sphere = [sphereXZ[0], 0.85 + bob, sphereXZ[1]];
			ball.set(0, { x: sphere[0], y: sphere[1], z: sphere[2], radius: SPHERE_R, color: tone('param', 1) });
			ball.upload();

			// Shadows: the ball's blob, and one per sheet, projected along the light.
			const specs = [sheetA(), sheetB()];
			const floorPt = projectToFloor(sphere, 0);
			shadows.blob(0, { center: floorPt, radius: SPHERE_R * 1.7 });
			specs.forEach((s, i) => {
				const o = s.origin;
				const corners = [o, add(o, s.u), add(add(o, s.u), s.v), add(o, s.v)].map((p) => projectToFloor(p, 0));
				shadows.quad(1 + i, { corners: corners as [Vec3, Vec3, Vec3, Vec3], blur: 0.12 });
			});
			shadows.count = 3;

			// Sheets draw back to front.
			const eye = rig.eye();
			const dist = (s: SheetSpec) => length(sub(add(add(s.origin, [s.u[0] / 2, s.u[1] / 2, s.u[2] / 2]), [s.v[0] / 2, s.v[1] / 2, s.v[2] / 2]), eye));
			specs.sort((p, q) => dist(q) - dist(p));
			specs.forEach((s, i) => sheets.set(i, s));

			// A pulse runs from the ball to each sheet every few seconds, only when it is not already travelling.
			pulseClock += frame.dt;
			if (!pulses.busy && pulseClock >= PULSE_EVERY) {
				pulseClock = 0;
				const a = sheetA();
				const b = sheetB();
				const mid = (s: SheetSpec): Vec3 => [
					s.origin[0] + (s.u[0] + s.v[0]) / 2,
					s.origin[1] + (s.u[1] + s.v[1]) / 2,
					s.origin[2] + (s.u[2] + s.v[2]) / 2
				];
				pulses.fire(
					[
						{ points: [sphere, mid(a), mid(b)], tone: 'grad' },
						{ points: [sphere, mid(b)], tone: 'data', delay: 0.3 }
					],
					1.4
				);
			}
			pulses.update(frame.dt);
		},
		draw(pass) {
			floor.draw(pass);
			shadows.draw(pass);
			mesh.draw(pass);
			path.draw(pass);
			sheets.draw(pass);
			ball.draw(pass);
			pulses.draw(pass);
		},
		command(id) {
			if (orbit.command(id)) return;
			if (id === 'pose') {
				poseIndex = (poseIndex + 1) % poses.length;
				orbit.stepPose(poseFor(poses[poseIndex]));
			}
		},
		pointer(e) {
			const r = orbit.pointer(e);
			if (r === 'click') {
				// Click the floor to move the ball there: exercises rayFromCss + intersectPlane.
				const hit = intersectPlane(rayFromCss(vp, ctx.view.width, ctx.view.height, e.x, e.y), [0, 0, 0], [0, 1, 0]);
				if (hit && dot(hit, hit) < 1e4) {
					sphereXZ[0] = Math.max(-FLOOR_HALF[0], Math.min(FLOOR_HALF[0], hit[0]));
					sphereXZ[1] = Math.max(-FLOOR_HALF[1], Math.min(FLOOR_HALF[1], hit[2]));
				}
			}
		},
		dispose() {
			floor.dispose();
			shadows.dispose();
			mesh.destroy();
			path.destroy();
			sheets.dispose();
			ball.destroy();
			pulses.dispose();
		}
	};
	return scene;
};

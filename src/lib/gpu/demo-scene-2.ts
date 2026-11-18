import { mat4 } from 'wgpu-matrix';
import { Camera2D, OrbitCamera } from './camera.ts';
import { diverging, tone } from './colors.ts';
import { Field } from './primitives/field.ts';
import { Lines } from './primitives/lines.ts';
import { SurfaceMesh } from './primitives/mesh.ts';
import { Points } from './primitives/points.ts';
import { Quads } from './primitives/quads.ts';
import type { SceneContext, SceneFactory } from './types.ts';
import { ViewUniforms } from './view.ts';

const FIELD_N = 48;
const GRID_COLS = 8;
const GRID_ROWS = 6;
const MESH_N = 64;
const MESH_HALF = 1.6;
const PATH_SEGMENTS = 96;

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Bowl with a ripple (no ridge). */
function bowl(x: number, z: number): number {
	const r = Math.hypot(x, z);
	return 0.25 * r * r + 0.06 * Math.cos(6 * r) * Math.exp(-r);
}

/** Ridge: a wall along z at x = 0.3. */
function ridge(x: number, z: number): number {
	const a = (x - 0.3) / 0.18;
	const b = z / 1.2;
	return 0.9 * Math.exp(-a * a) * Math.exp(-b * b);
}

function pathPoint(s: number): [number, number, number] {
	const x = -1.4 + 2.8 * s;
	const z = 0.4 * Math.sin(1.5 * x);
	return [x, bowl(x, z) + 0.05, z];
}

export const createPrimitivesDemoScene: SceneFactory = () => {
	let ctx: SceneContext;
	let leftView: ViewUniforms;
	let cam2d: Camera2D;
	let orbit: OrbitCamera;
	let field: Field;
	let quads: Quads;
	let mesh: SurfaceMesh;
	let ball: Points;
	let glow: Points;
	let path: Lines;
	let dragging = false;
	let lastX = 0;
	let lastY = 0;

	const scene: ReturnType<SceneFactory> = {
		init(c) {
			ctx = c;
			// Second view for the left half: its own uniform buffer and bind group.
			leftView = new ViewUniforms(c.gpu);
			const left = { gpu: c.gpu, target: c.target, view: leftView };
			cam2d = new Camera2D();
			orbit = new OrbitCamera();
			orbit.azimuth = 1.1;
			orbit.elevation = 0.5;

			field = new Field(left, { cols: FIELD_N, rows: FIELD_N });
			field.rect = [-1, 0.15, 1, 2.15];
			field.colorLo = tone('neg', 0.22);
			field.colorHi = tone('data', 0.3);
			field.iso = 0.5;

			quads = new Quads(left, { capacity: GRID_COLS * GRID_ROWS });
			const cell = 2 / GRID_COLS;
			let seed = 12345;
			const rnd = () => {
				seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
				return seed / 4294967296;
			};
			for (let r = 0; r < GRID_ROWS; r++) {
				for (let col = 0; col < GRID_COLS; col++) {
					quads.set(r * GRID_COLS + col, {
						x: -1 + cell * (col + 0.5),
						y: -0.15 - cell * (r + 0.5),
						w: cell - 0.03,
						h: cell - 0.03,
						radius: 4,
						color: diverging(rnd() * 2 - 1, 0.8)
					});
				}
			}
			quads.count = GRID_COLS * GRID_ROWS;
			quads.upload();

			mesh = new SurfaceMesh(c, { cols: MESH_N, rows: MESH_N });
			mesh.rect = [-MESH_HALF, -MESH_HALF, MESH_HALF, MESH_HALF];
			mesh.range = [-0.1, 1.6];
			for (let r = 0; r < MESH_N; r++) {
				for (let col = 0; col < MESH_N; col++) {
					const x = -MESH_HALF + (2 * MESH_HALF * col) / (MESH_N - 1);
					const z = -MESH_HALF + (2 * MESH_HALF * r) / (MESH_N - 1);
					mesh.heights[r * MESH_N + col] = bowl(x, z) + ridge(x, z);
				}
			}
			mesh.upload();

			path = new Lines(c, { capacity: PATH_SEGMENTS, blend: 'alpha', depthTest: true });
			for (let i = 0; i < PATH_SEGMENTS; i++) {
				path.set(i, {
					a: pathPoint(i / PATH_SEGMENTS),
					b: pathPoint((i + 1) / PATH_SEGMENTS),
					width: 2,
					color: tone('grad', 1.1, 0.9)
				});
			}
			path.count = PATH_SEGMENTS;
			path.upload();

			glow = new Points(c, { capacity: 1, blend: 'additive', depthTest: true });
			ball = new Points(c, { capacity: 1, blend: 'alpha', depthTest: true });
		},
		setParams() {},
		update(frame) {
			const v = ctx.view;
			const a = v.aspect;

			// Left half: 2D camera whose center is shifted so world x = 0 lands at the left quarter.
			cam2d.viewHeight = Math.max(4.6, 5.2 / a);
			cam2d.center = [(cam2d.viewHeight * a) / 4, 0.25];
			leftView.setSize(v.width, v.height, v.dpr);
			leftView.setTime(frame.time);
			leftView.setViewProj(cam2d.matrix(a));
			leftView.flush();

			const t = ctx.reducedMotion ? 0 : frame.time;
			const ang = t * 0.35;
			const ca = Math.cos(ang);
			const sa = Math.sin(ang);
			const bx = 0.5 * Math.cos(t * 0.5);
			const by = 0.5 * Math.sin(t * 0.4);
			for (let r = 0; r < FIELD_N; r++) {
				for (let c = 0; c < FIELD_N; c++) {
					const x = -1 + (2 * c) / (FIELD_N - 1);
					const y = -1 + (2 * r) / (FIELD_N - 1);
					const plane = sigmoid(3 * (x * ca + y * sa));
					const bump = Math.exp(-((x - bx) ** 2 + (y - by) ** 2) / 0.15);
					field.data[r * FIELD_N + c] = 0.75 * plane + 0.3 * bump;
				}
			}
			field.upload();

			// Right half: orbit camera on the main view, shifted by +0.5 NDC in x so the surface
			// sits in the right half of the window.
			orbit.distance = Math.max(6.5, 11.5 / a);
			const shift = mat4.translation([0.5, 0, 0]);
			v.setViewProj(mat4.multiply(shift, orbit.matrix(a)) as Float32Array);

			const s = ctx.reducedMotion ? 0.25 : (frame.time * 0.08) % 1;
			const [px, py, pz] = pathPoint(s);
			glow.set(0, { x: px, y: py + 0.03, z: pz, radius: 22, color: tone('grad', 2, 0.6), softness: 1 });
			glow.count = 1;
			glow.upload();
			ball.set(0, { x: px, y: py + 0.03, z: pz, radius: 7, color: tone('grad', 2.2) });
			ball.count = 1;
			ball.upload();
		},
		draw(pass) {
			field.draw(pass);
			quads.draw(pass);
			mesh.draw(pass);
			path.draw(pass);
			glow.draw(pass);
			ball.draw(pass);
		},
		pointer(e) {
			if (e.type === 'down') {
				dragging = true;
				lastX = e.x;
				lastY = e.y;
			} else if (e.type === 'move' && dragging) {
				orbit.azimuth -= (e.x - lastX) * 0.008;
				orbit.elevation = Math.min(1.4, Math.max(0.1, orbit.elevation + (e.y - lastY) * 0.008));
				lastX = e.x;
				lastY = e.y;
			} else if (e.type === 'up' || e.type === 'leave') {
				dragging = false;
			}
		},
		dispose() {
			field.destroy();
			quads.destroy();
			mesh.destroy();
			path.destroy();
			glow.destroy();
			ball.destroy();
			leftView.destroy();
		}
	};
	return scene;
};

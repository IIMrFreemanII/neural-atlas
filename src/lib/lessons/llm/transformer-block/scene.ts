import { mat4 } from 'wgpu-matrix';
import { Camera2D, OrbitRig, type Pose } from '../../../gpu/camera.ts';
import { diverging, sheetFill, tone } from '../../../gpu/colors.ts';
import { bezier, Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import { Quads } from '../../../gpu/primitives/quads.ts';
import { Shadows } from '../../../gpu/primitives/shadows.ts';
import { Sheets, type SheetSpec, type SheetTexture } from '../../../gpu/primitives/sheets.ts';
import { easeInOut, length, projectToFloor, sub, type Vec3 } from '../../../gpu/space.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { ViewUniforms } from '../../../gpu/view.ts';
import { insetAspect, insetMatrix, lessonInsets } from '../../fit.ts';
import { SceneOrbit } from '../../shared/orbit.ts';
import { Pulses, type PulsePath } from '../../shared/pulses.ts';
import { loadLm } from '../shared/lm.ts';
import { writeGrid } from '../shared/strip.ts';
import { buildBlockModel, TEXT, type BlockModel } from './model.ts';

// ---- 3D layout. y is up; tokens run along x; each floor is HX*2 wide and HZ*2 deep.
const N = 5;
const COL_X = [-1.9, -0.95, 0, 0.95, 1.9];
const HX = 2.55;
const HZ = 0.62;
const TILE_3D = 0.66; // side of the 8 x 8 tile on a floor
const LIFT = 0.012; // tiles and line ends sit this far above the floor
const Y0 = 0.35;
const GAP = 0.95;
const BYPASS_Z = 0.45;

type BKey = 'stream' | 'norm1' | 'attn' | 'add1' | 'norm2' | 'ff' | 'add2';
type VecKey = 'x' | 'norm1' | 'attnOut' | 'norm2' | 'mlpOut' | 'out';

interface FloorDef {
	name: string;
	b: BKey;
	tone: Tone;
	vec: VecKey;
}
const FLOORS: FloorDef[] = [
	{ name: 'Token vectors', b: 'stream', tone: 'muted', vec: 'x' },
	{ name: 'Normalize', b: 'norm1', tone: 'text', vec: 'norm1' },
	{ name: 'Self-attention', b: 'attn', tone: 'param', vec: 'attnOut' },
	{ name: 'Normalize', b: 'norm2', tone: 'text', vec: 'norm2' },
	{ name: 'Feed-forward', b: 'ff', tone: 'param', vec: 'mlpOut' },
	{ name: 'Block output', b: 'stream', tone: 'muted', vec: 'out' }
];
const fy = (k: number) => Y0 + GAP * k;
const Y_ATTN = fy(2);
const Y_TOP = fy(5);
const Y_ADD1 = (fy(2) + fy(3)) / 2;
const Y_ADD2 = (fy(4) + fy(5)) / 2;

// Stack view: input floor, then each block as four thin sheets, then the top floor.
const MINI = 0.2;
const BLOCK_TONES: Tone[] = ['text', 'param', 'text', 'param'];
function stackStep(d: number): number {
	return Math.min(1.1, Math.max(0.85, 3.6 / d));
}
function stackTop(d: number): number {
	return Y0 + GAP + d * stackStep(d);
}
function blockBase(b: number, d: number): number {
	return Y0 + GAP + b * stackStep(d);
}

// Next-token bars stand on the top floor.
const BAR_X0 = -2.0;
const BAR_PITCH = 0.57;
const BAR_W = 0.3;
const BAR_Z = 0.3;
/** Tallest bar: about a third of the stack's height. */
const barCap = (d: number) => Math.min(0.8, 0.35 * (stackTop(d) - Y0));
const TOP_TILE_Z = -0.28;

// Attention arcs
const ARC_LIFT0 = 0.2;
const ARC_LIFT_STEP = 0.12;
const ARC_SEGMENTS = 14;
const ARC_REVEAL_SECONDS = 0.9;

// Inspector panel (flat overlay), in its own units
const PANEL_W = 4.6;
const PANEL_H: Record<number, number> = { 3: 1.08, 4: 1.38, 5: 0.9 };
const GRID_GAP = 0.1;

const DIM = 0.35;
const DIM_STREAM = 0.5;
const PLAY_A = 1.3;
const PLAY_B = 0.8;
const PLAY_C = 1.4;
const PLAY_STACK = 2.6;

const ORDINALS = ['1st', '2nd', '3rd', '4th', '5th'];

const maxOf = (a: ArrayLike<number>) => Math.max(1e-6, ...Array.from(a, Math.abs));
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const smooth = (t: number) => t * t * (3 - 2 * t);
/** A share as a whole percent; anything above zero but under half a percent reads as '<1%'. */
const pct = (p: number) => (p > 0 && p < 0.005 ? '<1%' : `${Math.round(p * 100)}%`);

/** Draw a vector as a square-ish tile `size` world units wide: `cols` cells across, row-major, first row on top. */
function tile(q: Quads, start: number, vals: ArrayLike<number>, cols: number, x: number, y: number, size: number, maxAbs: number, alpha: number): number {
	const cell = size / (cols + GRID_GAP * (cols - 1));
	return writeGrid(q, start, vals, cols, { x, y, cell, gap: cell * GRID_GAP, maxAbs, alpha, intensity: 0.85, radius: 1 });
}

function rrPts(cx: number, cy: number, w: number, h: number, r: number): Array<[number, number]> {
	const pts: Array<[number, number]> = [];
	const corners: Array<[number, number, number]> = [
		[cx + w / 2 - r, cy + h / 2 - r, 0],
		[cx - w / 2 + r, cy + h / 2 - r, Math.PI / 2],
		[cx - w / 2 + r, cy - h / 2 + r, Math.PI],
		[cx + w / 2 - r, cy - h / 2 + r, 1.5 * Math.PI]
	];
	for (const [ox, oy, a0] of corners) {
		for (let i = 0; i <= 5; i++) {
			const a = a0 + (i / 5) * (Math.PI / 2);
			pts.push([ox + r * Math.cos(a), oy + r * Math.sin(a)]);
		}
	}
	pts.push(pts[0]);
	return pts;
}

/** Three-dimensional arc over the floor from column `from` to column `to` (in the plane z = 0). */
function arcPoints(from: number, to: number, y: number): Array<[number, number, number]> {
	const dist = Math.abs(from - to);
	const peak = ARC_LIFT0 + ARC_LIFT_STEP * (dist - 1);
	const ch = peak / 0.75;
	const pts = bezier([COL_X[from], 0], [COL_X[from], ch], [COL_X[to], ch], [COL_X[to], 0], ARC_SEGMENTS);
	return pts.map(([x, h]) => [x, y + h, 0]);
}

export const createTransformerBlockScene: SceneFactory = () => {
	let ctx: SceneContext;
	let model: BlockModel;
	let updMax = 1; // color scale for the updates and normalized vectors, from the data
	let probMax = 0.1;
	let barScale = 1;

	// 3D
	let rig: OrbitRig;
	let orbit: SceneOrbit;
	let vp: Float32Array = new Float32Array(16);
	let ground: Lines;
	let shadows: Shadows;
	let floors: Sheets;
	let tiles: Sheets;
	let bars: Sheets;
	let lines: Lines; // lanes and bypass lines
	let arcs: Lines; // attention arcs
	let dots: Points; // add nodes, arc end dots
	let pulses: Pulses;
	const detailTex: SheetTexture[][] = []; // [floor][token]
	const topTex: SheetTexture[] = [];
	let topTexDepth = 0;

	// 2D inspector overlay
	let panelView: ViewUniforms;
	let panelCam: Camera2D;
	let pback: Quads;
	let pquads: Quads;
	let plines: Lines;

	// params
	let focus = 1;
	let depth = 1;
	let zoom = 0;
	let column = 4;
	let scenario = 1;
	let paramsSeen = false;
	let hl: Tone | null = null;

	// eased state
	const B: Record<BKey, number> = { stream: 1, norm1: 1, attn: 1, add1: 1, norm2: 1, ff: 1, add2: 1 };
	let layoutT = 0; // 0 detail, 1 stack
	let barA = 0;
	const barVals: number[] = [];
	let barsSeen = false;
	let inspKind = 0; // 3, 4, 5 or 0
	let inspA = 0;
	let arcReveal = 1;
	const lit: Record<string, number> = {};
	let poseKey = '';

	// play
	let playing = false;
	let phase = 0; // 1: up to the attention floor, 2: across the arcs, 3: up to the top
	let phaseT = 0;
	let playStack = false;

	// bookkeeping
	let labelKey = '';
	let lastInspect = '';
	let frameNo = 0;

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);
	const neutral = () => (hl === null ? 1 : 0.3);

	function targets(): Record<BKey, number> {
		const t: Record<BKey, number> = { stream: 1, norm1: 1, attn: 1, add1: 1, norm2: 1, ff: 1, add2: 1 };
		if (focus >= 2 && focus <= 5) {
			t.stream = DIM_STREAM;
			t.norm1 = t.attn = t.add1 = t.norm2 = t.ff = t.add2 = DIM;
			if (focus === 2) t.attn = 1;
			if (focus === 3) t.add1 = 1;
			if (focus === 4) t.ff = 1;
			if (focus === 5) t.norm1 = t.norm2 = 1;
		}
		return t;
	}

	function stackMode(): boolean {
		return zoom >= 0.5;
	}

	/** Orbit pose that frames the box x in [x0, x1], y in [y0, y1], z in [-HZ, HZ] with ~8% margin at the given aspect. */
	function framed(az: number, el: number, x0: number, x1: number, y0: number, y1: number, aspect: number): Pose {
		const ce = Math.cos(el);
		const se = Math.sin(el);
		const right: Vec3 = [Math.cos(az), 0, -Math.sin(az)];
		const up: Vec3 = [-se * Math.sin(az), Math.cos(el), -se * Math.cos(az)];
		const eyeDir: Vec3 = [ce * Math.sin(az), se, ce * Math.cos(az)];
		const pts: Vec3[] = [];
		for (const x of [x0, x1]) for (const y of [y0, y1]) for (const z of [-HZ, HZ]) pts.push([x, y, z]);
		const dotv = (p: Vec3, q: Vec3) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
		let rl = Infinity;
		let rh = -Infinity;
		let ul = Infinity;
		let uh = -Infinity;
		for (const p of pts) {
			rl = Math.min(rl, dotv(p, right));
			rh = Math.max(rh, dotv(p, right));
			ul = Math.min(ul, dotv(p, up));
			uh = Math.max(uh, dotv(p, up));
		}
		const cr = (rl + rh) / 2;
		const cu = (ul + uh) / 2;
		const target: Vec3 = [right[0] * cr + up[0] * cu, right[1] * cr + up[1] * cu, right[2] * cr + up[2] * cu];
		const t = Math.tan(Math.PI / 8) * 0.92;
		let d = 1;
		for (const p of pts) {
			const rel = sub(p, target);
			const depth = -dotv(rel, eyeDir); // distance in front of the target plane
			const need = Math.max(Math.abs(dotv(rel, right)) / (t * aspect), Math.abs(dotv(rel, up)) / t);
			d = Math.max(d, need - depth);
		}
		return { target, azimuth: az, elevation: el, distance: d };
	}

	function pose(aspect: number): Pose {
		const lx = -HX - 1.2; // room for the floor names
		if (stackMode()) {
			const top = stackTop(depth);
			const bars7 = focus === 7;
			return framed(bars7 ? 0.75 : 0.8, bars7 ? 0.4 : 0.32, lx, HX, Y0 - 0.45, top + (bars7 ? barCap(depth) + 0.5 : 0.4), aspect);
		}
		switch (focus) {
			case 2:
				return framed(0.4, 0.32, -HX - 1.0, HX, Y_ATTN - 0.15, Y_ATTN + 0.85, aspect);
			case 3:
				return framed(0.5, 0.42, lx, HX, fy(2) - 0.15, fy(3) + 0.15, aspect);
			case 4:
				return framed(0.5, 0.42, lx, HX, fy(4) - 0.15, fy(5) + 0.15, aspect);
			case 5:
				return framed(0.5, 0.5, lx, HX, fy(1) - 0.15, fy(3) + 0.15, aspect);
			default:
				return framed(0.62, 0.48, lx, HX, Y0 - 0.45, Y_TOP + 0.4, aspect);
		}
	}

	function fillTile(tex: SheetTexture, vals: ArrayLike<number>, maxAbs: number) {
		// Texture row 0 is the near edge of the tile; the first row of values goes on the far edge.
		tex.write((x, y) => diverging(vals[(7 - y) * 8 + x] / maxAbs, 1, 1));
	}

	function floorSpec(y: number, edge: ReturnType<typeof tone>, edgePx: number, opacity: number): SheetSpec {
		return {
			origin: [-HX, y, HZ],
			u: [2 * HX, 0, 0],
			v: [0, 0, -2 * HZ],
			fill: sheetFill(),
			edge,
			edgePx,
			opacity
		};
	}

	function tileSpec(tex: SheetTexture, x: number, y: number, z: number, opacity: number): SheetSpec {
		return {
			origin: [x - TILE_3D / 2, y + LIFT, z + TILE_3D / 2],
			u: [TILE_3D, 0, 0],
			v: [0, 0, -TILE_3D],
			fill: [0, 0, 0, 0],
			edge: [0, 0, 0, 0],
			edgePx: 0,
			texture: tex,
			opacity
		};
	}

	/** Which floors the travelling pulse has reached, for the lighting. */
	function headY(): number {
		if (!playing) return -9;
		const e = (t: number, d: number) => easeInOut(clamp(t / d, 0, 1));
		if (playStack) return Y0 + (stackTop(depth) - Y0) * e(phaseT, PLAY_STACK);
		if (phase === 1) return Y0 + (Y_ATTN - Y0) * e(phaseT, PLAY_A);
		if (phase === 2) return Y_ATTN;
		return Y_ATTN + (Y_TOP - Y_ATTN) * e(phaseT, PLAY_C);
	}
	const litTarget = (head: number, y: number) => (head >= y - 0.02 ? Math.max(0, 1 - (head - y) / 1.6) : 0);
	const litOf = (key: string) => lit[key] ?? 0;
	function stepLit(key: string, head: number, y: number, k: number) {
		const t = ctx.reducedMotion ? 0 : litTarget(head, y);
		const cur = lit[key] ?? 0;
		lit[key] = ctx.reducedMotion ? 0 : cur + (t - cur) * k;
	}

	function arcWeights(): number[] {
		return model.attention[0][column];
	}

	function startPlay() {
		pulses.clear();
		playing = true;
		playStack = stackMode();
		phase = 1;
		phaseT = 0;
		runPhase();
	}
	function endPlay() {
		playing = false;
		phase = 0;
	}
	const lane = (i: number, ya: number, yb: number): PulsePath => ({
		points: [
			[COL_X[i], ya + LIFT + 0.02, 0],
			[COL_X[i], yb + LIFT + 0.02, 0]
		],
		tone: 'data'
	});
	function runPhase() {
		phaseT = 0;
		if (playStack) {
			pulses.fire(
				COL_X.map((_, i) => lane(i, Y0, stackTop(depth))),
				PLAY_STACK,
				endPlay
			);
			return;
		}
		if (phase === 1) {
			pulses.fire(
				COL_X.map((_, i) => lane(i, Y0, Y_ATTN)),
				PLAY_A,
				() => {
					const w = arcWeights();
					const has = w.some((v, j) => j < column && v >= 0.05);
					phase = has ? 2 : 3;
					runPhase();
				}
			);
		} else if (phase === 2) {
			const w = arcWeights();
			const paths: PulsePath[] = [];
			for (let j = 0; j < column; j++) {
				if (w[j] < 0.05) continue;
				paths.push({ points: arcPoints(j, column, Y_ATTN + LIFT), tone: 'data' });
			}
			pulses.fire(paths, PLAY_B, () => {
				phase = 3;
				runPhase();
			});
		} else {
			pulses.fire(
				COL_X.map((_, i) => lane(i, Y_ATTN, Y_TOP)),
				PLAY_C,
				endPlay
			);
		}
	}

	function pickColumn(x: number, y: number): number {
		const v = ctx.view;
		const top = stackMode() ? stackTop(depth) : Y_TOP;
		let best = -1;
		let bd = 44;
		for (let i = 0; i < N; i++) {
			const a = v.project(COL_X[i], Y0, 0);
			const b = v.project(COL_X[i], top, 0);
			const dx = b[0] - a[0];
			const dy = b[1] - a[1];
			const l2 = dx * dx + dy * dy || 1;
			const t = clamp(((x - a[0]) * dx + (y - a[1]) * dy) / l2, 0, 1);
			const d = Math.hypot(x - (a[0] + dx * t), y - (a[1] + dy * t));
			if (d < bd) {
				bd = d;
				best = i;
			}
		}
		return best;
	}

	function rebuild(ppuI: number) {
		const v = ctx.view;
		const sA = smooth(clamp(layoutT, 0, 1));
		const dA = 1 - sA;
		const detail = dA > 0.02;
		const stack = sA > 0.02;
		const nm = neutral();
		const mData = mul('data');
		const mParam = mul('param');
		const mOk = mul('ok');
		const d = depth;
		const topY = stackTop(d);
		barScale = barCap(d) / probMax;
		const labels: Label[] = [];

		const lab = (id: string, p: Vec3, text: string, o: Partial<Label> = {}, dx = 0, dy = 0) => {
			const [x, y] = v.project(p[0], p[1], p[2]);
			const op = o.opacity ?? 1;
			if (op < 0.03) return;
			if (x < -30 || y < -30 || x > v.width + 30 || y > v.height + 30) return;
			labels.push({ ...o, id, x: x + dx, y: y + dy, text, opacity: op });
		};

		const eye = rig.eye();
		const fadeUp = (k: number) => (focus === 2 && k > 2 ? 0.35 : 1);
		const sel = column;

		// ---- floors (sorted far to near; tiles and bars are drawn after, depth tested)
		{
			const list: Array<{ c: Vec3; spec: SheetSpec }> = [];
			const edgeOf = (t: Tone, bright: number, litv: number, mulTone: number) =>
				tone(t, clamp((0.3 + 0.55 * bright) * mulTone + 0.5 * litv, 0, 1), 1);
			const push = (y: number, spec: SheetSpec) => list.push({ c: [0, y, 0], spec });
			// the input floor is shared by both views
			push(Y0, floorSpec(Y0, edgeOf('muted', B.stream, litOf('f0'), nm), 1 + litOf('f0'), 1));
			if (detail) {
				for (let k = 1; k < FLOORS.length; k++) {
					const f = FLOORS[k];
					const m = f.tone === 'param' ? mParam : nm;
					push(fy(k), floorSpec(fy(k), edgeOf(f.tone, B[f.b], litOf(`f${k}`), m), 1 + litOf(`f${k}`), dA * fadeUp(k)));
				}
			}
			if (stack) {
				for (let b = 0; b < d; b++) {
					const lb = litOf(`b${b}`);
					for (let m = 0; m < 4; m++) {
						const y = blockBase(b, d) + m * MINI;
						const t = BLOCK_TONES[m];
						push(y, floorSpec(y, edgeOf(t, 1, lb, t === 'param' ? mParam : nm), 1 + lb, sA));
					}
				}
				push(topY, floorSpec(topY, edgeOf('muted', 1, litOf('top'), nm), 1 + litOf('top'), sA));
			}
			list.sort((p, q) => length(sub(q.c, eye)) - length(sub(p.c, eye)));
			let n = 0;
			for (const e of list) {
				if (n >= floors.capacity) break;
				floors.set(n++, e.spec);
			}
			floors.count = n;

			// the ground shadow of the lowest floor
			const s = floorSpec(Y0, tone('muted'), 1, 1);
			const corners = [
				s.origin,
				[s.origin[0] + s.u[0], s.origin[1], s.origin[2]] as Vec3,
				[s.origin[0] + s.u[0], s.origin[1], s.origin[2] + s.v[2]] as Vec3,
				[s.origin[0], s.origin[1], s.origin[2] + s.v[2]] as Vec3
			].map((p) => projectToFloor(p, 0));
			shadows.quad(0, { corners: corners as [Vec3, Vec3, Vec3, Vec3], blur: 0.16 });
			shadows.count = 1;
		}

		// ---- tiles
		{
			let n = 0;
			const put = (tex: SheetTexture, x: number, y: number, z: number, op: number) => {
				if (op < 0.02 || n >= tiles.capacity) return;
				tiles.set(n++, tileSpec(tex, x, y, z, clamp(op, 0, 1)));
			};
			for (let i = 0; i < N; i++) put(detailTex[0][i], COL_X[i], Y0, 0, B.stream * mData);
			if (detail) {
				for (let k = 1; k < FLOORS.length; k++) {
					const f = FLOORS[k];
					const m = k === 2 || k === 4 ? mOk : mData;
					const b = k === 5 ? B.stream : B[f.b];
					for (let i = 0; i < N; i++) put(detailTex[k][i], COL_X[i], fy(k), 0, dA * b * m * fadeUp(k));
				}
			}
			if (stack) {
				const z = TOP_TILE_Z * barA;
				for (let i = 0; i < N; i++) put(topTex[i], COL_X[i], topY, z, sA * mData);
			}
			tiles.count = n;
		}

		// ---- next-token bars (upright sheets on the top floor)
		{
			let n = 0;
			if (stack && barA > 0.02) {
				const a = barA * sA;
				let top = 0;
				for (let i = 0; i < barVals.length; i++) if (barVals[i] > barVals[top]) top = i;
				for (let i = 0; i < barVals.length; i++) {
					const x = BAR_X0 + i * BAR_PITCH;
					const h = Math.max(0.01, barVals[i] * barScale);
					const isTop = i === top;
					bars.set(n++, {
						origin: [x - BAR_W / 2, topY + LIFT, BAR_Z],
						u: [BAR_W, 0, 0],
						v: [0, h, 0],
						fill: tone(isTop ? 'ok' : 'data', isTop ? 1 : 0.55, 1),
						edge: tone('text', 0.6),
						edgePx: 1,
						opacity: a * mData
					});
				}
				const stagger = model.candidates.some((c) => c.length > 5);
				for (let i = 0; i < barVals.length; i++) {
					const x = BAR_X0 + i * BAR_PITCH;
					const h = Math.max(0.01, barVals[i] * barScale);
					lab(`cand${i}`, [x, topY, HZ], model.candidates[i], { mono: true, size: 12, opacity: a * nm }, 0, 11 + (stagger && i % 2 === 1 ? 15 : 0));
					lab(`prob${i}`, [x, topY + h, BAR_Z], pct(barVals[i]), { mono: true, size: 11, tone: i === top ? 'ok' : 'muted', opacity: a * nm }, 0, -10);
				}
				const rest = Math.max(0, 1 - barVals.reduce((p, c) => p + c, 0));
				lab('cap-bars', [-HX, topY + 1.2, 0], `next word after: ${TEXT}`, { tone: 'muted', size: 11, align: 'left', opacity: a * nm });
				lab('cap-rest', [HX, topY + 1.2, 0], `all other words ${pct(rest)}`, { tone: 'muted', size: 11, align: 'right', mono: true, opacity: a * nm }, 0, 0);
			}
			bars.count = n;
		}

		// ---- lines: lanes and bypasses
		let nL = 0;
		let nD = 0;
		const dim = B.stream * mData;
		for (let i = 0; i < N; i++) {
			const isSel = i === sel;
			const dens = (isSel ? 0.9 : 0.5) * dim;
			const w = isSel ? 2.2 : 1.5;
			if (detail) {
				nL = lines.setPolyline(
					nL,
					[
						[COL_X[i], Y0 + LIFT, 0],
						[COL_X[i], Y_TOP + LIFT, 0]
					],
					{ width: w, color: tone('data', 1, dens * dA) }
				);
				for (const [yFrom, yTo, bk] of [
					[Y0 + LIFT, Y_ADD1, 'add1'],
					[Y_ADD1, Y_ADD2, 'add2']
				] as Array<[number, number, BKey]>) {
					const under = bk === 'add1' && (inspKind === 4 || inspKind === 5) ? 1 - inspA : 1;
					nL = lines.setPolyline(
						nL,
						[
							[COL_X[i], yFrom, 0],
							[COL_X[i], yFrom, BYPASS_Z],
							[COL_X[i], yTo, BYPASS_Z],
							[COL_X[i], yTo, 0]
						],
						{ width: 1.2, color: tone('data', 0.6, (0.4 + 0.6 * B[bk]) * mData * under * dA) }
					);
				}
			}
			if (stack) {
				nL = lines.setPolyline(
					nL,
					[
						[COL_X[i], Y0 + LIFT, 0],
						[COL_X[i], topY + LIFT, 0]
					],
					{ width: w, color: tone('data', 1, dens * sA) }
				);
			}
		}
		lines.count = nL;

		// ---- add nodes (detail view): paper disc with an ink ring at each residual join
		if (detail) {
			for (const [y, bk] of [
				[Y_ADD1, 'add1'],
				[Y_ADD2, 'add2']
			] as Array<[number, BKey]>) {
				const under = bk === 'add1' && (inspKind === 4 || inspKind === 5) ? 1 - inspA : 1;
				const bright = B[bk] * under;
				for (let i = 0; i < N; i++) {
					dots.set(nD++, { x: COL_X[i], y, z: 0, radius: 8, color: sheetFill(dA * under) });
					dots.set(nD++, { x: COL_X[i], y, z: 0, radius: 8, ring: 1.2, color: tone('text', 0.7, 0.9 * bright * nm * dA) });
					lab(`plus-${bk}-${i}`, [COL_X[i], y, 0], '+', { size: 14, opacity: (0.4 + 0.6 * bright) * nm * dA, tone: 'text' });
				}
			}
		}

		// ---- attention arcs for the selected token
		let nA = 0;
		if (detail) {
			const w = arcWeights();
			const bAttn = B.attn * mData * dA;
			for (let j = 0; j <= sel; j++) {
				const wt = w[j];
				if (j === sel) {
					dots.set(nD++, { x: COL_X[j], y: Y_ATTN + LIFT + 0.01, z: 0, radius: 3 + 4 * wt, color: tone('data', 1, 0.95 * bAttn) });
					continue;
				}
				if (wt < 0.015) continue;
				dots.set(nD++, { x: COL_X[j], y: Y_ATTN + LIFT + 0.01, z: 0, radius: 2.5 + 4 * wt, color: tone('data', 1, (0.4 + 0.55 * wt) * bAttn) });
				nA = arcs.setPolyline(nA, arcPoints(sel, j, Y_ATTN + LIFT + 0.01), { width: 1 + 5 * wt, color: tone('data', 1, (0.2 + 0.7 * wt) * bAttn) });
			}
			if (focus === 2) {
				for (let j = 0; j < N; j++) {
					const t = j <= sel ? `${Math.round(w[j] * 100)}%` : '–';
					lab(`pct${j}`, [COL_X[j], Y_ATTN, HZ - 0.05], t, { mono: true, size: 11, tone: j <= sel ? 'data' : 'muted', opacity: dA }, 0, 10);
				}
			}
		}
		arcs.count = nA;
		dots.count = nD;

		// ---- name labels and chips
		for (let k = 0; k < FLOORS.length; k++) {
			if (k > 0 && !detail) continue;
			const f = FLOORS[k];
			const bright = B[f.b];
			const op = k === 0 ? (0.4 + 0.6 * bright) * nm : (0.4 + 0.6 * bright) * dA * nm * (f.tone === 'param' ? 0.5 + 0.5 * mParam : 1);
			lab(`name-${k}`, [-HX - 0.1, fy(k), 0], f.name, { size: 13, align: 'right', tone: f.tone === 'param' ? 'param' : 'text', opacity: op });
		}
		if (stack) {
			for (let b = 0; b < d; b++) {
				lab(`blk${b}`, [-HX - 0.1, blockBase(b, d) + 1.5 * MINI, 0], `block ${b + 1}`, { mono: true, size: 11, align: 'right', tone: 'param', opacity: sA * nm * (0.5 + 0.5 * mParam) });
			}
			lab('cap-after', [-HX - 0.1, topY, 0], d === 1 ? 'after 1 block' : `after ${d} blocks`, { tone: 'muted', size: 11, align: 'right', opacity: sA * nm });
		}
		for (let i = 0; i < N; i++) {
			lab(`chip${i}`, [COL_X[i], Y0, HZ], model.tokens[i], { mono: true, size: 13, tone: i === sel ? 'data' : 'text', opacity: B.stream * nm }, 0, 14);
		}

		// ---- inspector (flat overlay, in the area below the figure)
		let nB = 0;
		let nI = 0;
		let nIL = 0;
		const inspOn = inspKind !== 0 && inspA > 0.01 && detail;
		if (inspOn) {
			const w = PANEL_W;
			const h = PANEL_H[inspKind] ?? 1;
			const cx = 0;
			const cy = 0;
			const labI = (id: string, wx: number, wy: number, text: string, o: Partial<Label> = {}) => {
				const [x, y] = panelView.project(wx, wy);
				const op = (o.opacity ?? 1) * inspA;
				if (op < 0.03) return;
				labels.push({ ...o, id, x, y, text, opacity: op });
			};
			pback.set(nB++, { x: cx, y: cy, w, h, color: sheetFill(inspA), radius: 8 });
			nIL = plines.setPolyline(nIL, rrPts(cx, cy, w, h, 8 / ppuI), { width: 1, color: tone('muted', 1, 0.7 * inspA) });
			const tok = `${model.tokens[sel]} · ${ORDINALS[sel]}`;
			labI('insp-title', cx, cy + h / 2 - 0.09, tok, { size: 11, tone: 'muted', mono: true });
			const L = model.layers[0];
			const T = (arr: ArrayLike<number>, cols: number, x: number, y: number, size: number, maxAbs: number, a: number) => {
				nI = tile(pquads, nI, arr, cols, x, y, size, maxAbs, a * inspA);
			};
			const underline = (x: number, y: number, width: number) => {
				nIL = plines.setPolyline(nIL, [[x - width / 2, y], [x + width / 2, y]], { width: 2, color: tone('ok', 1, 0.7 * mOk * inspA) });
			};
			if (inspKind === 3) {
				const size = 0.5;
				const sy = cy - 0.07;
				T(L.x[sel], 8, -1.5, sy, size, model.maxStream, mData);
				T(L.attnOut[sel], 8, 0, sy, size, updMax, mOk);
				T(L.mid[sel], 8, 1.5, sy, size, model.maxStream, mData);
				underline(0, sy - size / 2 - 0.07, size);
				const ly = sy + size / 2 + 0.1;
				labI('i3-before', -1.5, ly, 'before', { size: 11, opacity: neutral() });
				labI('i3-update', 0, ly, 'update from attention', { size: 11, tone: 'ok', opacity: mOk });
				labI('i3-after', 1.5, ly, 'after', { size: 11, opacity: neutral() });
				labI('i3-plus', -0.75, sy, '+', { size: 18, opacity: neutral() });
				labI('i3-eq', 0.75, sy, '=', { size: 18, opacity: neutral() });
			} else if (inspKind === 4) {
				const small = 0.5;
				const wide = 0.8;
				const sy = cy - 0.13;
				T(L.norm2[sel], 8, -1.7, sy, small, updMax, mData);
				T(L.hidden[sel], 16, 0, sy, wide, model.maxHidden, mData);
				T(L.mlpOut[sel], 8, 1.7, sy, small, updMax, mOk);
				underline(1.7, sy - small / 2 - 0.07, small);
				const arrow = (x0: number, x1: number) => {
					const col = tone('muted', 1.6, 0.9 * inspA);
					nIL = plines.setPolyline(nIL, [[x0, sy], [x1, sy]], { width: 1.5, color: col });
					nIL = plines.setPolyline(nIL, [[x1 - 0.06, sy + 0.05], [x1, sy], [x1 - 0.06, sy - 0.05]], { width: 1.5, color: col });
				};
				arrow(-1.3, -0.56);
				arrow(0.56, 1.3);
				const ly = sy + wide / 2 + 0.1;
				labI('i4-in', -1.7, ly, 'normalized, 64 numbers', { size: 11, opacity: neutral() });
				labI('i4-wide', 0, ly, '4 times wider, 256 numbers', { size: 11, tone: 'data', opacity: mData });
				labI('i4-out', 1.7, ly, 'update, 64 numbers', { size: 11, tone: 'ok', opacity: mOk });
			} else {
				// 64 thin bars per vector. Each chart fills its box, so the shapes can be compared; the spread numbers carry the scale.
				const zy = cy - 0.06;
				const before = L.x[sel];
				const after = L.norm1[sel];
				const half = 0.98;
				const chart = (vals: ArrayLike<number>, x: number, id: string, title: string) => {
					const n0 = vals.length;
					const scale = 0.22 / Math.max(1e-6, ...Array.from(vals, Math.abs));
					const pitch = (2 * half) / n0;
					nIL = plines.setPolyline(nIL, [[x - half, zy], [x + half, zy]], { width: 1, color: tone('muted', 1.4, 0.9 * inspA) });
					for (let k = 0; k < n0; k++) {
						const val = vals[k];
						const bh = Math.max(0.004, Math.abs(val) * scale);
						pquads.set(nI++, {
							x: x - half + (k + 0.5) * pitch,
							y: zy + (val >= 0 ? bh / 2 : -bh / 2),
							w: pitch * 0.72,
							h: bh,
							color: diverging(val >= 0 ? 1 : -1, 0.8, 0.95 * mData * inspA),
							radius: 1
						});
					}
					const m = Array.from(vals).reduce((a, b) => a + b, 0) / n0;
					const sdv = Math.sqrt(Array.from(vals).reduce((a, b) => a + (b - m) ** 2, 0) / n0);
					labI(`i5-${id}`, x, cy + 0.3, title, { size: 11, opacity: neutral() });
					labI(`i5-sd-${id}`, x, cy - 0.37, `spread ${sdv.toFixed(2)}`, { size: 11, mono: true, tone: 'muted' });
				};
				chart(before, -1.2, 'a', 'before');
				chart(after, 1.2, 'b', 'after');
				labI('i5-arrow', 0, zy, '→', { size: 18, tone: 'muted' });
			}
		}
		pback.count = nB;
		pquads.count = nI;
		plines.count = nIL;
		pback.upload();
		pquads.upload();
		plines.upload();

		lines.upload();
		arcs.upload();
		dots.upload();
		floors.upload();
		tiles.upload();
		bars.upload();
		shadows.upload();

		// labels: only set when something changed
		const key = labels.map((l) => `${l.id}|${Math.round(l.x)}|${Math.round(l.y)}|${l.text}|${(l.opacity ?? 1).toFixed(2)}|${l.tone ?? ''}`).join(';');
		if (key !== labelKey || (frameNo < 180 && frameNo % 10 === 0)) {
			labelKey = key;
			ctx.labels.set(labels);
		}
	}

	const scene: Scene = {
		async init(c) {
			ctx = c;
			rig = new OrbitRig();
			orbit = new SceneOrbit(c, rig);
			panelView = new ViewUniforms(c.gpu);
			panelCam = new Camera2D();
			const pv = { gpu: c.gpu, target: c.target, view: panelView };

			ground = new Lines(c, { capacity: 24, depthTest: true });
			shadows = new Shadows(c, { capacity: 2 });
			floors = new Sheets(c, { capacity: 28 });
			tiles = new Sheets(c, { capacity: 40 });
			bars = new Sheets(c, { capacity: 8 });
			lines = new Lines(c, { capacity: 160, depthTest: true });
			arcs = new Lines(c, { capacity: 100, depthTest: true });
			dots = new Points(c, { capacity: 64, depthTest: true });
			pulses = new Pulses(c, { capacity: 10 });
			pback = new Quads(pv, { capacity: 2, blend: 'alpha' });
			pquads = new Quads(pv, { capacity: 400, blend: 'alpha' });
			plines = new Lines(pv, { capacity: 160, blend: 'alpha' });

			// A faint ruled ground under the stack.
			let ng = 0;
			for (let x = -4.5; x <= 4.51; x += 1) {
				ng = ground.setPolyline(ng, [[x, 0, -3], [x, 0, 3]], { width: 1, color: tone('text', 0.12) });
			}
			for (let z = -3; z <= 3.01; z += 1) {
				ng = ground.setPolyline(ng, [[-4.5, 0, z], [4.5, 0, z]], { width: 1, color: tone('text', 0.12) });
			}
			ground.count = ng;
			ground.upload();

			model = buildBlockModel(await loadLm());
			// Colour scales for the updates and the normalized vectors come from the data.
			updMax = 1e-6;
			for (const l of model.layers) {
				for (const m of [l.attnOut, l.mlpOut, l.norm2]) for (const row of m) for (const v of row) updMax = Math.max(updMax, Math.abs(v));
			}
			// One bar scale for every block count, so the bars can be compared as the slider moves.
			probMax = Math.max(0.1, ...model.probs.flat());
			barVals.length = 0;
			for (let i = 0; i < model.candidates.length; i++) barVals.push(0);

			const L = model.layers[0];
			const vecs: Record<VecKey, ArrayLike<number>[]> = { x: L.x, norm1: L.norm1, attnOut: L.attnOut, norm2: L.norm2, mlpOut: L.mlpOut, out: L.out };
			for (let k = 0; k < FLOORS.length; k++) {
				detailTex.push([]);
				for (let i = 0; i < N; i++) {
					const tex = tiles.createTexture(8, 8);
					fillTile(tex, vecs[FLOORS[k].vec][i], maxOf(vecs[FLOORS[k].vec][i]));
					detailTex[k].push(tex);
				}
			}
			for (let i = 0; i < N; i++) topTex.push(tiles.createTexture(8, 8));
		},
		setParams(p) {
			const prevScenario = scenario;
			const prevFocus = focus;
			const prevColumn = column;
			focus = clamp(Math.round(p.focus ?? focus), 1, 7);
			depth = clamp(Math.round(p.depth ?? depth), 1, model.depth);
			zoom = clamp(p.zoom ?? zoom, 0, 1);
			column = clamp(Math.round(p.column ?? column), 0, N - 1);
			scenario = Math.round(p.scenario ?? scenario);
			if (paramsSeen && scenario !== prevScenario) {
				playing = false;
				phase = 0;
				pulses.clear();
			}
			if (!paramsSeen) {
				const t = targets();
				for (const k of Object.keys(B) as BKey[]) B[k] = t[k];
				barA = focus === 7 ? 1 : 0;
				const pr = model.probs[depth - 1];
				for (let i = 0; i < pr.length; i++) barVals[i] = pr[i];
				barsSeen = true;
				layoutT = zoom >= 0.5 ? 1 : 0;
			} else if (column !== prevColumn || (focus === 2 && prevFocus !== 2)) {
				arcReveal = ctx.reducedMotion ? 1 : 0;
			}
			paramsSeen = true;
		},
		update(fr) {
			const v = ctx.view;
			const dt = fr.dt;
			const snap = ctx.reducedMotion;
			const k = snap ? 1 : 1 - Math.exp(-dt / 0.08);

			const t = targets();
			for (const key of Object.keys(B) as BKey[]) {
				const d = t[key] - B[key];
				B[key] = Math.abs(d) > 0.003 ? B[key] + d * k : t[key];
			}
			const barT = focus === 7 ? 1 : 0;
			barA = Math.abs(barT - barA) > 0.003 ? barA + (barT - barA) * k : barT;
			if (barsSeen) {
				const pr = model.probs[depth - 1];
				const k2 = snap ? 1 : 1 - Math.exp(-dt / 0.1);
				for (let i = 0; i < pr.length; i++) {
					const d = pr[i] - barVals[i];
					barVals[i] = Math.abs(d) > 0.0005 ? barVals[i] + d * k2 : pr[i];
				}
			}
			{
				const target = zoom >= 0.5 ? 1 : 0;
				const d = target - layoutT;
				layoutT = snap || Math.abs(d) < 0.01 ? target : layoutT + d * (1 - Math.exp(-dt / 0.12));
			}
			if (topTexDepth !== depth) {
				topTexDepth = depth;
				const out = model.layers[depth - 1].out;
				for (let i = 0; i < N; i++) fillTile(topTex[i], out[i], maxOf(out[i]));
			}

			// inspector kind and fade
			const wanted = zoom < 0.5 && focus >= 3 && focus <= 5 ? focus : 0;
			if (inspKind !== wanted) {
				inspA = snap ? 0 : Math.max(0, inspA - dt * 6);
				if (inspA <= 0) inspKind = wanted;
			} else if (inspKind !== 0 && inspA < 1) {
				inspA = snap ? 1 : Math.min(1, inspA + dt * 6);
			}
			if (inspKind === 0) inspA = 0;

			// arcs draw in when the selected token changes
			if (arcReveal < 1) arcReveal = Math.min(1, arcReveal + dt / ARC_REVEAL_SECONDS);
			arcs.setReveal(snap ? 1 : smooth(arcReveal));

			// the pulse lights each floor as it passes
			if (playing) phaseT += dt;
			const head = headY();
			const kl = 1 - Math.exp(-dt / 0.2);
			for (let f = 0; f < FLOORS.length; f++) stepLit(`f${f}`, head, fy(f), kl);
			for (let b = 0; b < depth; b++) stepLit(`b${b}`, head, blockBase(b, depth) + 1.5 * MINI, kl);
			stepLit('top', head, stackTop(depth), kl);

			// camera: area left after the card, chips, bar and the inspector panel
			const ins = lessonInsets(v);
			const availW = Math.max(1, v.width - ins.left - ins.right);
			const availH = Math.max(1, v.height - ins.top - ins.bottom);
			const hUnits = PANEL_H[inspKind] ?? 0;
			const ppuI = clamp(Math.min(availW / (PANEL_W + 0.1), hUnits > 0 ? (0.42 * availH) / hUnits : 121), 70, 121);
			const panelH = hUnits * ppuI;
			const ins3 = { ...ins, bottom: Math.min(v.height * 0.7, ins.bottom + (panelH + 12) * inspA) };

			const key = `${stackMode() ? 's' : 'd'}|${focus}|${stackMode() ? depth : 0}|${Math.round(insetAspect(v, ins) * 20)}`;
			if (key !== poseKey) {
				poseKey = key;
				orbit.stepPose(pose(insetAspect(v, ins)));
			}
			orbit.update(dt);
			const a = insetAspect(v, ins3);
			vp = mat4.multiply(insetMatrix(v, ins3), rig.matrix(a)) as Float32Array;
			v.setViewProj(vp);

			// the flat overlay: panel units -> px, centred in the free area just above the control bar
			panelView.setSize(v.width, v.height, v.dpr);
			const cxPx = ins.left + availW / 2;
			const cyPx = v.height - ins.bottom - panelH / 2 - 6;
			panelCam.viewHeight = v.height / ppuI;
			panelCam.center = [(v.width / 2 - cxPx) / ppuI, (cyPx - v.height / 2) / ppuI];
			panelView.setViewProj(panelCam.matrix(v.aspect));
			panelView.flush();

			frameNo++;
			rebuild(ppuI);
			pulses.update(dt);

			const inspect = `${model.tokens[column]} · ${ORDINALS[column]}`;
			if (inspect !== lastInspect) {
				lastInspect = inspect;
				ctx.emit('inspect', inspect);
			}
		},
		draw(pass) {
			ground.draw(pass);
			shadows.draw(pass);
			floors.draw(pass);
			tiles.draw(pass);
			bars.draw(pass);
			lines.draw(pass);
			arcs.draw(pass);
			dots.draw(pass);
			pulses.draw(pass);
			pback.draw(pass);
			pquads.draw(pass);
			plines.draw(pass);
		},
		command(id) {
			if (orbit.command(id)) return;
			if (id === 'play') startPlay();
		},
		highlight(t) {
			hl = t;
		},
		pointer(e) {
			const r = orbit.pointer(e);
			if (r !== 'click') return;
			const i = pickColumn(e.x, e.y);
			if (i < 0 || i === column) return;
			column = i;
			arcReveal = ctx.reducedMotion ? 1 : 0;
			ctx.emit('param:column', i);
		},
		dispose() {
			ground.destroy();
			shadows.dispose();
			floors.dispose();
			tiles.dispose();
			bars.dispose();
			lines.destroy();
			arcs.destroy();
			dots.destroy();
			pulses.dispose();
			pback.destroy();
			pquads.destroy();
			plines.destroy();
			panelView.destroy();
		}
	};
	return scene;
};

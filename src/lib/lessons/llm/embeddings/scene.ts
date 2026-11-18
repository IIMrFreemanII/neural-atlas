import { mat4 } from 'wgpu-matrix';
import { Camera2D, OrbitRig, type Pose } from '../../../gpu/camera.ts';
import { mixColor, tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import { Quads } from '../../../gpu/primitives/quads.ts';
import { Shadows } from '../../../gpu/primitives/shadows.ts';
import { cross, normalize, sub, type Vec3 } from '../../../gpu/space.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { ViewUniforms } from '../../../gpu/view.ts';
import { analogy, nearest } from '../../../ml/embedding.ts';
import { insetAspect, insetMatrix, lessonInsets } from '../../fit.ts';
import { SceneOrbit } from '../../shared/orbit.ts';
import { writeStrip } from '../shared/strip.ts';
import { ANALOGIES, CATEGORIES, POINTS, RELATIONS, VECTORS, WORDS, indexOf, project3D } from './data.ts';

// The PCA picture is a unit ball; it is scaled by S and lifted so every word floats above the floor (y = 0).
const S = 1.5;
const LIFT = 1.08;
const FLOOR = 2;
const FLOOR_STEP = 0.5;
const DOT_R = 0.032;
const DOT_R_BIG = 0.045;
const RING_SEGMENTS = 48;
const RING_PAD = 0.2;
const TAN_HALF_FOV = Math.tan(Math.PI / 8);
const REVEAL_SECONDS = 0.9;
const GLIDE_SECONDS = 0.85;
const SPIN_SPEED = 0.12; // rad/s
const SPIN_SLOW = 0.04; // while a relation or analogy is showing
const HOVER_PX = 14;
const PLAY_SECONDS = 1.2;
/** Reduced motion: Play shows the question for this long, then the answer, with no travel between. */
const PLAY_HOLD_REDUCED = 0.6;
const TRIM = 0.09; // arrows stop this far short of a word's sphere
const HEAD = 0.1;
const N = WORDS.length;

// Side panel, CSS px.
const STRIP_COLS = 20;
const STRIP_ROWS = 5;
const STRIP_GAP = 1.5;
const PANEL_MIN = 220;
const PANEL_MAX = 290;
const PANEL_GAP = 16;

/** Two words per category that are labelled when nothing else is. */
const LANDMARKS = [
	['king', 'woman'],
	['cat', 'horse'],
	['pizza', 'coffee'],
	['paris', 'japan'],
	['run', 'swam'],
	['big', 'hot']
].map((pair) => pair.map(indexOf));

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const easeInOut = (t: number) => t * t * (3 - 2 * t);

const toWorld = (p: ArrayLike<number>): Vec3 => [p[0] * S, (p[1] + LIFT) * S, p[2] * S];
const WP: Vec3[] = POINTS.map(toWorld);

/** Center and floor radius of each group's ring. */
const GROUPS = CATEGORIES.map((_, k) => {
	const m = WP.filter((_p, i) => WORDS[i].category === k);
	const cx = m.reduce((s, p) => s + p[0], 0) / m.length;
	const cz = m.reduce((s, p) => s + p[2], 0) / m.length;
	const r = Math.max(...m.map((p) => Math.hypot(p[0] - cx, p[2] - cz))) + RING_PAD;
	return { cx, cz, r };
});

interface Box {
	x0: number;
	x1: number;
	y0: number;
	y1: number;
}
const overlaps = (a: Box, b: Box) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

export const createEmbeddingsScene: SceneFactory = () => {
	let ctx: SceneContext;
	let rig: OrbitRig;
	let orbit: SceneOrbit;
	let panelView: ViewUniforms;
	let panelCam: Camera2D;
	let floor: Lines;
	let rings: Lines;
	let stems: Lines;
	let links: Lines;
	let anaAB: Lines;
	let anaCT: Lines;
	let shadows: Shadows;
	let dots: Points;
	let marks: Points;
	let strip: Quads;

	// params
	let colors = 1;
	let neighbors = 0;
	let relation = 0;
	let analogyIdx = 0;
	let showAnalogy = 0;
	let spin = 1;
	let wordParam = indexOf('cat');
	let scenario = 1;
	let paramsSeen = false;
	let hl: Tone | null = null;

	// state
	let selected = indexOf('cat');
	let hover = -1;
	let pressed = false;
	let markerT = 1;
	let glideLeft = 0;
	let poseKey = '';
	let poseMode = '';
	let vp: Float32Array = new Float32Array(16);

	// draw-in progress (0..1; the c to answer arrow starts late, so it may be negative)
	let relP = 1;
	let relKey = '';
	let abP = 1;
	let ctP = 1;
	let anaKey2 = '';
	let ringP = 0;

	// per frame projection of the words
	const px = new Float64Array(N);
	const py = new Float64Array(N);
	const pw = new Float64Array(N); // distance along the view axis; <= 0 is behind the camera
	const pr = new Float64Array(N); // sphere radius, CSS px
	const order = Array.from({ length: N }, (_, i) => i);

	// cached derived values
	let nbrKey = '';
	let nbrs: { index: number; similarity: number }[] = [];
	let anaKey = -1;
	let anaVec3: Vec3 = [0, 0, 0];
	let anaTip: Vec3 = [0, 0, 0];
	let anaFound = 0;

	// emitted readouts
	let lastWordOut = '';
	let lastBestOut = '';

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	function project(x: number, y: number, z: number): [number, number, number] {
		const cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
		if (cw <= 1e-4) return [NaN, NaN, cw];
		const cx = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
		const cy = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
		const v = ctx.view;
		return [(cx / cw * 0.5 + 0.5) * v.width, (1 - (cy / cw * 0.5 + 0.5)) * v.height, cw];
	}

	function ensureDerived() {
		const k = Math.round(neighbors);
		const key = `${selected}:${k}`;
		if (key !== nbrKey) {
			nbrKey = key;
			nbrs = k > 0 ? nearest(VECTORS, VECTORS[selected], k, [selected]) : [];
		}
		if (anaKey !== analogyIdx) {
			anaKey = analogyIdx;
			const { a, b, c } = ANALOGIES[analogyIdx];
			const v = analogy(VECTORS[a], VECTORS[b], VECTORS[c]);
			anaVec3 = project3D(v);
			anaTip = toWorld(anaVec3);
			anaFound = nearest(VECTORS, v, 1, [a, b, c])[0].index;
		}
	}

	/** The camera placement this step wants: the whole cloud, or what a relation, analogy or neighbor list shows. */
	function computePose(aspect: number): { key: string; mode: string; pose: Pose } {
		let pts: Vec3[] = [];
		let dir: Vec3 | null = null;
		let mode = 'o';
		if (relation > 0) {
			mode = `r${relation}`;
			const pairs = RELATIONS[relation - 1].pairs;
			for (const [a, b] of pairs) pts.push(WP[a], WP[b]);
			const d: Vec3 = [0, 0, 0];
			for (const [a, b] of pairs) for (let k = 0; k < 3; k++) d[k] += (WP[b][k] - WP[a][k]) / pairs.length;
			dir = d;
		} else if (showAnalogy >= 0.5) {
			const an = ANALOGIES[analogyIdx];
			mode = `a${analogyIdx}`;
			pts = [WP[an.a], WP[an.b], WP[an.c], WP[anaFound], anaTip];
			dir = sub(WP[an.b], WP[an.a]);
		} else if (nbrs.length > 0) {
			mode = `n${selected}:${nbrs.length}`;
			pts = [selected, ...nbrs.map((n) => n.index)].map((i) => WP[i]);
		} else {
			pts = WP;
		}
		let azimuth = 0.6;
		let elevation = 0.4;
		if (mode === 'o') {
			const grouped = colors >= 0.5;
			azimuth = grouped ? 0.95 : 0.6;
			elevation = grouped ? 0.62 : 0.42;
		}
		// Look across the arrows, not along them, so their parallel direction shows.
		if (dir && Math.hypot(dir[0], dir[2]) > 0.2) {
			azimuth = Math.atan2(-dir[2], dir[0]);
			if (Math.cos(azimuth - 0.6) < 0) azimuth += Math.PI;
		}
		// Everything that must stay on screen: the points, their floor feet (the stems) and, when shown, their group rings.
		const all: Vec3[] = [...pts, ...pts.map((p): Vec3 => [p[0], 0, p[2]])];
		if (colors >= 0.5) {
			const groups = new Set<number>();
			if (mode === 'o') CATEGORIES.forEach((_, k) => groups.add(k));
			else for (const p of pts) { const i = WP.indexOf(p); if (i >= 0) groups.add(WORDS[i].category); }
			if (mode !== 'o' && showAnalogy >= 0.5) for (const i of [ANALOGIES[analogyIdx].a, ANALOGIES[analogyIdx].b, ANALOGIES[analogyIdx].c, anaFound]) groups.add(WORDS[i].category);
			for (const k of groups) for (let m = 0; m < 16; m++) all.push(ringPoint(k, (m / 16) * Math.PI * 2));
		}
		// Fit through the real view: recentre on the projected bounds, then back off until every point is inside the margin.
		const ce = Math.cos(elevation);
		const e: Vec3 = [ce * Math.sin(azimuth), Math.sin(elevation), ce * Math.cos(azimuth)];
		const right = normalize(cross([0, 1, 0], e));
		const up = cross(e, right);
		const dotv = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
		let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
		for (const p of all) {
			x0 = Math.min(x0, dotv(p, right)); x1 = Math.max(x1, dotv(p, right));
			y0 = Math.min(y0, dotv(p, up)); y1 = Math.max(y1, dotv(p, up));
		}
		const cx = (x0 + x1) / 2;
		const cy = (y0 + y1) / 2;
		let tc: Vec3 = [0, 0, 0];
		for (const p of all) tc = [tc[0] + p[0] / all.length, tc[1] + p[1] / all.length, tc[2] + p[2] / all.length];
		// Keep the depth of the mean, move the target to the middle of the projected box.
		const zc0 = dotv(tc, e);
		const t: Vec3 = [
			right[0] * cx + up[0] * cy + e[0] * zc0,
			right[1] * cx + up[1] * cy + e[1] * zc0,
			right[2] * cx + up[2] * cy + e[2] * zc0
		];
		const margin = mode === 'o' ? 1.02 : 1.1;
		const tanV = TAN_HALF_FOV / margin;
		const tanH = (TAN_HALF_FOV * aspect) / margin;
		let distance = 2;
		for (const p of all) {
			const rel: Vec3 = [p[0] - t[0], p[1] - t[1], p[2] - t[2]];
			const need = dotv(rel, e) + Math.max(Math.abs(dotv(rel, right)) / tanH, Math.abs(dotv(rel, up)) / tanV);
			distance = Math.max(distance, need);
		}
		const key = mode === 'o' ? `o${colors >= 0.5 ? 1 : 0}:${Math.round(aspect * 10)}` : `${mode}:${Math.round(aspect * 10)}`;
		return { key, mode: mode[0], pose: { target: t, azimuth, elevation, distance } };
	}

	function radiusOf(i: number, anaDone: boolean, ana: (typeof ANALOGIES)[number] | null): number {
		return i === selected || i === hover || (ana !== null && anaDone && i === anaFound) ? DOT_R_BIG : DOT_R;
	}

	function ringPoint(k: number, a: number): Vec3 {
		const g = GROUPS[k];
		return [g.cx + g.r * Math.cos(a), 0, g.cz + g.r * Math.sin(a)];
	}

	function pickAt(x: number, y: number): number {
		let best = -1;
		let bestScore = HOVER_PX;
		for (let i = 0; i < N; i++) {
			if (!(pw[i] > 0)) continue;
			const d = Math.hypot(px[i] - x, py[i] - y);
			// Nearer words win a tie: favor them by up to 2 px.
			const near = clamp(1 - (pw[i] - (rig.current.distance - 1.7)) / 3.4, 0, 1);
			const score = d - 2 * near;
			if (score < bestScore) {
				bestScore = score;
				best = i;
			}
		}
		return best;
	}

	/** Shaft plus a two-segment head (turned to face the camera) from a to b, stopped trimA / trimB short. */
	function arrow(
		layer: Lines,
		n: number,
		a: Vec3,
		b: Vec3,
		width: number,
		color: [number, number, number, number],
		trimA: number,
		trimB: number
	): number {
		const d = sub(b, a);
		const len = Math.hypot(d[0], d[1], d[2]);
		if (len < trimA + trimB + 0.02) return n;
		const u: Vec3 = [d[0] / len, d[1] / len, d[2] / len];
		const s: Vec3 = [a[0] + u[0] * trimA, a[1] + u[1] * trimA, a[2] + u[2] * trimA];
		const e: Vec3 = [b[0] - u[0] * trimB, b[1] - u[1] * trimB, b[2] - u[2] * trimB];
		layer.set(n++, { a: s, b: e, width, color, t0: 0, t1: 1 });
		const eye = rig.eye();
		let perp = cross(u, normalize(sub(eye, e)));
		if (Math.hypot(perp[0], perp[1], perp[2]) < 1e-3) perp = cross(u, [0, 1, 0]);
		perp = normalize(perp);
		const c = Math.cos(0.45);
		const sn = Math.sin(0.45);
		for (const sg of [-1, 1]) {
			const h: Vec3 = [
				e[0] + (-u[0] * c + sg * perp[0] * sn) * HEAD,
				e[1] + (-u[1] * c + sg * perp[1] * sn) * HEAD,
				e[2] + (-u[2] * c + sg * perp[2] * sn) * HEAD
			];
			// The head appears when the pen reaches the end of the shaft.
			layer.set(n++, { a: e, b: h, width, color, t0: 1, t1: 1 });
		}
		return n;
	}

	/** The animation arrived: the word it found becomes the selected word. */
	function landAnalogy() {
		if (showAnalogy < 0.5) return;
		ensureDerived();
		selected = anaFound;
	}

	/** How far the marker has travelled, 0..1. Under reduced motion it waits at the start, then lands. */
	function markerEase(): number {
		if (ctx.reducedMotion) return markerT >= 1 ? 1 : 0;
		return easeInOut(markerT);
	}

	function advance(dt: number) {
		if (markerT < 1 && dt > 0) {
			markerT = Math.min(1, markerT + dt / (ctx.reducedMotion ? PLAY_HOLD_REDUCED : PLAY_SECONDS));
			if (markerT >= 1) landAnalogy();
		}
		// Draw-ins restart when what they show changes.
		const rk = `${selected}|${relation}|${neighbors > 0 ? 1 : 0}`;
		if (rk !== relKey) {
			relKey = rk;
			relP = ctx.reducedMotion ? 1 : 0;
		}
		const ak = `${showAnalogy >= 0.5 ? 1 : 0}|${analogyIdx}`;
		if (ak !== anaKey2) {
			anaKey2 = ak;
			abP = ctx.reducedMotion ? 1 : 0;
			ctP = ctx.reducedMotion || markerT < 1 ? 1 : -0.3;
		}
		const step = dt / REVEAL_SECONDS;
		relP = Math.min(1, relP + step);
		abP = Math.min(1, abP + step);
		ctP = Math.min(1, ctP + step);
		const ringTarget = colors >= 0.5 ? 1 : 0;
		if (ctx.reducedMotion) ringP = ringTarget;
		else ringP = ringTarget > ringP ? Math.min(ringTarget, ringP + step) : Math.max(ringTarget, ringP - step);
	}

	function update3D(dt: number) {
		const v = ctx.view;
		const ins = lessonInsets(v);
		const freeW = v.width - ins.left - ins.right;
		const panelW = clamp(freeW * 0.27, PANEL_MIN, PANEL_MAX);
		const panelX0 = v.width - ins.right - panelW;
		const cloudIns = { ...ins, right: ins.right + panelW + PANEL_GAP };
		const aspect = insetAspect(v, cloudIns);

		ensureDerived();
		const want = computePose(aspect);
		if (want.key !== poseKey) {
			const sameMode = want.mode === poseMode && poseMode !== 'o';
			poseKey = want.key;
			poseMode = want.mode;
			const pose = sameMode && rig.moved ? { ...want.pose, azimuth: rig.current.azimuth, elevation: rig.current.elevation } : want.pose;
			orbit.stepPose(pose);
			glideLeft = ctx.reducedMotion ? 0 : GLIDE_SECONDS;
		}
		orbit.update(dt);
		if (glideLeft > 0) glideLeft = Math.max(0, glideLeft - dt);
		else if (spin >= 0.5 && !pressed && !rig.moved && !ctx.reducedMotion && dt > 0) {
			const d = (relation > 0 || showAnalogy >= 0.5 ? SPIN_SLOW : SPIN_SPEED) * dt;
			rig.current.azimuth += d;
			rig.home.azimuth += d;
		}
		vp = mat4.multiply(insetMatrix(v, cloudIns), rig.matrix(aspect)) as Float32Array;
		v.setViewProj(vp);

		advance(dt);

		// Project the words.
		const yScale = Math.hypot(vp[1], vp[5], vp[9]);
		for (let i = 0; i < N; i++) {
			const [x, y, z] = project(WP[i][0], WP[i][1], WP[i][2]);
			px[i] = x;
			py[i] = y;
			pw[i] = z;
			pr[i] = z > 0 ? (DOT_R_BIG * yScale * v.height * 0.5) / z : 0;
		}
		order.sort((a, b) => pw[b] - pw[a]);

		const relIdx = relation > 0 ? RELATIONS[relation - 1] : null;
		const relSet = new Set<number>();
		relIdx?.pairs.forEach(([a, b]) => {
			relSet.add(a);
			relSet.add(b);
		});
		const ana = showAnalogy >= 0.5 ? ANALOGIES[analogyIdx] : null;
		const anaDone = markerT >= 1;
		const anaSet = new Set<number>();
		if (ana) {
			anaSet.add(ana.a);
			anaSet.add(ana.b);
			anaSet.add(ana.c);
			if (anaDone) anaSet.add(anaFound);
		}
		const nbrSet = new Set(nbrs.map((n) => n.index));
		const focusMode = relIdx !== null || ana !== null;
		const inFocus = (i: number) => (relIdx ? relSet.has(i) : ana ? anaSet.has(i) : true);

		// Floor: a faint ruled grid.
		let nf = 0;
		const n = Math.round((2 * FLOOR) / FLOOR_STEP);
		for (let k = 0; k <= n; k++) {
			const t = -FLOOR + k * FLOOR_STEP;
			const edge = k === 0 || k === n;
			const c = tone('text', edge ? 0.22 : 0.12);
			floor.set(nf++, { a: [t, 0, -FLOOR], b: [t, 0, FLOOR], width: 1, color: c });
			floor.set(nf++, { a: [-FLOOR, 0, t], b: [FLOOR, 0, t], width: 1, color: c });
		}
		floor.count = nf;
		floor.upload();

		// Group rings on the floor.
		let nr = 0;
		if (ringP > 0.001) {
			CATEGORIES.forEach((cat, k) => {
				const pts: Vec3[] = [];
				for (let m = 0; m <= RING_SEGMENTS; m++) pts.push(ringPoint(k, (m / RING_SEGMENTS) * Math.PI * 2));
				nr = rings.setPolyline(nr, pts, { width: 1.4, color: tone(cat.tone, 0.5 * mul(cat.tone) * (focusMode ? 0.5 : 1)) });
			});
		}
		rings.count = nr;
		rings.setReveal(ctx.reducedMotion ? 1 : easeInOut(ringP));
		rings.upload();

		// Shadows on the floor under each word; higher words cast a larger, fainter one.
		for (let i = 0; i < N; i++) {
			const p = WP[i];
			const dim = (focusMode && !inFocus(i) ? 0.4 : 1) * mul(CATEGORIES[WORDS[i].category].tone);
			shadows.blob(i, {
				center: [p[0], 0, p[2]],
				radius: 1.6 * DOT_R + 0.01 * p[1],
				squash: 1,
				alpha: 0.6 * clamp(1 - 0.12 * p[1], 0.35, 1) * dim
			});
		}
		shadows.count = N;
		shadows.upload();

		// Spheres, far to near.
		const base0 = tone('muted', 1);
		const ink = tone('text', 1);
		let nd = 0;
		for (const i of order) {
			if (!(pw[i] > 0)) continue;
			const cat = CATEGORIES[WORDS[i].category];
			const c1 = tone(cat.tone, 1);
			const rgb = mixColor(i === selected ? ink : base0, c1, colors);
			const d = (focusMode && !inFocus(i) ? 0.28 : 1) * mul(cat.tone) * (0.85 + 0.15 * colors);
			dots.set(nd++, { x: WP[i][0], y: WP[i][1], z: WP[i][2], radius: radiusOf(i, anaDone, ana), color: [rgb[0], rgb[1], rgb[2], clamp(d, 0, 1)] });
		}
		dots.count = nd;
		dots.upload();

		// Stems from the floor up to each sphere.
		let ns = 0;
		for (let i = 0; i < N; i++) {
			const cat = CATEGORIES[WORDS[i].category];
			const hot = i === hover || i === selected;
			const d = (focusMode && !inFocus(i) ? 0.12 : 0.42) * mul(cat.tone);
			stems.set(ns++, {
				a: [WP[i][0], 0, WP[i][2]],
				b: WP[i],
				width: hot ? 1.5 : 1,
				color: hot ? tone(cat.tone, 0.9 * mul(cat.tone)) : tone('muted', d)
			});
		}
		stems.count = ns;
		stems.upload();

		// Neighbor links and relation arrows.
		let nl = 0;
		nbrs.forEach((nb, rank) => {
			links.set(nl++, {
				a: WP[selected],
				b: WP[nb.index],
				width: 1.5,
				color: tone('data', 0.9 * (1 - rank * 0.06) * mul('data'))
			});
		});
		if (relIdx) {
			const c = tone('text', 0.85 * mul('text'));
			for (const [a, b] of relIdx.pairs) nl = arrow(links, nl, WP[a], WP[b], 1.6, c, TRIM, TRIM);
		}
		links.count = nl;
		links.setReveal(ctx.reducedMotion ? 1 : easeInOut(clamp(relP, 0, 1)));
		links.upload();

		// Analogy: one offset a to b, copied from c to where it lands.
		let nab = 0;
		let nct = 0;
		if (ana) {
			const m = mul('param');
			nab = arrow(anaAB, 0, WP[ana.a], WP[ana.b], 2, tone('param', 0.95 * m), TRIM, TRIM);
			nct = arrow(anaCT, 0, WP[ana.c], anaTip, 2, tone('param', 0.85 * m), TRIM, 0);
		}
		anaAB.count = nab;
		anaAB.setReveal(ctx.reducedMotion ? 1 : easeInOut(clamp(abP, 0, 1)));
		anaAB.upload();
		anaCT.count = nct;
		anaCT.setReveal(markerT < 1 ? markerEase() : ctx.reducedMotion ? 1 : easeInOut(clamp(ctP, 0, 1)));
		anaCT.upload();

		// Rings and the travelling marker, on top of everything.
		let nm = 0;
		const ringAt = (i: number, extra: number, color: [number, number, number, number], width = 1.4) => {
			if (!(pw[i] > 0)) return;
			marks.set(nm++, { x: WP[i][0], y: WP[i][1], z: WP[i][2], radius: pr[i] + extra, ring: width, color });
		};
		ringAt(selected, 4, tone('text', 0.95 * mul(CATEGORIES[WORDS[selected].category].tone)));
		if (hover >= 0 && hover !== selected) ringAt(hover, 3, tone('text', 0.6));
		if (ana && anaDone) {
			ringAt(anaFound, 5, tone('ok', 1 * mul('ok')), 1.6);
			marks.set(nm++, { x: anaTip[0], y: anaTip[1], z: anaTip[2], radius: 7, ring: 1.6, color: tone('ok', 1 * mul('ok')) });
		}
		if (ana && !anaDone) {
			const e = markerEase();
			const c = WP[ana.c];
			const mx = c[0] + (anaTip[0] - c[0]) * e;
			const my = c[1] + (anaTip[1] - c[1]) * e;
			const mz = c[2] + (anaTip[2] - c[2]) * e;
			marks.set(nm++, { x: mx, y: my, z: mz, radius: 9, ring: 1.4, color: tone('text', 0.6) });
			marks.set(nm++, { x: mx, y: my, z: mz, radius: 4.5, color: tone('text', 1) });
		}
		marks.count = nm;
		marks.upload();

		const labels = buildLabels({ relIdx, relSet, anaSet, nbrSet, ana, focusMode, panelX0, ins });
		drawPanel(labels, panelX0, panelW, ins);
		ctx.labels.set(labels);
		emitReadouts();
	}

	/** Word labels go right of the dot, else left, else are hidden; nearer words claim space first. */
	function buildLabels(o: {
		relIdx: (typeof RELATIONS)[number] | null;
		relSet: Set<number>;
		anaSet: Set<number>;
		nbrSet: Set<number>;
		ana: (typeof ANALOGIES)[number] | null;
		focusMode: boolean;
		panelX0: number;
		ins: { top: number; left: number; bottom: number; right: number };
	}): Label[] {
		const v = ctx.view;
		const labels: Label[] = [];
		const boxes: Box[] = [];
		const free: Box = { x0: o.ins.left, x1: o.panelX0 - 8, y0: o.ins.top, y1: v.height - o.ins.bottom };
		const inside = (b: Box) => b.x0 >= free.x0 && b.x1 <= free.x1 && b.y0 >= free.y0 && b.y1 <= free.y1;
		const camD = rig.current.distance;
		const depthT = (i: number) => clamp(1 - (pw[i] - (camD - 1.7)) / 3.4, 0, 1);

		// In a relation, start words go on the side the arrows come from and end words on the side they point to.
		const prefer = new Map<number, 'l' | 'r'>();
		if (o.relIdx) {
			let avg = 0;
			for (const [a, b] of o.relIdx.pairs) avg += px[b] - px[a];
			const from = avg >= 0 ? 'l' : 'r';
			const to = from === 'l' ? 'r' : 'l';
			for (const [a, b] of o.relIdx.pairs) {
				prefer.set(a, from);
				prefer.set(b, to);
			}
		}

		const shown = new Set<number>();
		const place = (i: number, opacity: number, force: boolean) => {
			if (i < 0 || shown.has(i) || !(pw[i] > 0)) return;
			const text = WORDS[i].word;
			const size = 12;
			const w = text.length * size * 0.58 + 4;
			const h = size + 3;
			const r = pr[i] + 4;
			const pref = prefer.get(i) ?? 'r';
			const sides: ('l' | 'r')[] = pref === 'r' ? ['r', 'l'] : ['l', 'r'];
			let chosen: { box: Box; side: 'l' | 'r' } | null = null;
			for (const side of sides) {
				const x0 = side === 'r' ? px[i] + r : px[i] - r - w;
				const box = { x0, x1: x0 + w, y0: py[i] - h / 2, y1: py[i] + h / 2 };
				if (inside(box) && !boxes.some((b) => overlaps(b, box))) {
					chosen = { box, side };
					break;
				}
			}
			if (!chosen) {
				if (!force) return;
				const x0 = px[i] + r;
				chosen = { box: { x0, x1: x0 + w, y0: py[i] - h / 2, y1: py[i] + h / 2 }, side: 'r' };
			}
			shown.add(i);
			boxes.push(chosen.box);
			labels.push({
				id: `w-${i}`,
				x: chosen.side === 'r' ? chosen.box.x0 : chosen.box.x1,
				y: py[i],
				text,
				tone: 'text',
				size,
				align: chosen.side === 'r' ? 'left' : 'right',
				opacity: clamp(opacity, 0, 1) * mul(CATEGORIES[WORDS[i].category].tone)
			});
		};
		const byNear = (ids: Iterable<number>) => [...ids].sort((a, b) => pw[a] - pw[b]);

		place(hover, 1, true);
		place(selected, 1, true);
		for (const i of byNear(new Set([...o.relSet, ...o.nbrSet, ...o.anaSet]))) place(i, 1, false);
		if (o.nbrSet.size === 0 && !o.relIdx && !o.ana) {
			for (const i of byNear(LANDMARKS.flat())) place(i, 0.45 + 0.55 * depthT(i), false);
		}

		// Group names under the nearest part of each ring on the floor.
		if (ringP > 0.05) {
			CATEGORIES.forEach((cat, k) => {
				let bx = NaN;
				let by = -Infinity;
				for (let m = 0; m < 24; m++) {
					const [x, y, w] = project(...ringPoint(k, (m / 24) * Math.PI * 2));
					if (w > 0 && y > by) {
						by = y;
						bx = x;
					}
				}
				if (!Number.isFinite(bx)) return;
				const text = cat.name;
				const w = text.length * 12 * 0.58 + 4;
				const y = by + 14;
				const box = { x0: bx - w / 2, x1: bx + w / 2, y0: y - 8, y1: y + 8 };
				if (!inside(box) || boxes.some((b) => overlaps(b, box))) return;
				boxes.push(box);
				labels.push({
					id: `cat-${k}`,
					x: bx,
					y,
					text,
					tone: cat.tone,
					size: 12,
					opacity: clamp(ringP, 0, 1) * mul(cat.tone) * (o.focusMode ? 0.5 : 1)
				});
			});
		}
		return labels;
	}

	/** The flat side panel: the selected word's 100 numbers, then its closest words or the analogy sum. */
	function drawPanel(labels: Label[], x0: number, w: number, ins: { top: number; bottom: number }) {
		const v = ctx.view;
		panelView.setSize(v.width, v.height, v.dpr);
		panelCam.viewHeight = v.height;
		panelCam.center = [v.width / 2, -v.height / 2];
		panelView.setViewProj(panelCam.matrix(v.width / Math.max(1, v.height)));
		panelView.flush();

		const catTone = CATEGORIES[WORDS[selected].category].tone;
		const m = mul(catTone);
		const top = ins.top + 4;
		labels.push({ id: 'p-word', x: x0, y: top + 10, text: WORDS[selected].word, tone: catTone, size: 18, align: 'left', opacity: m });
		labels.push({ id: 'p-cap-vec', x: x0, y: top + 34, text: 'its 100 numbers', tone: 'muted', size: 11, align: 'left', opacity: m });

		let nq = 0;
		const cell = (w - (STRIP_COLS - 1) * STRIP_GAP) / STRIP_COLS;
		const pitch = cell + STRIP_GAP;
		const stripTop = top + 46;
		const vec = VECTORS[selected];
		for (let r = 0; r < STRIP_ROWS; r++) {
			nq = writeStrip(strip, nq, vec.slice(r * STRIP_COLS, (r + 1) * STRIP_COLS), {
				x: x0 + w / 2,
				y: -(stripTop + cell / 2 + r * pitch),
				cell,
				gap: STRIP_GAP,
				maxAbs: 1.6,
				intensity: 0.9,
				alpha: mul('data'),
				radius: 1.5
			});
		}
		const below = stripTop + STRIP_ROWS * pitch + 26;

		if (showAnalogy >= 0.5) {
			const ana = ANALOGIES[analogyIdx];
			const done = markerT >= 1;
			// "king − man + woman" is c - a + b: the same sum as b - a + c.
			const rows: [string, Tone][] = [
				[WORDS[ana.c].word, 'text'],
				[`− ${WORDS[ana.a].word}`, 'param'],
				[`+ ${WORDS[ana.b].word}`, 'param'],
				[done ? `≈ ${WORDS[anaFound].word}` : '≈ ?', 'ok']
			];
			labels.push({ id: 'p-cap-sum', x: x0, y: below, text: 'the sum', tone: 'muted', size: 11, align: 'left', opacity: mul('param') });
			rows.forEach(([text, t], i) => {
				labels.push({
					id: `p-sum-${i}`,
					x: x0,
					y: below + 24 + i * 24 + (i === 3 ? 4 : 0),
					text,
					tone: t,
					size: i === 3 ? 16 : 13,
					align: 'left',
					opacity: mul(t === 'text' ? 'param' : t)
				});
			});
			if (done && anaFound !== ana.expected) {
				labels.push({
					id: 'p-miss',
					x: x0,
					y: below + 24 + 4 * 24 + 8,
					text: `not ${WORDS[ana.expected].word}`,
					tone: 'text',
					size: 12,
					align: 'left',
					opacity: mul('ok')
				});
			}
		} else if (nbrs.length > 0) {
			const mData = mul('data');
			labels.push({ id: 'p-cap-near', x: x0, y: below, text: 'closest words', tone: 'muted', size: 11, align: 'left', opacity: mData });
			const rowTop = below + 22;
			const avail = ctx.view.height - ins.bottom - rowTop;
			const rowPitch = clamp(avail / nbrs.length, 16, 22);
			const bx0 = x0 + 70;
			const trackW = w - 70 - 38;
			nbrs.forEach((nb, i) => {
				const y = rowTop + i * rowPitch;
				labels.push({ id: `p-nw-${i}`, x: x0, y, text: WORDS[nb.index].word, tone: 'text', size: 12, align: 'left', opacity: mData });
				strip.set(nq++, { x: bx0 + trackW / 2, y: -y, w: trackW, h: 6, color: tone('text', 0.1 * mData), radius: 3 });
				const len = Math.max(trackW * clamp(nb.similarity, 0, 1), 2);
				strip.set(nq++, { x: bx0 + len / 2, y: -y, w: len, h: 6, color: tone('data', 0.85 * mData), radius: 3 });
				labels.push({
					id: `p-ns-${i}`,
					x: x0 + w,
					y,
					text: nb.similarity.toFixed(2),
					tone: 'muted',
					mono: true,
					size: 11,
					align: 'right',
					opacity: mData
				});
			});
		}
		strip.count = nq;
		strip.upload();
	}

	function emitReadouts() {
		const w = WORDS[selected].word;
		if (w !== lastWordOut) {
			lastWordOut = w;
			ctx.emit('word', w);
		}
		// Whatever the stage panel names as closest: the analogy answer, else the first listed neighbor.
		const best = showAnalogy >= 0.5 ? (markerT >= 1 ? WORDS[anaFound].word : '—') : nbrs.length > 0 ? WORDS[nbrs[0].index].word : '—';
		if (best !== lastBestOut) {
			lastBestOut = best;
			ctx.emit('best', best);
		}
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			rig = new OrbitRig({ minElevation: 0.08, maxElevation: 1.45 });
			orbit = new SceneOrbit(c, rig, { draggable: true });
			panelView = new ViewUniforms(c.gpu);
			panelCam = new Camera2D();
			const flat = { gpu: c.gpu, target: c.target, view: panelView };
			floor = new Lines(c, { capacity: 24, depthTest: true });
			rings = new Lines(c, { capacity: CATEGORIES.length * RING_SEGMENTS + 8, depthTest: true });
			stems = new Lines(c, { capacity: N + 4, depthTest: true });
			links = new Lines(c, { capacity: 48, depthTest: true });
			anaAB = new Lines(c, { capacity: 8, depthTest: true });
			anaCT = new Lines(c, { capacity: 8, depthTest: true });
			shadows = new Shadows(c, { capacity: N });
			dots = new Points(c, { capacity: N, shade: 'sphere', size: 'world', depthTest: true, depthWrite: true });
			marks = new Points(c, { capacity: 12 });
			strip = new Quads(flat, { capacity: 140 });
		},
		setParams(p) {
			const prev = { colors, neighbors, relation, analogyIdx, showAnalogy, spin, wordParam, scenario };
			colors = p.colors ?? colors;
			neighbors = Math.round(p.neighbors ?? neighbors);
			relation = clamp(Math.round(p.relation ?? relation), 0, RELATIONS.length);
			analogyIdx = clamp(Math.round(p.analogy ?? analogyIdx), 0, ANALOGIES.length - 1);
			showAnalogy = Math.round(p.showAnalogy ?? showAnalogy);
			spin = Math.round(p.spin ?? spin);
			wordParam = clamp(Math.round(p.word ?? wordParam), 0, N - 1);
			scenario = Math.round(p.scenario ?? scenario);
			if (!paramsSeen || wordParam !== prev.wordParam || scenario !== prev.scenario) {
				selected = wordParam;
				hover = -1;
				markerT = 1;
			}
			if (paramsSeen && showAnalogy >= 0.5 && analogyIdx !== prev.analogyIdx) {
				markerT = ctx.reducedMotion ? 1 : 0;
				if (markerT >= 1) landAnalogy();
			}
			paramsSeen = true;
		},
		update(fr) {
			update3D(fr.dt);
		},
		draw(pass) {
			floor.draw(pass);
			shadows.draw(pass);
			rings.draw(pass);
			dots.draw(pass);
			stems.draw(pass);
			links.draw(pass);
			anaAB.draw(pass);
			anaCT.draw(pass);
			marks.draw(pass);
			strip.draw(pass);
		},
		command(id) {
			if (orbit.command(id)) return;
			if (id === 'play' && showAnalogy >= 0.5) {
				markerT = 0;
				abP = ctx.reducedMotion ? 1 : 0;
				ctP = 1;
			}
		},
		highlight(t) {
			hl = t;
		},
		pointer(e) {
			if (e.type === 'down') pressed = true;
			else if (e.type === 'up' || e.type === 'leave') pressed = false;
			const r = orbit.pointer(e);
			if (r === 'click') {
				const i = pickAt(e.x, e.y);
				if (i >= 0) selected = i;
				hover = pickAt(e.x, e.y);
			} else if (r === 'drag') {
				hover = -1;
			} else if (e.type === 'leave') {
				hover = -1;
			} else {
				hover = pickAt(e.x, e.y);
			}
		},
		dispose() {
			floor.destroy();
			rings.destroy();
			stems.destroy();
			links.destroy();
			anaAB.destroy();
			anaCT.destroy();
			shadows.dispose();
			dots.destroy();
			marks.destroy();
			strip.destroy();
			panelView.destroy();
		}
	};
	return scene;
};

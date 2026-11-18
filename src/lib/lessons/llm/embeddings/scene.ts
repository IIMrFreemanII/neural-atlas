import { Camera2D } from '../../../gpu/camera.ts';
import { mixColor, tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import { Quads } from '../../../gpu/primitives/quads.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { analogy, nearest } from '../../../ml/embedding.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { pxPerWorld } from '../shared/chips.ts';
import { writeStrip } from '../shared/strip.ts';
import { ANALOGIES, CATEGORIES, CENTROIDS_3D, POINTS, RELATIONS, VECTORS, WORDS, indexOf, project3D } from './data.ts';

const WORLD: WorldRect = { x0: -2.6, x1: 2.6, y0: -2.1, y1: 2.1 };
const CX = -0.75;
const CY = -0.1;
const R = 1.7;
const FADE_IN = 1.45;
const FADE_OUT = 1.6;
const VIEW_TAU = 0.35; // seconds
const PITCH0 = 0.35;
const SPIN_SPEED = 0.15; // rad/s
const SPIN_SLOW = 0.05; // while a relation or analogy is showing
const DRAG_RAD_PER_PX = 0.01;
const DRAG_SLOP = 4;
const HOVER_PX = 14;
const PLAY_SECONDS = 1.2;
const N = WORDS.length;

// Side panel, world units.
const PX0 = 1.15;
const PX1 = 2.5;
const PANEL_Y_WORD = 1.75;
const PANEL_Y_CAP = 1.5;
const PANEL_Y_STRIP = 1.33;
const STRIP_CELL = 0.12;
const PANEL_Y_NEAR = 1.02;
const BAR_X0 = 1.72;
const BAR_MAX = 0.5;

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

export const createEmbeddingsScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let lines: Lines;
	let dots: Points;
	let quads: Quads;

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
	let yaw = 0.5;
	let pitch = PITCH0;
	let selected = indexOf('cat');
	let hover = -1;
	let down = false;
	let dragging = false;
	let lastX = 0;
	let lastY = 0;
	let downX = 0;
	let downY = 0;
	let markerT = 1;
	let dirty = true;
	let lastScale = 0;
	let lastW = 0;
	let lastH = 0;

	// view transform on top of the rotation: zoom on what is being shown
	let zoomNow = 1;
	let fxNow = 0;
	let fyNow = 0;
	let viewSnap = true;

	// per frame projection of the 74 points
	const sx = new Float64Array(N); // world x
	const sy = new Float64Array(N); // world y
	const sz = new Float64Array(N); // depth, toward the viewer positive
	const px = new Float64Array(N); // css px x
	const py = new Float64Array(N); // css px y
	const fade = new Float64Array(N); // 0..1, 0 when outside the stage area the cloud may use
	const order = Array.from({ length: N }, (_, i) => i);
	let cosY = 1;
	let sinY = 0;
	let cosP = 1;
	let sinP = 0;

	// cached derived values
	let nbrKey = '';
	let nbrs: { index: number; similarity: number }[] = [];
	let anaKey = -1;
	let anaVec3: [number, number, number] = [0, 0, 0];
	let anaFound = 0;

	// emitted readouts
	let lastWordOut = '';
	let lastBestOut = '';

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	function setAngles() {
		cosY = Math.cos(yaw);
		sinY = Math.sin(yaw);
		cosP = Math.cos(pitch);
		sinP = Math.sin(pitch);
	}

	/** Rotate a point of the 3D picture by yaw then pitch, in unit-scaled coordinates. Returns x, y, depth. */
	function rotUnit(x: number, y: number, z: number): [number, number, number] {
		const x1 = x * cosY + z * sinY;
		const z1 = -x * sinY + z * cosY;
		const y2 = y * cosP - z1 * sinP;
		const z2 = y * sinP + z1 * cosP;
		return [x1, y2, z2];
	}

	/** Rotate, then apply the zoom and focus. Returns world x, world y, depth. */
	function rot(x: number, y: number, z: number): [number, number, number] {
		const [a, b, c] = rotUnit(x, y, z);
		return [CX + R * zoomNow * (a - fxNow), CY + R * zoomNow * (b - fyNow), c];
	}

	/** 1 inside the usable disc around the cloud center, fading to 0 at FADE_OUT world units. */
	function fadeAt(wx: number, wy: number): number {
		return clamp((FADE_OUT - Math.hypot(wx - CX, wy - CY)) / (FADE_OUT - FADE_IN), 0, 1);
	}

	/** Where the view wants to be: zoomed on the shown relation or analogy, else the whole cloud. */
	function viewTarget(): { zoom: number; fx: number; fy: number } {
		const pts: [number, number][] = [];
		if (relation > 0) {
			for (const [a, b] of RELATIONS[relation - 1].pairs) {
				for (const i of [a, b]) {
					const [x, y] = rotUnit(POINTS[i][0], POINTS[i][1], POINTS[i][2]);
					pts.push([x, y]);
				}
			}
		} else if (showAnalogy >= 0.5) {
			const an = ANALOGIES[analogyIdx];
			for (const i of [an.a, an.b, an.c, anaFound]) {
				const [x, y] = rotUnit(POINTS[i][0], POINTS[i][1], POINTS[i][2]);
				pts.push([x, y]);
			}
			const [tx, ty] = rotUnit(anaVec3[0], anaVec3[1], anaVec3[2]);
			pts.push([tx, ty]);
		}
		if (pts.length === 0) return { zoom: 1, fx: 0, fy: 0 };
		let fx = 0;
		let fy = 0;
		for (const [x, y] of pts) {
			fx += x / pts.length;
			fy += y / pts.length;
		}
		let far = 0;
		for (const [x, y] of pts) far = Math.max(far, Math.hypot(x - fx, y - fy));
		return { zoom: clamp(0.8 / Math.max(far, 1e-6), 1, 2.6), fx, fy };
	}

	function projectAll() {
		setAngles();
		for (let i = 0; i < N; i++) {
			const p = POINTS[i];
			const [x, y, z] = rot(p[0], p[1], p[2]);
			sx[i] = x;
			sy[i] = y;
			sz[i] = z;
			fade[i] = fadeAt(x, y);
			const [cx, cy] = ctx.view.project(x, y);
			px[i] = cx;
			py[i] = cy;
		}
		order.sort((a, b) => sz[a] - sz[b]);
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
			anaFound = nearest(VECTORS, v, 1, [a, b, c])[0].index;
		}
	}

	function depthT(i: number): number {
		return clamp((sz[i] + 1) / 2, 0, 1);
	}

	function pickAt(x: number, y: number): number {
		let best = -1;
		let bestScore = HOVER_PX;
		for (let i = 0; i < N; i++) {
			if (fade[i] <= 0) continue;
			const d = Math.hypot(px[i] - x, py[i] - y) - 2 * depthT(i);
			if (d < bestScore) {
				bestScore = d;
				best = i;
			}
		}
		return best;
	}

	/** Line segments for an arrow from (ax, ay) to (bx, by), world units; the tip stops `trimPx` short. */
	function arrow(
		n: number,
		ax: number,
		ay: number,
		bx: number,
		by: number,
		width: number,
		color: [number, number, number, number],
		scale: number,
		trimPx = 0,
		head = true
	): number {
		const dx = bx - ax;
		const dy = by - ay;
		const len = Math.hypot(dx, dy);
		if (len < 1e-4) return n;
		const ux = dx / len;
		const uy = dy / len;
		const tx = bx - (ux * trimPx) / scale;
		const ty = by - (uy * trimPx) / scale;
		lines.set(n++, { a: [ax, ay], b: [tx, ty], width, color });
		if (head) {
			const hl2 = 9 / scale;
			for (const s of [-1, 1]) {
				const ang = s * 0.5;
				const c = Math.cos(ang);
				const sn = Math.sin(ang);
				const hx = -(ux * c - uy * sn);
				const hy = -(ux * sn + uy * c);
				lines.set(n++, { a: [tx, ty], b: [tx + hx * hl2, ty + hy * hl2], width, color });
			}
		}
		return n;
	}

	let placed: { x0: number; x1: number; y: number }[] = [];

	function wordLabel(labels: Label[], i: number, opacity: number, panelLeftPx: number, side = 0) {
		const word = WORDS[i].word;
		const t = depthT(i);
		const r = 3 + 3.5 * t;
		const wEst = word.length * 12 * 0.56;
		const flip = side < 0 || px[i] + r + 5 + wEst > panelLeftPx;
		const lx = px[i] + (flip ? -(r + 5) : r + 5);
		const x0 = flip ? lx - wEst : lx;
		const x1 = x0 + wEst;
		// Nudge the label up or down until it no longer sits on one placed before it.
		let ly = py[i];
		for (let k = 0; k < 8; k++) {
			const hit = placed.some((p) => Math.abs(p.y - ly) < 12 && p.x0 < x1 && x0 < p.x1);
			if (!hit) break;
			ly = py[i] + (k % 2 === 0 ? 1 : -1) * 12 * Math.ceil((k + 1) / 2);
		}
		placed.push({ x0, x1, y: ly });
		labels.push({
			id: `w-${i}`,
			x: lx,
			y: ly,
			text: word,
			tone: 'text',
			size: 12,
			align: flip ? 'right' : 'left',
			opacity: clamp(opacity, 0, 1) * fade[i] * mul(CATEGORIES[WORDS[i].category].tone)
		});
	}

	/** The animation arrived: the word it found becomes the selected word. */
	function landAnalogy() {
		if (showAnalogy < 0.5) return;
		ensureDerived();
		selected = anaFound;
	}

	function frameUpdate(dt: number) {
		if (spin >= 0.5 && !down && !ctx.reducedMotion && dt > 0) {
			yaw += (relation > 0 || showAnalogy >= 0.5 ? SPIN_SLOW : SPIN_SPEED) * dt;
			dirty = true;
		}
		if (markerT < 1 && dt > 0) {
			markerT = Math.min(1, markerT + dt / PLAY_SECONDS);
			if (markerT >= 1) landAnalogy();
			dirty = true;
		}
		// Ease the zoom and focus toward what is being shown.
		setAngles();
		ensureDerived();
		const t = viewTarget();
		if (viewSnap || ctx.reducedMotion) {
			viewSnap = false;
			zoomNow = t.zoom;
			fxNow = t.fx;
			fyNow = t.fy;
			dirty = true;
		} else {
			const k = 1 - Math.exp(-Math.max(0, dt) / VIEW_TAU);
			const dz = t.zoom - zoomNow;
			const dx = t.fx - fxNow;
			const dy = t.fy - fyNow;
			const gap = Math.abs(dz) + Math.abs(dx) + Math.abs(dy);
			if (gap > 1e-4) {
				const done = k >= 1 || gap < 1e-3;
				zoomNow = done ? t.zoom : zoomNow + dz * k;
				fxNow = done ? t.fx : fxNow + dx * k;
				fyNow = done ? t.fy : fyNow + dy * k;
				dirty = true;
			}
		}
	}

	function draw3D(scale: number): Label[] {
		const labels: Label[] = [];
		const nbrSet = new Set(nbrs.map((n) => n.index));
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
		const dimOthers = relIdx !== null || ana !== null;
		const inFocus = (i: number) => (relIdx ? relSet.has(i) : ana ? anaSet.has(i) : true);

		// Lines first (behind the dots).
		let nl = 0;
		const axis = tone('muted', 1, 0.25);
		const axes: [number, number, number][] = [
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1]
		];
		const [oX, oY] = rot(0, 0, 0);
		for (const [ax, ay, az] of axes) {
			// Keep only the part of the axis inside the usable disc.
			const [qx, qy] = rot(ax, ay, az);
			const bx = qx - oX;
			const by = qy - oY;
			const ex = oX - CX;
			const ey = oY - CY;
			const qa = bx * bx + by * by;
			const qb = 2 * (ex * bx + ey * by);
			const qc = ex * ex + ey * ey - FADE_IN * FADE_IN;
			const disc = qb * qb - 4 * qa * qc;
			if (qa < 1e-9 || disc < 0) continue;
			const t0 = Math.max(-1.15, (-qb - Math.sqrt(disc)) / (2 * qa));
			const t1 = Math.min(1.15, (-qb + Math.sqrt(disc)) / (2 * qa));
			if (t1 <= t0) continue;
			lines.set(nl++, { a: [oX + bx * t0, oY + by * t0], b: [oX + bx * t1, oY + by * t1], width: 1, color: axis });
		}
		nbrs.forEach((nb, rank) => {
			const alpha = (0.72 - rank * 0.045) * mul('data') * Math.min(fade[selected], fade[nb.index]);
			lines.set(nl++, {
				a: [sx[selected], sy[selected]],
				b: [sx[nb.index], sy[nb.index]],
				width: 1.5,
				color: tone('data', 0.9, alpha)
			});
		});
		if (relIdx) {
			for (const [a, b] of relIdx.pairs) {
				const c = tone('text', 0.8, 0.8 * Math.min(fade[a], fade[b]));
				nl = arrow(nl, sx[a], sy[a], sx[b], sy[b], 1.5, c, scale, 7);
			}
		}
		let tipX = 0;
		let tipY = 0;
		if (ana) {
			const m = mul('param');
			[tipX, tipY] = rot(anaVec3[0], anaVec3[1], anaVec3[2]);
			const f1 = Math.min(fade[ana.a], fade[ana.b]);
			nl = arrow(nl, sx[ana.a], sy[ana.a], sx[ana.b], sy[ana.b], 2, tone('param', 1, 0.9 * m * f1), scale, 7);
			// The same offset again, starting at c.
			const e = easeInOut(markerT);
			const bx = sx[ana.c] + (tipX - sx[ana.c]) * e;
			const by = sy[ana.c] + (tipY - sy[ana.c]) * e;
			const f2 = Math.min(fade[ana.c], fadeAt(tipX, tipY));
			nl = arrow(nl, sx[ana.c], sy[ana.c], bx, by, 2, tone('param', 1, 0.8 * m * f2), scale, anaDone ? 5 : 0, anaDone);
		}
		lines.count = nl;
		lines.upload();

		// Dots, back to front.
		let n = 0;
		for (const i of order) {
			const t = depthT(i);
			const cat = CATEGORIES[WORDS[i].category];
			const base = mixColor(tone('muted', 1.7, 1), tone(cat.tone, cat.tone === 'text' ? 0.7 : 1.15, 1), colors);
			const dim = (dimOthers && !inFocus(i) ? 0.35 : 1) * mul(cat.tone);
			const a = (0.45 + 0.55 * t) * dim * fade[i];
			dots.set(n++, { x: sx[i], y: sy[i], radius: 3 + 3.5 * t, color: [base[0], base[1], base[2], a] });
		}
		const ringAt = (i: number, extra: number, color: [number, number, number, number], width = 1.4) => {
			const c: [number, number, number, number] = [color[0], color[1], color[2], color[3] * fade[i]];
			dots.set(n++, { x: sx[i], y: sy[i], radius: 3 + 3.5 * depthT(i) + extra, ring: width, color: c });
		};
		if (selected >= 0) ringAt(selected, 4, tone('text', 1, 0.9 * mul(CATEGORIES[WORDS[selected].category].tone)));
		if (hover >= 0 && hover !== selected) ringAt(hover, 3, tone('text', 0.8, 0.6));
		if (ana && anaDone) {
			dots.set(n++, { x: sx[anaFound], y: sy[anaFound], radius: 17, softness: 1, color: tone('ok', 1, 0.4 * mul('ok') * fade[anaFound]) });
			ringAt(anaFound, 5, tone('ok', 1.1, 0.95 * mul('ok')), 1.6);
			dots.set(n++, { x: tipX, y: tipY, radius: 7, ring: 1.6, color: tone('ok', 1.1, 0.95 * mul('ok') * fadeAt(tipX, tipY)) });
		}
		if (ana && !anaDone) {
			const e = easeInOut(markerT);
			const mx = sx[ana.c] + (tipX - sx[ana.c]) * e;
			const my = sy[ana.c] + (tipY - sy[ana.c]) * e;
			dots.set(n++, { x: mx, y: my, radius: 12, softness: 1, color: tone('text', 0.8, 0.25 * fadeAt(mx, my)) });
			dots.set(n++, { x: mx, y: my, radius: 4.5, color: tone('text', 1, fadeAt(mx, my)) });
		}
		dots.count = n;
		dots.upload();

		// Labels.
		const panelLeft = ctx.view.project(PX0, 0)[0] - 10;
		placed = [];
		const shown = new Set<number>();
		// In a relation, start words go on the side the arrows come from and end words on the side they point to.
		const sides = new Map<number, number>();
		if (relIdx) {
			let avg = 0;
			for (const [a, b] of relIdx.pairs) avg += sx[b] - sx[a];
			const from = avg >= 0 ? -1 : 1;
			for (const [a, b] of relIdx.pairs) {
				sides.set(a, from);
				sides.set(b, -from);
			}
		}
		const add = (i: number, opacity: number) => {
			if (i < 0 || shown.has(i)) return;
			shown.add(i);
			wordLabel(labels, i, opacity, panelLeft, sides.get(i) ?? 0);
		};
		for (const i of relSet) add(i, 1);
		add(selected, 1);
		add(hover, 1);
		for (const i of nbrSet) add(i, 1);
		for (const i of anaSet) add(i, 1);
		if (neighbors < 0.5 && !relIdx && !ana) {
			for (const pair of LANDMARKS) for (const i of pair) add(i, 0.3 + 0.7 * depthT(i));
		}

		// Category names at their centroids, pushed outward from the cloud center.
		CATEGORIES.forEach((cat, k) => {
			const c = CENTROIDS_3D[k];
			const len = Math.hypot(c[0], c[1], c[2]) || 1;
			const push = (len + 0.3) / len;
			const [wx, wy, wz] = rot(c[0] * push, c[1] * push, c[2] * push);
			const [cx, cy] = ctx.view.project(wx, wy);
			labels.push({
				id: `cat-${k}`,
				x: cx,
				y: cy,
				text: cat.name,
				tone: cat.tone,
				size: 12,
				opacity: fadeAt(wx, wy) * clamp(colors, 0, 1) * (0.45 + 0.55 * clamp((wz + 1) / 2, 0, 1)) * mul(cat.tone) * (dimOthers ? 0.5 : 1)
			});
		});
		return labels;
	}

	function drawPanel(labels: Label[]) {
		const v = ctx.view;
		const catTone = CATEGORIES[WORDS[selected].category].tone;
		const m = mul(catTone);
		const [wx, wy] = v.project(PX0, PANEL_Y_WORD);
		labels.push({
			id: 'p-word',
			x: wx,
			y: wy,
			text: WORDS[selected].word,
			tone: catTone,
			size: 18,
			align: 'left',
			opacity: m
		});
		const [cx, cy] = v.project(PX0, PANEL_Y_CAP);
		labels.push({ id: 'p-cap-vec', x: cx, y: cy, text: 'its vector', tone: 'muted', size: 11, align: 'left', opacity: m });

		let nq = writeStrip(quads, 0, VECTORS[selected], {
			x: PX0 + (10 * STRIP_CELL + 9 * STRIP_CELL * 0.12) / 2,
			y: PANEL_Y_STRIP,
			cell: STRIP_CELL,
			maxAbs: 1.6,
			intensity: 0.9,
			alpha: mul('data')
		});

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
			const [capx, capy] = v.project(PX0, PANEL_Y_NEAR);
			labels.push({ id: 'p-cap-sum', x: capx, y: capy, text: 'the sum', tone: 'muted', size: 11, align: 'left', opacity: mul('param') });
			rows.forEach(([text, t], i) => {
				const [lx, ly] = v.project(PX0, 0.78 - i * 0.24 - (i === 3 ? 0.06 : 0));
				labels.push({
					id: `p-sum-${i}`,
					x: lx,
					y: ly,
					text,
					tone: t,
					size: i === 3 ? 16 : 13,
					align: 'left',
					opacity: mul(t === 'text' ? 'param' : t)
				});
			});
		} else if (nbrs.length > 0) {
			const mData = mul('data');
			const [capx, capy] = v.project(PX0, PANEL_Y_NEAR);
			labels.push({ id: 'p-cap-near', x: capx, y: capy, text: 'closest words', tone: 'muted', size: 11, align: 'left', opacity: mData });
			nbrs.forEach((nb, i) => {
				const y = 0.8 - i * 0.2;
				const [lx, ly] = v.project(PX0, y);
				labels.push({ id: `p-nw-${i}`, x: lx, y: ly, text: WORDS[nb.index].word, tone: 'text', size: 12, align: 'left', opacity: mData });
				const len = BAR_MAX * clamp(nb.similarity, 0, 1);
				quads.set(nq++, {
					x: BAR_X0 + len / 2,
					y,
					w: Math.max(len, 0.01),
					h: 0.05,
					color: tone('data', 0.7, 0.9 * mData),
					radius: 2
				});
				const [rx, ry] = v.project(PX1, y);
				labels.push({
					id: `p-ns-${i}`,
					x: rx,
					y: ry,
					text: nb.similarity.toFixed(2),
					tone: 'muted',
					mono: true,
					size: 11,
					align: 'right',
					opacity: mData
				});
			});
		}
		quads.count = nq;
		quads.upload();
	}

	function emitReadouts() {
		const w = WORDS[selected].word;
		if (w !== lastWordOut) {
			lastWordOut = w;
			ctx.emit('word', w);
		}
		const best = showAnalogy >= 0.5 && markerT >= 1 ? WORDS[anaFound].word : '—';
		if (best !== lastBestOut) {
			lastBestOut = best;
			ctx.emit('best', best);
		}
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			lines = new Lines(c, { capacity: 72, blend: 'alpha' });
			dots = new Points(c, { capacity: 128, blend: 'alpha' });
			quads = new Quads(c, { capacity: 40, blend: 'alpha' });
			setAngles();
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
			dirty = true;
		},
		update(fr) {
			const v = ctx.view;
			fitCamera(camera, v, WORLD, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));
			const scale = pxPerWorld(v, camera);
			if (scale !== lastScale || v.width !== lastW || v.height !== lastH) {
				lastScale = scale;
				lastW = v.width;
				lastH = v.height;
				dirty = true;
			}
			frameUpdate(fr.dt);
			if (!dirty) return;
			dirty = false;
			ensureDerived();
			projectAll();
			const labels = draw3D(scale);
			drawPanel(labels);
			ctx.labels.set(labels);
			emitReadouts();
		},
		draw(pass) {
			lines.draw(pass);
			dots.draw(pass);
			quads.draw(pass);
		},
		command(id) {
			if (id === 'play' && showAnalogy >= 0.5) {
				markerT = ctx.reducedMotion ? 1 : 0;
				if (markerT >= 1) landAnalogy();
				dirty = true;
			}
		},
		highlight(t) {
			hl = t;
			dirty = true;
		},
		pointer(e) {
			dirty = true;
			if (e.type === 'down') {
				down = true;
				dragging = false;
				lastX = downX = e.x;
				lastY = downY = e.y;
			} else if (e.type === 'move') {
				if (down) {
					if (!dragging && Math.hypot(e.x - downX, e.y - downY) > DRAG_SLOP) dragging = true;
					if (dragging) {
						yaw += (e.x - lastX) * DRAG_RAD_PER_PX;
						pitch = clamp(pitch + (e.y - lastY) * DRAG_RAD_PER_PX, -1.2, 1.2);
					}
					lastX = e.x;
					lastY = e.y;
				} else {
					hover = pickAt(e.x, e.y);
				}
			} else if (e.type === 'up') {
				if (down && !dragging) {
					const i = pickAt(e.x, e.y);
					if (i >= 0) selected = i;
				}
				down = false;
				dragging = false;
				hover = pickAt(e.x, e.y);
			} else {
				down = false;
				dragging = false;
				hover = -1;
			}
		},
		dispose() {
			lines.destroy();
			dots.destroy();
			quads.destroy();
		}
	};
	return scene;
};

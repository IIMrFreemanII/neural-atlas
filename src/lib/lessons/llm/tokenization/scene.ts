import { Camera2D } from '../../../gpu/camera.ts';
import { tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import { Quads } from '../../../gpu/primitives/quads.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { countPairs, encode, segment, splitWords } from '../../../ml/bpe.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { layoutChips, MONO_ADVANCE, pxPerWorld, showToken, type ChipBox } from '../shared/chips.ts';
import { DEFAULT_TEXT, MAX_MERGES, MODEL, TRAIN_MAX, TRAIN_MODEL, TRAIN_TEXT, WORD_LIST, WORDS, wordKey } from './data.ts';

const WORLD: WorldRect = { x0: -2.6, x1: 2.6, y0: -2.1, y1: 2.1 };
const TEXT_X0 = -2.4;
const TEXT_X1 = 2.4;
const TEXT_Y = 0.1; // block center without the chart
const TEXT_Y_CHART = 0.75; // block center above the chart
const CHIP_FONT = 19;
const CHART = { x0: -2.0, x1: 2.0, y0: -1.7, y1: -0.4 };
const EASE_TIME = 0.12; // seconds, chip motion time constant
const FADE_TIME = 0.3; // seconds, view cross-fade
const RUN_PERIOD = 0.8; // seconds between merges while "Keep merging" runs
const BASE_CHARS = MODEL.vocab.length - MODEL.merges.length;
const TRAIN_BASE_CHARS = TRAIN_MODEL.vocab.length - TRAIN_MODEL.merges.length;
const WORD_ID = new Map(WORD_LIST.map((w, i) => [w, i]));

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

interface ChipTarget {
	/** Index of the chip's first character in its text. */
	key: number;
	text: string;
	x: number;
	y: number;
	w: number;
	h: number;
	tone: Tone;
	fill: number;
	/** Small caption under the chip (the token id), when shown. */
	sub?: string;
}

interface Chip extends ChipTarget {
	cx: number;
	cy: number;
	cw: number;
}

/** Chips that ease toward their targets. A chip is identified by its key. */
class ChipField {
	chips: Chip[] = [];
	private byKey = new Map<number, Chip>();

	set(targets: ChipTarget[]): void {
		const next = new Map<number, Chip>();
		const out: Chip[] = [];
		for (const t of targets) {
			const old = this.byKey.get(t.key);
			const c: Chip = { ...t, cx: old ? old.cx : t.x, cy: old ? old.cy : t.y, cw: old ? old.cw : t.w };
			next.set(t.key, c);
			out.push(c);
		}
		this.chips = out;
		this.byKey = next;
	}

	/** Returns true when anything moved. */
	step(dt: number, instant: boolean): boolean {
		const k = instant ? 1 : 1 - Math.exp(-dt / EASE_TIME);
		let moved = false;
		for (const c of this.chips) {
			const dx = c.x - c.cx;
			const dy = c.y - c.cy;
			const dw = c.w - c.cw;
			if (dx === 0 && dy === 0 && dw === 0) continue;
			moved = true;
			if (Math.abs(dx) + Math.abs(dy) + Math.abs(dw) < 0.002) {
				c.cx = c.x;
				c.cy = c.y;
				c.cw = c.w;
			} else {
				c.cx += dx * k;
				c.cy += dy * k;
				c.cw += dw * k;
			}
		}
		return moved;
	}
}

/** Token chips of `words`, wrapped between words (never inside one). Indices are global. */
function layoutWordLines(
	words: string[][],
	o: { x0: number; x1: number; y: number; scale: number; fontPx: number; padPx: number; gapPx: number; heightPx: number; rowGapPx: number }
): ChipBox[] {
	const gap = o.gapPx / o.scale;
	const width = (t: string) =>
		(Math.max(1, [...showToken(t)].length) * o.fontPx * MONO_ADVANCE + o.padPx * 2) / o.scale;
	const wordW = words.map((w) => w.reduce((s, t) => s + width(t), 0) + gap * Math.max(0, w.length - 1));
	const lines: number[][] = [];
	let cur: number[] = [];
	let used = 0;
	words.forEach((w, i) => {
		if (w.length === 0) return;
		if (cur.length > 0 && used + gap + wordW[i] > o.x1 - o.x0) {
			lines.push(cur);
			cur = [];
			used = 0;
		}
		used += (cur.length > 0 ? gap : 0) + wordW[i];
		cur.push(i);
	});
	if (cur.length > 0) lines.push(cur);
	const step = (o.heightPx + o.rowGapPx) / o.scale;
	const boxes: ChipBox[] = [];
	lines.forEach((line, row) => {
		const toks = line.flatMap((i) => words[i]);
		const part = layoutChips(toks, {
			x0: o.x0,
			x1: o.x1,
			y: o.y - row * step,
			scale: o.scale,
			fontPx: o.fontPx,
			padPx: o.padPx,
			gapPx: o.gapPx,
			heightPx: o.heightPx
		});
		for (const b of part) boxes.push({ ...b, index: boxes.length, row });
	});
	return boxes;
}

export const createTokenizationScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let q0: Quads; // view 0 chips
	let q1: Quads; // view 1 chips, bars, learned pieces
	let lines: Lines;
	let dot: Points;

	// params
	let view = 0;
	let mode = 2;
	let merges = MAX_MERGES;
	let tmerges = 0;
	let showIds = true;
	let chart = 0;
	let scenario = 1;
	let hl: Tone | null = null;

	// state
	let text = DEFAULT_TEXT;
	const chips0 = new ChipField();
	const chips1 = new ChipField();
	let tokenCount0 = 0;
	let vocab0 = 0;
	let learned: ChipBox[] = [];
	let pairRows: { label: string; count: number }[] = [];
	let chartCounts: number[] = [];
	let chartDirty = true;
	let mix = 0;
	let chartMix = 0;
	let running = false;
	let runTimer = 0;
	let lastScale = 0;
	let lastW = 0;
	let lastH = 0;
	let dirtyLayout = true;
	let dirty = true;
	let lastTokens = -1;
	let lastVocab = -1;
	let seen = false;

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);
	const scaleNow = () => pxPerWorld(ctx.view, camera);

	function retarget0(scale: number): void {
		const chars = [...text];
		let toks: string[] = [];
		let subs: (string | undefined)[] = [];
		let tones: Tone[] = [];
		if (mode === 0) {
			toks = chars;
			subs = chars.map((c) => String(MODEL.ids.get(c) ?? '?'));
			tones = chars.map((c) => (MODEL.ids.has(c) ? 'data' : 'neg'));
			vocab0 = BASE_CHARS;
		} else if (mode === 1) {
			toks = splitWords(text);
			subs = toks.map((w) => {
				const k = wordKey(w);
				return k === '' ? '?' : WORD_ID.has(k) ? String(WORD_ID.get(k)) : '?';
			});
			tones = toks.map((w) => {
				const k = wordKey(w);
				if (k !== '' && !WORDS.has(k)) return 'neg';
				return [...w].length > 1 ? 'param' : 'data';
			});
			vocab0 = WORDS.size;
		} else {
			const e = encode(text, MODEL, Math.round(merges));
			toks = e.tokens;
			subs = e.ids.map((i) => (i < 0 ? '?' : String(i)));
			tones = e.tokens.map((t, i) => (e.ids[i] < 0 ? 'neg' : [...t].length > 1 ? 'param' : 'data'));
			vocab0 = BASE_CHARS + Math.round(merges);
		}
		tokenCount0 = toks.length;
		const keys: number[] = [];
		let at = 0;
		for (const t of toks) {
			keys.push(at);
			at += [...t].length;
		}
		const gapRows = showIds ? 34 : 14;
		// Wrap between words where possible, so a word's pieces stay on one row. A word starts at a token with a leading space.
		const groups: string[][] = [];
		for (const t of toks) {
			if (groups.length === 0 || t.startsWith(' ')) groups.push([t]);
			else groups[groups.length - 1].push(t);
		}
		const chipLayout = {
			x0: TEXT_X0,
			x1: TEXT_X1,
			y: 0,
			scale,
			fontPx: CHIP_FONT,
			padPx: 6,
			gapPx: 4,
			heightPx: CHIP_FONT + 15,
			rowGapPx: gapRows
		};
		const groupPx = (g: string[]) =>
			g.reduce((sum, t) => sum + [...showToken(t)].length * CHIP_FONT * MONO_ADVANCE + 12, 0) + 4 * (g.length - 1);
		const fits = groups.every((g) => groupPx(g) <= (TEXT_X1 - TEXT_X0) * scale);
		// A single word wider than the row (a long text without spaces) falls back to plain wrapping.
		const boxes = fits ? layoutWordLines(groups, chipLayout) : layoutChips(toks, chipLayout);
		if (boxes.length > 0) {
			let top = -Infinity;
			let bottom = Infinity;
			for (const b of boxes) {
				top = Math.max(top, b.y + b.h / 2);
				bottom = Math.min(bottom, b.y - b.h / 2 - (showIds ? 20 / scale : 0));
			}
			const shift = (chart >= 1 ? TEXT_Y_CHART : TEXT_Y) - (top + bottom) / 2;
			chips0.set(
				boxes.map((b, i) => ({
					key: keys[i],
					text: b.text,
					x: b.x,
					y: b.y + shift,
					w: b.w,
					h: b.h,
					tone: tones[i],
					fill: i % 2 === 0 ? 0.16 : 0.26,
					sub: showIds ? subs[i] : undefined
				}))
			);
		} else chips0.set([]);
	}

	function retarget1(scale: number): void {
		const words = segment(TRAIN_TEXT, TRAIN_MODEL.merges, tmerges);
		const counts = countPairs(words);
		const best = counts.length > 0 && counts[0].count >= 2 ? counts[0].pair : null;
		const hot = new Set<number>();
		let n = 0;
		for (const w of words) {
			for (let i = 0; i < w.length; i++) {
				if (best && i + 1 < w.length && w[i] === best[0] && w[i + 1] === best[1]) {
					hot.add(n + i);
					hot.add(n + i + 1);
					i++;
				}
			}
			n += w.length;
		}
		const boxes = layoutWordLines(words, {
			x0: TEXT_X0,
			x1: TEXT_X1,
			y: 1.7,
			scale,
			fontPx: 13,
			padPx: 6,
			gapPx: 4,
			heightPx: 25,
			rowGapPx: 8
		});
		const toks = words.flat();
		const keys: number[] = [];
		let at = 0;
		for (const t of toks) {
			keys.push(at);
			at += [...t].length;
		}
		chips1.set(
			boxes.map((b, i) => {
				const t: Tone = hot.has(i) ? 'grad' : [...toks[i]].length > 1 ? 'param' : 'data';
				return {
					key: keys[i],
					text: b.text,
					x: b.x,
					y: b.y,
					w: b.w,
					h: b.h,
					tone: t,
					fill: t === 'grad' ? (i % 2 === 0 ? 0.3 : 0.4) : i % 2 === 0 ? 0.16 : 0.26
				};
			})
		);
		pairRows = counts
			.slice(0, 5)
			.filter((c) => c.count >= 2)
			.map((c) => ({ label: `${showToken(c.pair[0])} + ${showToken(c.pair[1])}`, count: c.count }));
		const made = TRAIN_MODEL.merges.slice(0, tmerges).map(([a, b]) => a + b);
		learned = layoutChips(made, {
			x0: 0.2,
			x1: 2.4,
			y: 0.3 - 40 / scale,
			scale,
			fontPx: 12,
			padPx: 5,
			gapPx: 4,
			heightPx: 22,
			rowGapPx: 6,
			align: 'left'
		});
		tokenCount1 = toks.length;
	}
	let tokenCount1 = 0;

	function computeChart(): void {
		chartCounts = [];
		for (let l = 0; l <= MAX_MERGES; l++) chartCounts.push(encode(text, MODEL, l).tokens.length);
		chartDirty = false;
	}

	function drawChips(q: Quads, field: ChipField, start: number, radius: number, a: number): number {
		let n = start;
		for (const c of field.chips) {
			q.set(n++, {
				x: c.cx,
				y: c.cy,
				w: c.cw,
				h: c.h,
				radius,
				color: tone(c.tone, c.fill, 0.9 * a * mul(c.tone))
			});
		}
		return n;
	}

	function chipLabels(out: Label[], field: ChipField, prefix: string, a: number, fontPx: number, scale: number): void {
		const v = ctx.view;
		for (const c of field.chips) {
			const [sx, sy] = v.project(c.cx, c.cy);
			const m = a * mul(c.tone);
			out.push({ id: `${prefix}${c.key}`, x: sx, y: sy, text: c.text, mono: true, size: fontPx, opacity: m });
			if (c.sub !== undefined) {
				const [bx, by] = v.project(c.cx, c.cy - c.h / 2 - 11 / scale);
				out.push({
					id: `${prefix}i${c.key}`,
					x: bx,
					y: by,
					text: c.sub,
					tone: c.sub === '?' ? 'neg' : 'muted',
					mono: true,
					size: 12,
					opacity: m
				});
			}
		}
	}

	function rebuild(scale: number): void {
		const v = ctx.view;
		const a0 = 1 - mix;
		const a1 = mix;
		const labels: Label[] = [];

		// view 0
		let n = 0;
		if (a0 > 0.003) {
			n = drawChips(q0, chips0, 0, 6, a0);
			chipLabels(labels, chips0, 'a', a0, CHIP_FONT, scale);
			if (chips0.chips.length === 0) {
				const [hx, hy] = v.project(0, chart >= 1 ? TEXT_Y_CHART : TEXT_Y);
				labels.push({ id: 'hint', x: hx, y: hy, text: 'Type something below', tone: 'muted', size: 15, opacity: a0 });
			}
		}
		q0.count = n;
		q0.upload();

		// chart
		let ln = 0;
		let pn = 0;
		const ac = a0 * chartMix;
		if (ac > 0.003 && chartCounts.length > 0 && chartCounts[0] > 0) {
			const w = CHART.x1 - CHART.x0;
			const h = CHART.y1 - CHART.y0;
			const top = chartCounts[0];
			const mm = mul('data');
			const axis = tone('muted', 1, 0.6 * ac);
			lines.set(ln++, { a: [CHART.x0, CHART.y0], b: [CHART.x0, CHART.y1], width: 1, color: axis });
			lines.set(ln++, { a: [CHART.x0, CHART.y0], b: [CHART.x1, CHART.y0], width: 1, color: axis });
			const pts: [number, number][] = chartCounts.map((c, l) => [
				CHART.x0 + (w * l) / MAX_MERGES,
				CHART.y0 + (h * c) / top
			]);
			ln = lines.setPolyline(ln, pts, { width: 2, color: tone('data', 0.8, 0.85 * ac * mm) });
			const m = clamp(Math.round(merges), 0, MAX_MERGES);
			const [mx, my] = pts[m];
			dot.set(pn++, { x: mx, y: my, radius: 11, softness: 1, color: tone('data', 0.8, 0.5 * ac * mm) });
			dot.set(pn++, { x: mx, y: my, radius: 4.5, color: tone('data', 1.4, ac * mm) });
			const lab = (id: string, x: number, y: number, t: string, al: 'left' | 'right' | 'center', tn: Tone, dx = 0, dy = 0) => {
				const [sx, sy] = v.project(x, y);
				labels.push({ id, x: sx + dx, y: sy + dy, text: t, tone: tn, mono: true, size: 11, align: al, opacity: ac * (tn === 'data' ? mm : 1) });
			};
			lab('c-x', 0, CHART.y0, 'merges learned', 'center', 'muted', 0, 28);
			lab('c-x0', CHART.x0, CHART.y0, '0', 'center', 'muted', 0, 13);
			lab('c-x1', CHART.x1, CHART.y0, String(MAX_MERGES), 'center', 'muted', 0, 13);
			lab('c-y', CHART.x0, CHART.y1, 'tokens', 'left', 'muted', 0, -24);
			lab('c-e0', pts[0][0], pts[0][1], String(chartCounts[0]), 'right', 'data', -9, 0);
			lab('c-e1', pts[MAX_MERGES][0], pts[MAX_MERGES][1], String(chartCounts[MAX_MERGES]), 'left', 'data', 9, 0);
			if (m > 12 && m < MAX_MERGES - 12) lab('c-now', mx, my, String(chartCounts[m]), 'center', 'data', 0, -16);
		}
		lines.count = ln;
		lines.upload();
		dot.count = pn;
		dot.upload();

		// view 1
		n = 0;
		if (a1 > 0.003) {
			n = drawChips(q1, chips1, 0, 5, a1);
			chipLabels(labels, chips1, 'b', a1, 13, scale);
			const capY = 0.3;
			const left = -2.4;
			const cap = (id: string, x: number, t: string, tn: Tone = 'muted', size = 11, dy = 0) => {
				const [sx, sy] = v.project(x, capY);
				labels.push({ id, x: sx, y: sy + dy, text: t, tone: tn, size, align: 'left', opacity: a1 });
			};
			cap('p-cap', left, 'Most common pairs');
			cap('l-cap', 0.2, 'Learned pieces');
			if (pairRows.length === 0) {
				cap('p-none', left, 'Nothing left to merge', 'muted', 12, 30);
			} else {
				const maxCount = pairRows[0].count;
				pairRows.forEach((r, i) => {
					const y = capY - (30 + i * 28) / scale;
					const first = i === 0;
					const tn: Tone = first ? 'grad' : 'muted';
					const bw = (112 * r.count) / maxCount;
					const m = first ? mul('grad') : mul('muted');
					q1.set(n++, {
						x: left + (122 + bw / 2) / scale,
						y,
						w: bw / scale,
						h: 12 / scale,
						radius: 3,
						color: first ? tone('grad', 0.6, 0.9 * a1 * m) : tone('muted', 1, 0.85 * a1 * m)
					});
					const [lx, ly] = v.project(left, y);
					labels.push({ id: `p-l${i}`, x: lx, y: ly, text: r.label, tone: first ? 'grad' : 'text', mono: true, size: 12, align: 'left', opacity: a1 * m });
					const [cx2, cy2] = v.project(left + (122 + bw + 8) / scale, y);
					labels.push({ id: `p-c${i}`, x: cx2, y: cy2, text: String(r.count), tone: tn, mono: true, size: 12, align: 'left', opacity: a1 * m });
				});
			}
			if (learned.length === 0) {
				cap('l-none', 0.2, 'None yet', 'muted', 12, 30);
			} else {
				const mp = mul('param');
				learned.forEach((b, i) => {
					const newest = i === learned.length - 1;
					q1.set(n++, {
						x: b.x,
						y: b.y,
						w: b.w,
						h: b.h,
						radius: 5,
						color: tone('param', newest ? 0.5 : 0.2, 0.9 * a1 * mp)
					});
					const [sx, sy] = v.project(b.x, b.y);
					labels.push({ id: `l${i}`, x: sx, y: sy, text: b.text, mono: true, size: 12, opacity: a1 * mp });
				});
			}
		}
		q1.count = n;
		q1.upload();
		ctx.labels.set(labels);
	}

	function emitReadouts(): void {
		const t = view >= 0.5 ? tokenCount1 : tokenCount0;
		const vo = view >= 0.5 ? TRAIN_BASE_CHARS + tmerges : vocab0;
		if (t !== lastTokens) {
			lastTokens = t;
			ctx.emit('tokens', t);
		}
		if (vo !== lastVocab) {
			lastVocab = vo;
			ctx.emit('vocab', vo);
		}
	}

	function stepMerge(): boolean {
		if (tmerges >= TRAIN_MAX) return false;
		tmerges++;
		ctx.emit('param:tmerges', tmerges);
		dirtyLayout = true;
		return true;
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			q0 = new Quads(c, { capacity: 64, blend: 'alpha' });
			q1 = new Quads(c, { capacity: 128, blend: 'alpha' });
			lines = new Lines(c, { capacity: 340, blend: 'alpha' });
			dot = new Points(c, { capacity: 4, blend: 'alpha' });
		},
		setParams(p) {
			const prev = { view, mode, m: Math.round(merges), tmerges, showIds, chart };
			view = Math.round(p.view ?? view);
			mode = Math.round(p.mode ?? mode);
			merges = p.merges ?? merges;
			tmerges = clamp(Math.round(p.tmerges ?? tmerges), 0, TRAIN_MAX);
			showIds = (p.ids ?? (showIds ? 1 : 0)) >= 0.5;
			chart = Math.round(p.chart ?? chart);
			const sc = Math.round(p.scenario ?? scenario);
			if (sc !== scenario || view !== prev.view) running = false;
			scenario = sc;
			if (!seen) {
				// First params: start on the right view instead of fading in from the other one.
				seen = true;
				mix = view >= 1 ? 1 : 0;
				chartMix = chart >= 1 ? 1 : 0;
			}
			if (
				prev.view !== view ||
				prev.mode !== mode ||
				prev.m !== Math.round(merges) ||
				prev.tmerges !== tmerges ||
				prev.showIds !== showIds
			)
				dirtyLayout = true;
			if (prev.chart !== chart) dirtyLayout = true;
		},
		update(fr) {
			const v = ctx.view;
			fitCamera(camera, v, WORLD, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));
			const scale = scaleNow();
			if (scale !== lastScale || v.width !== lastW || v.height !== lastH) {
				lastScale = scale;
				lastW = v.width;
				lastH = v.height;
				dirtyLayout = true;
			}

			if (running && fr.dt > 0) {
				runTimer += fr.dt;
				if (runTimer >= RUN_PERIOD) {
					runTimer = 0;
					if (!stepMerge()) running = false;
					else if (tmerges >= TRAIN_MAX) running = false;
				}
			}

			const instant = ctx.reducedMotion;
			let animating = false;
			const fadeTo = (cur: number, target: number) => {
				if (instant || cur === target) return target;
				const nx = cur + Math.sign(target - cur) * (fr.dt / FADE_TIME);
				animating = true;
				return Math.abs(target - nx) < 1e-3 || (nx - target) * (cur - target) < 0 ? target : nx;
			};
			const prevMix = mix;
			mix = fadeTo(mix, view >= 1 ? 1 : 0);
			chartMix = fadeTo(chartMix, chart >= 1 ? 1 : 0);
			if (mix !== prevMix || chartMix !== (chart >= 1 ? 1 : 0)) animating = true;

			if (dirtyLayout) {
				retarget0(scale);
				retarget1(scale);
				dirtyLayout = false;
				dirty = true;
			}
			if (chart >= 1 || chartMix > 0) {
				if (chartDirty) {
					computeChart();
					dirty = true;
				}
			}
			if (chips0.step(fr.dt, instant)) animating = true;
			if (chips1.step(fr.dt, instant)) animating = true;

			if (dirty || animating) rebuild(scale);
			dirty = false;
			emitReadouts();
		},
		draw(pass) {
			q0.draw(pass);
			lines.draw(pass);
			dot.draw(pass);
			q1.draw(pass);
		},
		command(id) {
			if (id.startsWith('text:')) {
				text = id.slice(5);
				chartDirty = true;
				dirtyLayout = true;
			} else if (id === 'merge') {
				running = false;
				stepMerge();
			} else if (id === 'run') {
				if (running) running = false;
				else if (tmerges < TRAIN_MAX) {
					running = true;
					runTimer = 0;
				}
			} else if (id === 'reset') {
				running = false;
				if (tmerges !== 0) {
					tmerges = 0;
					ctx.emit('param:tmerges', 0);
					dirtyLayout = true;
				}
			}
		},
		highlight(t) {
			hl = t;
			dirty = true;
		},
		dispose() {
			q0.destroy();
			q1.destroy();
			lines.destroy();
			dot.destroy();
		}
	};
	return scene;
};

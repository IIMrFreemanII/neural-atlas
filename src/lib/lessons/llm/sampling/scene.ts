import { Camera2D } from '../../../gpu/camera.ts';
import { tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import { Quads } from '../../../gpu/primitives/quads.ts';
import type { Label, Scene, SceneContext, SceneFactory, Tone } from '../../../gpu/types.ts';
import { mulberry32 } from '../../../ml/rng.ts';
import { distribution, sampleIndex } from '../../../ml/sampling.ts';
import { softmax } from '../../../ml/vec.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { layoutChips, pxPerWorld, textWidthPx } from '../shared/chips.ts';
import { END } from '../../../ml/wordtok.ts';
import { loadLm, type Lm } from '../shared/lm.ts';
import { nextScores, promptIds, tokenText } from './data.ts';

const WORLD: WorldRect = { x0: -2.6, x1: 2.6, y0: -2.1, y1: 2.1 };

// text
const TEXT_X0 = -2.45;
const TEXT_X1 = 2.45;
const TEXT_Y = 1.9;
const TEXT_ROWS = 3;
const CHIP_FONT = 13;

// bars
const SHOWN = 12;
const ROWS = SHOWN + 1;
const BAR_Y0 = 1.0;
const BAR_Y1 = -1.05;
const BAR_H = 0.12;
const TOKEN_X = -1.25;
const BAR_X = -1.15;
const BAR_LEN = 3.05;

// strip
const STRIP_X0 = -2.3;
const STRIP_LEN = 4.6;
const STRIP_Y = -1.52;
const STRIP_H = 0.3;

// timing, seconds
const NEEDLE_TIME = 0.45;
const NEEDLE_TIME_NO_STRIP = 0.1;
const TRAVEL_START = 0.05;
const TRAVEL_TIME = 0.35;
const EASE_TIME = 0.12;
const RUN_LIMIT = 24;
const RUN_GAP = 0.15;
const RUN_GAP_REDUCED = 0.4;

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const easeOut = (t: number) => 1 - Math.pow(1 - clamp(t, 0, 1), 3);
const easeInOut = (t: number) => {
	const x = clamp(t, 0, 1);
	return x * x * (3 - 2 * x);
};
const rowY = (r: number) => BAR_Y0 + (r * (BAR_Y1 - BAR_Y0)) / (ROWS - 1);

function percent(p: number): string {
	if (p <= 0) return '';
	if (p >= 0.995) return '100%';
	if (p < 0.005) return '<1%';
	return `${Math.max(1, Math.round(p * 100))}%`;
}

interface Pick {
	u: number;
	row: number;
	id: number;
	t: number;
	needle: number;
}

export const createSamplingScene: SceneFactory = () => {
	let ctx: SceneContext;
	let lm: Lm;
	let camera: Camera2D;
	let quads: Quads;
	let lines: Lines;
	let dots: Points;

	// params
	let temperature = 1;
	let topK = 0;
	let topP = 1;
	let prompt = 0;
	let stripOn = true;
	let scenario = 1;
	let paramsSeen = false;
	let hl: Tone | null = null;

	// state
	let tokens: number[] = [];
	let promptLen = 0;
	let generated = 0;
	let seedCount = 0;
	let rand = mulberry32(1);
	let running = false;
	let wait = 0;
	let pick: Pick | null = null;
	let needleFade = 0;
	let needleX = STRIP_X0;

	// distribution for the current context
	let logits: number[] = [];
	let order: number[] = []; // vocabulary ids, most likely first
	let tBase = new Array<number>(ROWS).fill(0);
	let tPre = new Array<number>(ROWS).fill(0);
	let tFinal = new Array<number>(ROWS).fill(0);
	let tCut = new Array<number>(ROWS).fill(0);
	let probs: number[] = [];
	let rowNames: string[] = new Array<string>(ROWS).fill('');
	// displayed (eased) values
	const dBase = new Array<number>(ROWS).fill(0);
	const dPre = new Array<number>(ROWS).fill(0);
	const dFinal = new Array<number>(ROWS).fill(0);
	const dCut = new Array<number>(ROWS).fill(0);
	let snap = true;

	let dirty = true;
	let lastW = 0;
	let lastH = 0;
	let lastKept = -1;
	let lastTop = '';
	let kept = 0;
	let topText = '';

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);

	function recomputeLogits() {
		logits = nextScores(lm, tokens);
		order = Array.from({ length: logits.length }, (_, i) => i);
		order.sort((a, b) => logits[b] - logits[a] || a - b);
		refreshTargets();
	}

	function refreshTargets() {
		const V = logits.length;
		const base = softmax(logits, 1);
		const pre = distribution(logits, { temperature, topK: 0, topP: 1 });
		probs = distribution(logits, { temperature, topK, topP });
		const sum = (src: number[], r: number) => {
			let s = 0;
			if (r < SHOWN) return src[order[r]];
			for (let i = SHOWN; i < V; i++) s += src[order[i]];
			return s;
		};
		kept = 0;
		let best = 0;
		for (let i = 0; i < V; i++) {
			if (probs[i] > 0) kept++;
			if (probs[i] > probs[best]) best = i;
		}
		for (let r = 0; r < ROWS; r++) {
			tBase[r] = sum(base, r);
			tPre[r] = sum(pre, r);
			tFinal[r] = sum(probs, r);
			tCut[r] = tFinal[r] <= 0 && tPre[r] > 0 ? 1 : 0;
			rowNames[r] = r < SHOWN ? tokenText(lm, order[r]) : `all ${Math.max(0, V - SHOWN)} others`;
		}
		topText = `${tokenText(lm, best)} ${percent(probs[best]).replace('<1%', '1%')}`;
		dirty = true;
	}

	function reset(fromAction: boolean) {
		if (fromAction) seedCount++;
		else seedCount = 0;
		rand = mulberry32(40 + scenario * 17 + seedCount);
		running = false;
		wait = 0;
		pick = null;
		needleFade = 0;
		generated = 0;
		tokens = promptIds(lm, prompt);
		promptLen = tokens.length;
		recomputeLogits();
	}

	/** Which row of the display the value u falls in, and which vocabulary token that means. */
	function choose(u: number): { row: number; id: number } {
		let cum = 0;
		for (let r = 0; r < SHOWN; r++) {
			cum += tFinal[r];
			if (cum > u) return { row: r, id: order[r] };
		}
		if (tFinal[ROWS - 1] > 0) {
			const rest = order.slice(SHOWN).map((i) => probs[i]);
			const j = sampleIndex(rest, u - cum);
			if (rest[j] > 0) return { row: ROWS - 1, id: order[SHOWN + j] };
		}
		for (let r = ROWS - 1; r >= 0; r--) {
			if (tFinal[r] > 0) {
				if (r < SHOWN) return { row: r, id: order[r] };
				const rest = order.slice(SHOWN).map((i) => probs[i]);
				return { row: r, id: order[SHOWN + sampleIndex(rest, 0)] };
			}
		}
		return { row: 0, id: order[0] };
	}

	function append(id: number) {
		tokens = [...tokens, id];
		generated++;
		if (id === END) running = false;
		recomputeLogits();
	}

	function startPick() {
		const u = rand();
		const c = choose(u);
		if (ctx.reducedMotion) {
			append(c.id);
			wait = RUN_GAP_REDUCED;
			if (generated >= RUN_LIMIT) running = false;
			return;
		}
		pick = { u, row: c.row, id: c.id, t: 0, needle: stripOn ? NEEDLE_TIME : NEEDLE_TIME_NO_STRIP };
		needleFade = 1;
		dirty = true;
	}

	function finishPick() {
		if (!pick) return;
		append(pick.id);
		pick = null;
		wait = RUN_GAP;
		if (generated >= RUN_LIMIT) running = false;
	}

	/** Chips for the text so far (plus an optional extra token), oldest tokens dropped to fit in 3 rows. */
	function textLayout(ppw: number, extra: number | null) {
		const ids = extra === null ? tokens : [...tokens, extra];
		let start = 0;
		for (;;) {
			const texts = (start > 0 ? ['…'] : []).concat(ids.slice(start).map((i) => lm.vocab.words[i]));
			const boxes = layoutChips(texts, {
				x0: TEXT_X0,
				x1: TEXT_X1,
				y: TEXT_Y - 0.1,
				scale: ppw,
				fontPx: CHIP_FONT,
				align: 'left',
				minChars: 2
			});
			const rows = boxes.length > 0 ? boxes[boxes.length - 1].row + 1 : 0;
			if (rows <= TEXT_ROWS || start >= ids.length - 1) {
				return boxes.map((box) => ({ box, token: start > 0 ? (box.index === 0 ? -1 : start + box.index - 1) : box.index, total: ids.length }));
			}
			start++;
		}
	}

	function ease(dt: number): boolean {
		const k = ctx.reducedMotion || snap ? 1 : 1 - Math.exp(-dt / EASE_TIME);
		let moving = false;
		const step = (d: number[], t: number[]) => {
			for (let r = 0; r < ROWS; r++) {
				const diff = t[r] - d[r];
				if (Math.abs(diff) > 1e-4) {
					d[r] += diff * k;
					moving = true;
				} else d[r] = t[r];
			}
		};
		step(dBase, tBase);
		step(dPre, tPre);
		step(dFinal, tFinal);
		step(dCut, tCut);
		snap = false;
		return moving;
	}

	function draw2d(): void {
		const v = ctx.view;
		const ppw = pxPerWorld(v, camera);
		const labels: Label[] = [];
		let qn = 0;
		let ln = 0;
		let pn = 0;

		// ---- text so far
		const mText = mul('text');
		const chips = textLayout(ppw, null);
		for (const { box, token, total } of chips) {
			const isPrompt = token >= 0 && token < promptLen;
			const newest = token === total - 1 && generated > 0 && !(pick && pick.t > 0);
			const color =
				token < 0
					? tone('muted', 0.2, 0.9 * mText)
					: isPrompt
						? tone('text', 0.16, 0.9 * mText)
						: tone('data', newest ? 0.4 : 0.24, 0.9 * mText);
			quads.set(qn++, { x: box.x, y: box.y, w: box.w, h: box.h, color, radius: 6 });
			const [px, py] = v.project(box.x, box.y);
			labels.push({
				id: `chip${box.index}`,
				x: px,
				y: py,
				text: box.text,
				mono: true,
				size: CHIP_FONT,
				tone: token < 0 ? 'muted' : 'text',
				opacity: mText
			});
		}

		// ---- bars
		const mData = mul('data');
		const mNeg = mul('neg');
		const mOk = mul('ok');
		const showTicks = !(Math.abs(temperature - 1) < 0.01 && topK <= 0 && topP >= 1);
		const flashRow = pick && pick.t >= pick.needle ? pick.row : -1;
		const flash = pick ? clamp((pick.t - pick.needle) / 0.1, 0, 1) : 0;
		for (let r = 0; r < ROWS; r++) {
			const y = rowY(r);
			const cut = dCut[r];
			const isCutTarget = tCut[r] > 0.5;
			const dim = isCutTarget ? 0.45 : 1;
			// row highlight for the picked token
			if (r === flashRow) {
				quads.set(qn++, {
					x: (BAR_X - 0.1 + 2.4) / 2,
					y,
					w: 2.4 - (BAR_X - 0.1),
					h: 0.17,
					color: tone('ok', 0.14, 0.9 * flash * mOk),
					radius: 5
				});
			}
			if (cut > 0.01 && dPre[r] * BAR_LEN > 0.004) {
				const w = dPre[r] * BAR_LEN;
				quads.set(qn++, { x: BAR_X + w / 2, y, w, h: BAR_H, color: tone('neg', 0.42, 0.9 * cut * mNeg), radius: 3 });
			}
			const wf = dFinal[r] * BAR_LEN;
			if (wf > 0.004) {
				const picked = r === flashRow;
				const color = picked
					? tone('ok', 0.45 + 0.15 * flash, 0.95 * mOk)
					: tone('data', 0.62, 0.95 * mData);
				quads.set(qn++, { x: BAR_X + wf / 2, y, w: wf, h: BAR_H, color, radius: 3 });
			}
			// token name
			const [lx, ly] = v.project(TOKEN_X, y);
			labels.push({
				id: `name${r}`,
				x: lx,
				y: ly,
				text: rowNames[r],
				mono: true,
				size: 13,
				align: 'right',
				tone: isCutTarget ? 'muted' : 'text',
				opacity: dim * (isCutTarget ? mNeg : mData)
			});
			// percentage or "cut"
			const endLen = isCutTarget ? dPre[r] : dFinal[r];
			const txt = isCutTarget ? 'cut' : percent(tFinal[r]);
			if (txt) {
				const tickLen = showTicks ? dBase[r] : 0;
				const [ex, ey] = v.project(BAR_X + Math.max(0.02, endLen, tickLen) * BAR_LEN + 0.08, y);
				labels.push({
					id: `pct${r}`,
					x: ex,
					y: ey,
					text: txt,
					mono: true,
					size: 11,
					align: 'left',
					tone: isCutTarget ? 'neg' : 'text',
					opacity: isCutTarget ? 0.8 * mNeg : mData
				});
			}
			// tick at the original probability
			if (showTicks && dBase[r] > 0.001) {
				const tx = BAR_X + dBase[r] * BAR_LEN;
				lines.set(ln++, {
					a: [tx, y - 0.085],
					b: [tx, y + 0.085],
					width: 1.5,
					color: tone('text', 0.6, 0.9 * Math.min(mData, 0.9))
				});
			}
		}

		// ---- strip
		if (stripOn) {
			let cum = 0;
			for (let r = 0; r < ROWS; r++) {
				const w = dFinal[r] * STRIP_LEN;
				const x0 = STRIP_X0 + cum * STRIP_LEN;
				cum += dFinal[r];
				if (w < 0.006) continue;
				const gap = Math.min(0.012, w * 0.3);
				const color =
					r === ROWS - 1
						? tone('muted', 0.9, 0.9 * mData)
						: tone('data', r % 2 === 0 ? 0.26 : 0.42, 0.95 * mData);
				quads.set(qn++, { x: x0 + w / 2, y: STRIP_Y, w: w - gap, h: STRIP_H, color, radius: 3 });
				if (r === flashRow) {
					quads.set(qn++, {
						x: x0 + w / 2,
						y: STRIP_Y,
						w: w - gap,
						h: STRIP_H,
						color: tone('ok', 0.6, 0.95 * flash * mOk),
						radius: 3
					});
				}
				if (w * ppw > 46) {
					const [sx, sy] = v.project(x0 + w / 2, STRIP_Y);
					labels.push({
						id: `slice${r}`,
						x: sx,
						y: sy,
						text: r === ROWS - 1 ? 'others' : rowNames[r],
						mono: true,
						size: 11,
						tone: 'text',
						opacity: mData
					});
				}
			}
			const [cx, cy] = v.project(0, STRIP_Y - 0.33);
			labels.push({
				id: 'strip-caption',
				x: cx,
				y: cy,
				text: 'one slice per token, as wide as its probability',
				tone: 'muted',
				size: 11,
				opacity: mData
			});
			// needle
			if (pick || needleFade > 0.01) {
				const a = pick ? 1 : needleFade;
				const nx = needleX;
				lines.set(ln++, {
					a: [nx, STRIP_Y - 0.24],
					b: [nx, STRIP_Y + 0.24],
					width: 2,
					color: tone('text', 0.7, 0.9 * a)
				});
				dots.set(pn++, { x: nx, y: STRIP_Y + 0.24, radius: 4.5, color: tone('text', 1.1, 0.95 * a) });
			}
		}

		// ---- the picked token travels from its bar to the end of the text
		if (pick) {
			const tt = (pick.t - pick.needle - TRAVEL_START) / TRAVEL_TIME;
			if (tt > 0) {
				const target = textLayout(ppw, pick.id);
				const end = target[target.length - 1].box;
				const k = easeInOut(tt);
				const text = tokenText(lm, pick.id);
				const w = Math.max(textWidthPx(text, CHIP_FONT), textWidthPx('..', CHIP_FONT)) + 12;
				const ww = w / ppw;
				const sx = TOKEN_X - ww / 2 - 0.05;
				const sy = rowY(pick.row);
				const x = sx + (end.x - sx) * k;
				const y = sy + (end.y - sy) * k;
				quads.set(qn++, { x, y, w: ww, h: end.h, color: tone('ok', 0.4, 0.95), radius: 6 });
				const [px, py] = v.project(x, y);
				labels.push({ id: 'travel', x: px, y: py, text, mono: true, size: CHIP_FONT, tone: 'text' });
			}
		}

		quads.count = qn;
		quads.upload();
		lines.count = ln;
		lines.upload();
		dots.count = pn;
		dots.upload();
		ctx.labels.set(labels);
	}

	const scene: Scene = {
		async init(c) {
			ctx = c;
			lm = await loadLm();
			camera = new Camera2D();
			quads = new Quads(c, { capacity: 220, blend: 'alpha' });
			lines = new Lines(c, { capacity: 24, blend: 'alpha' });
			dots = new Points(c, { capacity: 4, blend: 'alpha' });
		},
		setParams(p) {
			const prev = { prompt, scenario, temperature, topK, topP };
			temperature = clamp(p.temperature ?? temperature, 0, 2);
			topK = Math.round(p.topK ?? topK);
			topP = clamp(p.topP ?? topP, 0.05, 1);
			prompt = clamp(Math.round(p.prompt ?? prompt), 0, 2);
			scenario = Math.round(p.scenario ?? scenario);
			stripOn = (p.strip ?? 1) >= 0.5;
			if (!paramsSeen || prev.prompt !== prompt || prev.scenario !== scenario) {
				reset(false);
			} else if (prev.temperature !== temperature || prev.topK !== topK || prev.topP !== topP) {
				refreshTargets();
			}
			paramsSeen = true;
			dirty = true;
		},
		update(fr) {
			const v = ctx.view;
			fitCamera(camera, v, WORLD, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));
			if (v.width !== lastW || v.height !== lastH) {
				lastW = v.width;
				lastH = v.height;
				dirty = true;
			}

			// picking and running
			if (pick) {
				pick.t += fr.dt;
				const k = pick.t < pick.needle ? easeOut(pick.t / pick.needle) : 1;
				needleX = STRIP_X0 + pick.u * STRIP_LEN * k;
				dirty = true;
				if (pick.t >= pick.needle + TRAVEL_START + TRAVEL_TIME) finishPick();
			} else {
				if (needleFade > 0) {
					needleFade = Math.max(0, needleFade - fr.dt / 0.3);
					dirty = true;
				}
				if (running && fr.dt > 0) {
					wait -= fr.dt;
					if (wait <= 0) {
						if (generated >= RUN_LIMIT) running = false;
						else startPick();
					}
				}
			}

			if (ease(fr.dt)) dirty = true;
			if (dirty) {
				dirty = false;
				draw2d();
			}

			if (kept !== lastKept) {
				lastKept = kept;
				ctx.emit('kept', kept);
			}
			if (topText !== lastTop) {
				lastTop = topText;
				ctx.emit('top', topText);
			}
		},
		draw(pass) {
			quads.draw(pass);
			lines.draw(pass);
			dots.draw(pass);
		},
		command(id) {
			if (id === 'sample') {
				running = false;
				if (!pick) startPick();
			} else if (id === 'run') {
				if (running) {
					running = false;
				} else {
					if (generated >= RUN_LIMIT) reset(true);
					running = true;
					wait = 0;
				}
			} else if (id === 'reset') {
				reset(true);
			}
			dirty = true;
		},
		highlight(t) {
			hl = t;
			dirty = true;
		},
		dispose() {
			quads.destroy();
			lines.destroy();
			dots.destroy();
		}
	};
	return scene;
};

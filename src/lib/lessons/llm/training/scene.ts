import { Camera2D } from '../../../gpu/camera.ts';
import { tone } from '../../../gpu/colors.ts';
import { Lines } from '../../../gpu/primitives/lines.ts';
import { Points } from '../../../gpu/primitives/points.ts';
import { Quads } from '../../../gpu/primitives/quads.ts';
import type { Label, Scene, SceneContext, SceneFactory, ScenePointer, Tone } from '../../../gpu/types.ts';
import { fitCamera, lessonInsets, type WorldRect } from '../../fit.ts';
import { layoutChips, pxPerWorld } from '../shared/chips.ts';
import {
	LAST,
	LOG,
	axisX,
	choices,
	curveLossAt,
	formatStep,
	percent,
	stepOfPosition,
	tokenLabel
} from './data.ts';

// One world unit is about 100 CSS px at 1280x800 (the stage is about 680 px wide there, with the sidebar open).
const WORLD: WorldRect = { x0: -3.1, x1: 3.1, y0: -2.52, y1: 2.52 };

// chart band: the plot area, loss from 0 to LOSS_MAX
const PX0 = -2.85;
const PX1 = 2.95;
const PY0 = 0.78; // loss 0
const PY1 = 2.15; // loss LOSS_MAX
const LOSS_MAX = 8.5;
const DASH = 0.09;
const DASH_GAP = 0.06;

// story band
const STORY_X0 = -3.1;
const STORY_X1 = 3.1;
const STORY_CAPTION_Y = 0.2;
const STORY_Y = -0.04;
const STORY_ROWS = 3;
const STORY_FONT = 12;

// probe band
const PROBE_X0 = -3.1;
const PROBE_X1 = 3.1;
const PROBE_CAPTION_Y = -0.95;
const PROBE_Y = -1.2;
const PROBE_FONT = 11;
const TRACK_TOP = -1.38;
const TRACK_H = 0.46;
const PCT_Y = -1.95;
const DETAIL_Y = -2.24;

// timing, seconds
const BAND_TIME = 0.25;
const MOVE_TIME = 0.3;
const CHECKPOINT_TIME = 1.1;
const CHECKPOINT_TIME_REDUCED = 0.6;
const FADE_TIME = 0.12;
const BAR_TAU = 0.07;

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const easeInOut = (t: number) => {
	const x = clamp(t, 0, 1);
	return x * x * (3 - 2 * x);
};
const toward = (v: number, target: number, step: number) =>
	v < target ? Math.min(target, v + step) : Math.max(target, v - step);

const cx = (step: number) => PX0 + axisX(step) * (PX1 - PX0);
const cy = (loss: number) => PY0 + (clamp(loss, 0, LOSS_MAX) / LOSS_MAX) * (PY1 - PY0);

type Pt = [number, number];

interface Hit {
	x0: number;
	x1: number;
	y0: number;
	y1: number;
}

export const createTrainingScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let quads: Quads;
	let frame: Lines;
	let under: Lines;
	let over: Lines;
	let dots: Points;

	// params
	let prompt = 0;
	let dice = 0;
	let chartOn = 1;
	let storyOn = 1;
	let probeOn = 1;
	let validOn = 0;
	let hl: Tone | null = null;
	let paramsSeen = false;

	// position along the checkpoints, 0..LAST
	let pos = 0;
	let lastAt = 0;
	let moveFrom = 0;
	let moveTo = 0;
	let moveT = 1;
	let running = false;
	let runWait = 0;

	// shown state
	let chartMix = 0;
	let storyMix = 0;
	let probeMix = 0;
	let validMix = 0;
	let shownCk = 0;
	let shownPrompt = 0;
	let shownDice = 0;
	let textAlpha = 1;
	let promptAlpha = 1;
	const bars = new Array<number>(LOG.probe.length - 1).fill(0);
	let barsSnap = true;
	let hover = -1;
	let hits: Hit[] = [];
	const defaultSel = Math.max(1, LOG.probe.indexOf('park'));

	let dirty = true;
	let lastW = 0;
	let lastH = 0;
	let lastStep = '';
	let lastLoss = Number.NaN;
	let lastChoices = '';

	const mul = (t: Tone) => (hl === null || hl === t ? 1 : 0.3);
	const currentCk = () => clamp(Math.floor(pos + 1e-6), 0, LAST);
	const selected = () => (hover >= 1 ? hover : defaultSel);

	function arrive(k: number) {
		lastAt = k;
		ctx.emit('param:at', k);
	}

	function moveTowards(k: number) {
		if (ctx.reducedMotion || !paramsSeen) {
			pos = k;
			moveT = 1;
		} else {
			moveFrom = pos;
			moveTo = k;
			moveT = 0;
		}
		dirty = true;
	}

	function promptTokens(p: number): string[] {
		return (LOG.prompts[p] ?? '').split(' ').filter((t) => t !== '');
	}

	function writtenTokens(ck: number, p: number, dc: number): string[] {
		const c = LOG.checkpoints[ck];
		return (dc >= 0.5 ? c.sampled[p] : c.greedy[p]) ?? [];
	}

	// ---- chart

	function drawChart(labels: Label[], loss: number, step: number, qn: number): number {
		const v = ctx.view;
		const a = chartMix;
		let k = 0;
		let c = 0;
		let o = 0;
		let pn = 0;
		if (a > 0.005) {
			const mM = mul('muted');
			const mG = mul('grad');
			const mT = mul('text');

			// axes, grid, ticks
			const axis = tone('muted', 1, 0.5 * a);
			frame.set(k++, { a: [PX0, PY0], b: [PX0, PY1], width: 1, color: axis });
			frame.set(k++, { a: [PX0, PY0], b: [PX1, PY0], width: 1, color: axis });
			for (const y of [0, 2, 4, 6, 8]) {
				if (y > 0) frame.set(k++, { a: [PX0, cy(y)], b: [PX1, cy(y)], width: 1, color: tone('muted', 1, 0.14 * a) });
				frame.set(k++, { a: [PX0 - 0.05, cy(y)], b: [PX0, cy(y)], width: 1, color: axis });
				const [lx, ly] = v.project(PX0 - 0.1, cy(y));
				labels.push({ id: `cy${y}`, x: lx, y: ly, text: String(y), tone: 'muted', mono: true, size: 10, align: 'right', opacity: a });
			}
			const [capX, capY] = v.project(PX0, PY1 + 0.15);
			labels.push({ id: 'cap-loss', x: capX, y: capY, text: 'loss', tone: 'muted', size: 11, opacity: a });
			LOG.checkpoints.forEach((ck, i) => {
				const x = cx(ck.step);
				frame.set(k++, { a: [x, PY0], b: [x, PY0 - 0.05], width: 1, color: axis });
				const [lx, ly] = v.project(x, PY0 - 0.05);
				labels.push({ id: `cx${i}`, x: lx, y: ly + 11, text: formatStep(ck.step), tone: 'muted', mono: true, size: 10, opacity: a });
			});
			const [sx, sy] = v.project((PX0 + PX1) / 2, PY0 - 0.05);
			labels.push({ id: 'cap-steps', x: sx, y: sy + 30, text: 'training steps', tone: 'muted', size: 11, opacity: a });

			// the loss of guessing blindly
			const uy = cy(LOG.uniformLoss);
			const dashColor = tone('muted', 1, 0.6 * a);
			for (let x = PX0; x < PX1 - 1e-6; x += DASH + DASH_GAP) {
				frame.set(k++, { a: [x, uy], b: [Math.min(PX1, x + DASH), uy], width: 1, color: dashColor });
			}
			const [ux, uyPx] = v.project(PX1, uy);
			labels.push({
				id: 'uniform',
				x: ux,
				y: uyPx - 9,
				text: `guessing blindly ${LOG.uniformLoss.toFixed(2)}`,
				tone: 'muted',
				mono: true,
				size: 11,
				align: 'right',
				opacity: a * mM
			});

			// the curve, up to the current step
			const pts: Pt[] = [[cx(0), cy(LOG.checkpoints[0].trainLoss)]];
			for (let i = 0; i < LOG.curveStep.length; i++) {
				if (LOG.curveStep[i] >= step) break;
				pts.push([cx(LOG.curveStep[i]), cy(LOG.curveLoss[i])]);
			}
			const mx = cx(step);
			const my = cy(loss);
			if (step > 0) pts.push([mx, my]);
			if (pts.length > 1) {
				c = under.setPolyline(0, pts, { width: 9, color: tone('grad', 1, 0.14 * a * mG) });
				o = over.setPolyline(0, pts, { width: 2.5, color: tone('grad', 1.5, 0.9 * a * mG) });
			}

			// marker, guide and label
			frame.set(k++, { a: [mx, PY0], b: [mx, my], width: 1, color: tone('grad', 1, 0.28 * a * mG) });
			dots.set(pn++, { x: mx, y: my, radius: 11, softness: 1, color: tone('grad', 1.2, 0.5 * a * mG) });
			dots.set(pn++, { x: mx, y: my, radius: 4.5, color: tone('grad', 1.8, 0.98 * a * mG) });
			const [mpx, mpy] = v.project(mx, my);
			const [rightPx] = v.project(PX1, 0);
			const [, topPx] = v.project(0, PY1);
			const nearTop = mpy - 16 < topPx - 4;
			const flip = mpx + 52 > rightPx;
			labels.push({
				id: 'marker',
				x: flip ? mpx - 12 : mpx + 12,
				y: nearTop ? mpy : mpy - 15,
				text: loss.toFixed(2),
				tone: 'grad',
				mono: true,
				size: 11,
				align: flip ? 'right' : 'left',
				opacity: a * mG
			});

			// loss on stories the model never saw
			const vm = validMix * a;
			if (vm > 0.005) {
				const cur = currentCk();
				LOG.checkpoints.forEach((ck, i) => {
					if (ck.step > step + 1e-6) return;
					const isCur = i === cur;
					dots.set(pn++, {
						x: cx(ck.step),
						y: cy(ck.validLoss),
						radius: isCur ? 4 : 3.5,
						color: tone('text', 1.1, 0.95 * vm * mT)
					});
				});
				const ck = LOG.checkpoints[cur];
				const [dx, dy] = v.project(cx(ck.step), cy(ck.validLoss));
				const [axisPx] = v.project(PX0, 0);
				const textPx = 125;
				const toLeft = dx - 8 - textPx > axisPx + 4;
				labels.push({
					id: 'valid',
					x: toLeft ? dx - 8 : dx + 8,
					y: dy + 15,
					text: `unseen stories ${ck.validLoss.toFixed(2)}`,
					tone: 'text',
					mono: true,
					size: 11,
					align: toLeft ? 'right' : 'left',
					opacity: vm * mT
				});
			}
		}
		frame.count = k;
		under.count = c;
		over.count = o;
		dots.count = pn;
		return qn;
	}

	// ---- story

	function drawStory(labels: Label[], qn: number, ppw: number): number {
		const a = storyMix;
		if (a <= 0.005) return qn;
		const v = ctx.view;
		const mText = mul('text');
		const mData = mul('data');
		const [cx0, cy0] = v.project(STORY_X0, STORY_CAPTION_Y);
		labels.push({
			id: 'cap-story',
			x: cx0,
			y: cy0,
			text: 'the model continues the story',
			tone: 'muted',
			size: 11,
			align: 'left',
			opacity: a
		});
		const given = promptTokens(shownPrompt);
		const written = writtenTokens(shownCk, shownPrompt, shownDice);
		const all = [...given.map(tokenLabel), ...written.map(tokenLabel)];
		const boxes = layoutChips(all, {
			x0: STORY_X0,
			x1: STORY_X1,
			y: STORY_Y,
			scale: ppw,
			fontPx: STORY_FONT,
			padPx: 5,
			gapPx: 3,
			rowGapPx: 6,
			align: 'left',
			minChars: 2
		});
		for (const box of boxes) {
			if (box.row >= STORY_ROWS) break;
			const isGiven = box.index < given.length;
			const isEnd = !isGiven && written[box.index - given.length] === '<end>';
			const fade = a * (isGiven ? promptAlpha * mText : textAlpha * mData);
			const color = isEnd
				? tone('muted', 0.35, 0.9 * fade)
				: isGiven
					? tone('text', 0.16, 0.9 * fade)
					: tone('data', 0.24, 0.9 * fade);
			quads.set(qn++, { x: box.x, y: box.y, w: box.w, h: box.h, color, radius: 6 });
			const [lx, ly] = v.project(box.x, box.y);
			labels.push({
				id: `story${box.index}`,
				x: lx,
				y: ly,
				text: box.text,
				mono: true,
				size: STORY_FONT,
				tone: isEnd ? 'muted' : isGiven ? 'text' : 'data',
				opacity: fade
			});
		}
		return qn;
	}

	// ---- probe

	function drawProbe(labels: Label[], qn: number, ppw: number): number {
		hits = [];
		const a = probeMix;
		if (a <= 0.005) return qn;
		const v = ctx.view;
		const mText = mul('text');
		const mOk = mul('ok');
		const mGrad = mul('grad');
		const ck = LOG.checkpoints[currentCk()];
		const sel = selected();
		const tokens = LOG.probe;
		const boxes = layoutChips(tokens, {
			x0: PROBE_X0,
			x1: PROBE_X1,
			y: PROBE_Y,
			scale: ppw,
			fontPx: PROBE_FONT,
			padPx: 4,
			gapPx: 3,
			heightPx: 23,
			minChars: 4,
			align: 'center'
		});
		const left = Math.min(...boxes.map((b) => b.x - b.w / 2));
		const right = Math.max(...boxes.map((b) => b.x + b.w / 2));
		const centerX = (left + right) / 2;

		const [lx, ly] = v.project(left, PROBE_CAPTION_Y);
		labels.push({ id: 'cap-probe', x: lx, y: ly, text: 'guess the next token', tone: 'muted', size: 11, align: 'left', opacity: a });
		const [rx, ry] = v.project(right, PROBE_CAPTION_Y);
		labels.push({
			id: 'probe-loss',
			x: rx,
			y: ry,
			text: `loss on this sentence ${ck.probeLoss.toFixed(2)}`,
			tone: 'grad',
			mono: true,
			size: 11,
			align: 'right',
			opacity: a * mGrad
		});

		const trackH = TRACK_H;
		const trackCy = TRACK_TOP - trackH / 2;
		const inset = 2 / ppw;
		for (const box of boxes) {
			const j = box.index;
			const isSel = j === sel;
			const dim = j > sel ? 0.35 : 1;
			if (isSel) {
				const rim = 2 / ppw;
				quads.set(qn++, { x: box.x, y: box.y, w: box.w + 2 * rim, h: box.h + 2 * rim, color: tone('ok', 0.9, 0.95 * a * mOk), radius: 8 });
			}
			quads.set(qn++, { x: box.x, y: box.y, w: box.w, h: box.h, color: tone('text', isSel ? 0.22 : 0.16, 0.9 * a * mText * dim), radius: 6 });
			const [px, py] = v.project(box.x, box.y);
			labels.push({ id: `probe${j}`, x: px, y: py, text: box.text, mono: true, size: PROBE_FONT, tone: 'text', opacity: a * mText * dim });
			if (j === 0) continue;

			const tw = box.w - 6 / ppw;
			quads.set(qn++, {
				x: box.x,
				y: trackCy,
				w: tw,
				h: trackH,
				color: isSel ? tone('ok', 0.3, 0.3 * a * mOk) : tone('muted', 1, 0.28 * a * mOk),
				radius: 4
			});
			const value = bars[j - 1];
			if (value > 0) {
				const bh = Math.max(1.2 / ppw, value * (trackH - 2 * inset));
				quads.set(qn++, {
					x: box.x,
					y: TRACK_TOP - trackH + inset + bh / 2,
					w: tw - 2 * inset,
					h: bh,
					color: tone('ok', 0.9, 0.95 * a * mOk),
					radius: 2
				});
			}
			const [bx, by] = v.project(box.x, PCT_Y);
			labels.push({
				id: `pct${j}`,
				x: bx,
				y: by,
				text: percent(ck.probe[j - 1]),
				tone: isSel ? 'ok' : 'text',
				mono: true,
				size: 10,
				opacity: a * mOk * (isSel ? 1 : 0.75)
			});
			const [h0x, h0y] = v.project(box.x - box.w / 2 - 3 / ppw, box.y + box.h / 2);
			const [h1x, h1y] = v.project(box.x + box.w / 2 + 3 / ppw, PCT_Y - 0.09);
			hits[j] = { x0: h0x, x1: h1x, y0: h0y, y1: h1y };
		}

		// the detail line for the selected position
		const guess = ck.guess[sel - 1];
		const guessP = ck.guessP[sel - 1];
		const right_ = tokens[sel];
		const rightP = ck.probe[sel - 1];
		const NB = ' ';
		const text =
			guess === right_
				? `top guess here: "${tokenLabel(guess)}" ${percent(guessP)}, the right token`
				: `top guess here: "${tokenLabel(guess)}" ${percent(guessP)}${NB}${NB}·${NB}${NB}right token: "${tokenLabel(right_)}" ${percent(rightP)}`;
		const [dx, dy] = v.project(centerX, DETAIL_Y);
		labels.push({ id: 'detail', x: dx, y: dy, text, tone: 'text', size: 12, opacity: a * mOk });
		return qn;
	}

	function rebuild() {
		const v = ctx.view;
		const ppw = pxPerWorld(v, camera);
		const step = stepOfPosition(pos);
		const loss = curveLossAt(step);
		const labels: Label[] = [];
		let qn = 0;
		drawChart(labels, loss, step, qn);
		qn = drawStory(labels, qn, ppw);
		qn = drawProbe(labels, qn, ppw);
		quads.count = qn;
		quads.upload();
		frame.upload();
		under.upload();
		over.upload();
		dots.upload();
		ctx.labels.set(labels);
	}

	function advanceBars(dt: number): boolean {
		const ck = LOG.checkpoints[currentCk()];
		const k = ctx.reducedMotion || barsSnap ? 1 : 1 - Math.exp(-dt / BAR_TAU);
		let moving = false;
		for (let i = 0; i < bars.length; i++) {
			const target = ck.probe[i] ?? 0;
			const diff = target - bars[i];
			if (Math.abs(diff) > 1e-4) {
				bars[i] += diff * k;
				moving = true;
			} else bars[i] = target;
		}
		barsSnap = false;
		return moving;
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			quads = new Quads(c, { capacity: 320, blend: 'alpha' });
			frame = new Lines(c, { capacity: 140 + 3 * LOG.checkpoints.length, blend: 'alpha' });
			under = new Lines(c, { capacity: LOG.curveStep.length + 4, blend: 'alpha' });
			over = new Lines(c, { capacity: LOG.curveStep.length + 4, blend: 'alpha' });
			dots = new Points(c, { capacity: LOG.checkpoints.length + 6, blend: 'alpha' });
		},
		setParams(p) {
			prompt = clamp(Math.round(p.prompt ?? prompt), 0, LOG.prompts.length - 1);
			dice = p.dice ?? dice;
			chartOn = p.chart ?? chartOn;
			storyOn = p.story ?? storyOn;
			probeOn = p.probe ?? probeOn;
			validOn = p.valid ?? validOn;
			const at = clamp(Math.round(p.at ?? 0), 0, LAST);
			if (!paramsSeen) {
				pos = at;
				lastAt = at;
				moveT = 1;
				shownCk = at;
				shownPrompt = prompt;
				shownDice = dice >= 0.5 ? 1 : 0;
				chartMix = chartOn >= 0.5 ? 1 : 0;
				storyMix = storyOn >= 0.5 ? 1 : 0;
				probeMix = probeOn >= 0.5 ? 1 : 0;
				validMix = validOn >= 0.5 ? 1 : 0;
			} else if (at !== lastAt) {
				running = false;
				lastAt = at;
				moveTowards(at);
			}
			paramsSeen = true;
			dirty = true;
		},
		update(fr) {
			const v = ctx.view;
			const dt = fr.dt;
			fitCamera(camera, v, WORLD, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));
			if (v.width !== lastW || v.height !== lastH) {
				lastW = v.width;
				lastH = v.height;
				dirty = true;
			}
			const jump = ctx.reducedMotion || dt === 0;

			// band visibility
			const mixes: [number, number][] = [
				[chartMix, chartOn >= 0.5 ? 1 : 0],
				[storyMix, storyOn >= 0.5 ? 1 : 0],
				[probeMix, probeOn >= 0.5 ? 1 : 0],
				[validMix, validOn >= 0.5 ? 1 : 0]
			];
			const next = mixes.map(([m, t]) => (m === t ? m : jump ? t : toward(m, t, dt / BAND_TIME)));
			if (next.some((m, i) => m !== mixes[i][0])) dirty = true;
			[chartMix, storyMix, probeMix, validMix] = next;

			// position
			if (running) {
				const before = currentCk();
				if (ctx.reducedMotion) {
					runWait += dt;
					if (runWait >= CHECKPOINT_TIME_REDUCED) {
						runWait = 0;
						pos = Math.min(LAST, before + 1);
					}
				} else {
					pos = Math.min(LAST, pos + dt / CHECKPOINT_TIME);
				}
				const now = currentCk();
				if (now > before) arrive(now);
				if (pos >= LAST) {
					pos = LAST;
					running = false;
				}
				dirty = true;
			} else if (moveT < 1) {
				moveT = Math.min(1, moveT + dt / MOVE_TIME);
				pos = moveFrom + (moveTo - moveFrom) * easeInOut(moveT);
				if (moveT >= 1) pos = moveTo;
				dirty = true;
			}

			// the written text and the prompt follow the checkpoint reached, with a short fade
			const ck = currentCk();
			const wantDice = dice >= 0.5 ? 1 : 0;
			const stale = shownCk !== ck || shownPrompt !== prompt || shownDice !== wantDice;
			if (stale) {
				if (jump) {
					shownCk = ck;
					shownPrompt = prompt;
					shownDice = wantDice;
					textAlpha = 1;
					promptAlpha = 1;
				} else {
					textAlpha = Math.max(0, textAlpha - dt / FADE_TIME);
					if (shownPrompt !== prompt) promptAlpha = Math.max(0, promptAlpha - dt / FADE_TIME);
					if (textAlpha <= 0) {
						if (shownPrompt !== prompt) promptAlpha = 0;
						shownCk = ck;
						shownPrompt = prompt;
						shownDice = wantDice;
					}
				}
				dirty = true;
			} else if (textAlpha < 1 || promptAlpha < 1) {
				textAlpha = jump ? 1 : Math.min(1, textAlpha + dt / FADE_TIME);
				promptAlpha = jump ? 1 : Math.min(1, promptAlpha + dt / FADE_TIME);
				dirty = true;
			}

			if (advanceBars(dt)) dirty = true;

			if (dirty) {
				dirty = false;
				rebuild();
			}

			const step = stepOfPosition(pos);
			const loss = curveLossAt(step);
			const stepText = formatStep(step);
			if (stepText !== lastStep) {
				lastStep = stepText;
				ctx.emit('step', stepText);
			}
			if (Number.isNaN(lastLoss) || Math.abs(loss - lastLoss) > 1e-4) {
				lastLoss = loss;
				ctx.emit('loss', loss);
				// An untrained model can be slightly worse than guessing blindly; it still has only LOG.vocab tokens to choose from.
				const text = `${Math.min(LOG.vocab, choices(loss)).toLocaleString('en-US')} tokens`;
				if (text !== lastChoices) {
					lastChoices = text;
					ctx.emit('choices', text);
				}
			}
		},
		draw(pass) {
			quads.draw(pass);
			frame.draw(pass);
			under.draw(pass);
			over.draw(pass);
			dots.draw(pass);
		},
		command(id) {
			if (id === 'train') {
				if (running) {
					running = false;
					const k = currentCk();
					if (pos !== k) moveTowards(k);
				} else {
					if (moveT < 1) {
						pos = moveTo;
						moveT = 1;
					}
					if (pos >= LAST - 1e-6) {
						pos = 0;
						arrive(0);
					}
					running = true;
					runWait = 0;
				}
			} else if (id === 'reset') {
				running = false;
				pos = 0;
				moveT = 1;
				arrive(0);
			}
			dirty = true;
		},
		pointer(e: ScenePointer) {
			let next = -1;
			if (e.type !== 'leave' && probeMix > 0.5) {
				for (let j = 1; j < hits.length; j++) {
					const h = hits[j];
					if (h && e.x >= h.x0 && e.x <= h.x1 && e.y >= h.y0 && e.y <= h.y1) {
						next = j;
						break;
					}
				}
			}
			if (next !== hover) {
				hover = next;
				dirty = true;
			}
		},
		highlight(t) {
			hl = t;
			dirty = true;
		},
		dispose() {
			quads.destroy();
			frame.destroy();
			under.destroy();
			over.destroy();
			dots.destroy();
		}
	};
	return scene;
};

import { Camera2D } from '../../gpu/camera.ts';
import { tone } from '../../gpu/colors.ts';
import { Lines } from '../../gpu/primitives/lines.ts';
import { Points } from '../../gpu/primitives/points.ts';
import type { Scene, SceneContext, SceneFactory, Tone } from '../../gpu/types.ts';
import { MAP_H, MAP_W, links, stars } from '../../content/constellation.ts';
import { mulberry32 } from '../../ml/rng.ts';
import { fitCamera } from '../fit.ts';

const BACKGROUND = 140;
const TRACK_TONE: Record<string, Tone> = {
	fundamentals: 'data',
	'classic-ml': 'ok',
	llm: 'param',
	vision: 'grad'
};
/** Links stop this far (design units) short of each star's center. */
const GAP = 9;
const NO_INSETS = { top: 0, right: 0, bottom: 0, left: 0 };
const WORLD = { x0: 0, x1: MAP_W, y0: 0, y1: MAP_H };
/** Seconds the hover highlight takes to ease in or out. */
const EASE = 0.1;

export const createConstellationScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let bg: Points;
	let lines: Lines;
	let glows: Points;
	let cores: Points;
	let rings: Points;

	let done = 0;
	let hover = -1;
	let next = -1;

	const bx = new Float32Array(BACKGROUND);
	const by = new Float32Array(BACKGROUND);
	const br = new Float32Array(BACKGROUND);
	const bi = new Float32Array(BACKGROUND);
	const bs = new Float32Array(BACKGROUND);
	const bp = new Float32Array(BACKGROUND);
	/** Eased hover amount per star, 0..1. */
	const amount = new Float32Array(stars.length);

	const isDone = (i: number) => (done & (1 << i)) !== 0;
	const starTone = (i: number): Tone => TRACK_TONE[stars[i].track] ?? 'data';

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			bg = new Points(c, { capacity: BACKGROUND, blend: 'additive' });
			lines = new Lines(c, { capacity: Math.max(1, links.length), blend: 'additive' });
			glows = new Points(c, { capacity: stars.length, blend: 'additive' });
			cores = new Points(c, { capacity: stars.length, blend: 'additive' });
			rings = new Points(c, { capacity: stars.length * 2, blend: 'additive' });

			const rnd = mulberry32(4242);
			for (let i = 0; i < BACKGROUND; i++) {
				bx[i] = rnd() * MAP_W;
				by[i] = rnd() * MAP_H;
				br[i] = 0.6 + rnd() * 0.8;
				bi[i] = 0.25 + rnd() * 0.35;
				bs[i] = 0.4 + rnd() * 0.8;
				bp[i] = rnd() * Math.PI * 2;
			}
		},
		setParams(p) {
			done = Math.round(p.done ?? 0);
			hover = Math.round(p.hover ?? -1);
			next = Math.round(p.next ?? -1);
		},
		update(frame) {
			const v = ctx.view;
			const still = ctx.reducedMotion;
			fitCamera(camera, v, WORLD, NO_INSETS);
			v.setViewProj(camera.matrix(v.aspect));

			// Background stars.
			for (let i = 0; i < BACKGROUND; i++) {
				const twinkle = still ? 1 : 0.75 + 0.25 * Math.sin(frame.time * bs[i] + bp[i]);
				bg.set(i, {
					x: bx[i],
					y: MAP_H - by[i],
					radius: br[i],
					softness: 0.6,
					color: tone('text', bi[i] * twinkle, 1)
				});
			}
			bg.count = BACKGROUND;

			// Ease the hover highlight.
			for (let i = 0; i < stars.length; i++) {
				const target = i === hover ? 1 : 0;
				if (still) amount[i] = target;
				else {
					const step = frame.dt / EASE;
					amount[i] = target > amount[i] ? Math.min(target, amount[i] + step) : Math.max(target, amount[i] - step);
				}
			}

			// Links.
			let n = 0;
			for (const l of links) {
				const a = stars[l.from];
				const b = stars[l.to];
				const dx = b.x - a.x;
				const dy = b.y - a.y;
				const len = Math.hypot(dx, dy);
				if (len < GAP * 2 + 1) continue;
				const ux = dx / len;
				const uy = dy / len;
				const both = isDone(l.from) && isDone(l.to);
				const touched = hover >= 0 && (l.from === hover || l.to === hover);
				let t: Tone;
				let width: number;
				let intensity: number;
				let alpha: number;
				if (l.cross) {
					t = 'text';
					width = 1;
					intensity = 0.8;
					alpha = both ? 0.3 : 0.12;
				} else {
					t = starTone(l.from);
					width = 1.2;
					intensity = 0.9;
					alpha = both ? 0.6 : 0.22;
				}
				if (touched) alpha = 0.65;
				const pulse = touched && !still;
				lines.set(n++, {
					a: [a.x + ux * GAP, MAP_H - (a.y + uy * GAP)],
					b: [b.x - ux * GAP, MAP_H - (b.y - uy * GAP)],
					width,
					color: tone(t, intensity, alpha),
					pulseSpeed: pulse ? 0.5 : 0,
					pulseCount: 1,
					pulseGain: pulse ? 3 : 0
				});
			}
			lines.count = n;

			// Stars.
			let rc = 0;
			for (const s of stars) {
				const i = s.index;
				const t = starTone(i);
				const h = amount[i];
				const wy = MAP_H - s.y;
				const complete = isDone(i);
				const glowR = (complete ? 22 : 9) * (1 + 0.5 * h);
				glows.set(i, {
					x: s.x,
					y: wy,
					radius: glowR,
					softness: 1,
					color: complete ? tone(t, 0.9, 0.35) : tone(t, 0.5, 0.18)
				});
				cores.set(i, {
					x: s.x,
					y: wy,
					radius: complete ? 4.5 : 3,
					softness: 0.3,
					color: complete ? tone(t, 2.2, 1) : tone(t, 0.7, 0.85)
				});
				if (i === next) {
					const phase = still ? 0 : 0.5 - 0.5 * Math.cos((frame.time / 2.4) * Math.PI * 2);
					const r = still ? 10 : 9 + 4 * phase;
					const a = still ? 0.5 : 0.6 - 0.4 * phase;
					rings.set(rc++, { x: s.x, y: wy, radius: r, ring: 1.5, softness: 0.2, color: tone('text', 0.9, a) });
				}
				if (h > 0.001) {
					rings.set(rc++, { x: s.x, y: wy, radius: 12, ring: 1.5, softness: 0.2, color: tone(t, 1.2, 0.8 * h) });
				}
			}
			glows.count = stars.length;
			cores.count = stars.length;
			rings.count = rc;

			bg.upload();
			lines.upload();
			glows.upload();
			cores.upload();
			rings.upload();
		},
		draw(pass) {
			bg.draw(pass);
			lines.draw(pass);
			glows.draw(pass);
			cores.draw(pass);
			rings.draw(pass);
		},
		dispose() {
			bg.destroy();
			lines.destroy();
			glows.destroy();
			cores.destroy();
			rings.destroy();
		}
	};
	return scene;
};

import { Camera2D } from '../../gpu/camera.ts';
import { tone } from '../../gpu/colors.ts';
import { Lines } from '../../gpu/primitives/lines.ts';
import { Points } from '../../gpu/primitives/points.ts';
import type { Scene, SceneContext, SceneFactory } from '../../gpu/types.ts';

const NODES = 70;
const LINE_CAP = 600;
const WORLD_H = 10;
const LINK = 2.2;

function mulberry32(seed: number) {
	let a = seed;
	return () => {
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** Brightness factor across the stage: 0.15 at the left edge rising to 1 at 60% of the width. */
function fade(u: number): number {
	const t = Math.min(1, Math.max(0, u / 0.6));
	const s = t * t * (3 - 2 * t);
	return 0.15 + 0.85 * s;
}

export const createHeroScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let lines: Lines;
	let glows: Points;
	let dots: Points;

	const px = new Float32Array(NODES);
	const py = new Float32Array(NODES);
	const vx = new Float32Array(NODES);
	const vy = new Float32Array(NODES);
	const kind = new Uint8Array(NODES);
	let wx = 0; // world half-extents
	let wy = WORLD_H / 2;
	let seeded = false;

	function seed() {
		const rnd = mulberry32(1234);
		for (let i = 0; i < NODES; i++) {
			// unit positions in [-1, 1]; scaled to the world on first frame and on resize
			px[i] = rnd() * 2 - 1;
			py[i] = rnd() * 2 - 1;
			const a = rnd() * Math.PI * 2;
			const s = 0.1 + rnd() * 0.18;
			vx[i] = Math.cos(a) * s;
			vy[i] = Math.sin(a) * s;
			kind[i] = rnd() < 0.18 ? 1 : 0;
		}
	}

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			lines = new Lines(c, { capacity: LINE_CAP, blend: 'additive' });
			glows = new Points(c, { capacity: NODES, blend: 'additive' });
			dots = new Points(c, { capacity: NODES, blend: 'additive' });
			seed();
		},
		setParams() {},
		update(frame) {
			const v = ctx.view;
			const motion = ctx.reducedMotion ? 0 : 1;
			const nwy = WORLD_H / 2;
			const nwx = nwy * v.aspect;
			if (!seeded) {
				for (let i = 0; i < NODES; i++) {
					px[i] *= nwx;
					py[i] *= nwy;
				}
				seeded = true;
			} else if (nwx !== wx) {
				const k = nwx / wx;
				for (let i = 0; i < NODES; i++) px[i] *= k;
			}
			wx = nwx;
			wy = nwy;

			camera.viewHeight = WORLD_H;
			camera.center = [0, 0];
			v.setViewProj(camera.matrix(v.aspect));

			const dt = frame.dt * motion;
			for (let i = 0; i < NODES; i++) {
				let x = px[i] + vx[i] * dt;
				let y = py[i] + vy[i] * dt;
				if (x > wx) {
					x = wx;
					vx[i] = -Math.abs(vx[i]);
				} else if (x < -wx) {
					x = -wx;
					vx[i] = Math.abs(vx[i]);
				}
				if (y > wy) {
					y = wy;
					vy[i] = -Math.abs(vy[i]);
				} else if (y < -wy) {
					y = -wy;
					vy[i] = Math.abs(vy[i]);
				}
				px[i] = x;
				py[i] = y;
			}

			const shade = (x: number) => 0.8 * fade((x + wx) / (2 * wx));
			let gi = 0;
			let di = 0;
			for (let i = 0; i < NODES; i++) {
				const f = shade(px[i]);
				const t = kind[i] === 1 ? 'param' : 'data';
				glows.set(gi++, { x: px[i], y: py[i], radius: 16, softness: 1, color: tone(t, 0.9 * f, 0.35 * f) });
				dots.set(di++, { x: px[i], y: py[i], radius: 2.4, softness: 0.4, color: tone(t, 1.6 * f, 0.9 * f) });
			}
			glows.count = gi;
			dots.count = di;

			let n = 0;
			const thr2 = LINK * LINK;
			for (let i = 0; i < NODES && n < LINE_CAP; i++) {
				for (let j = i + 1; j < NODES && n < LINE_CAP; j++) {
					const dx = px[j] - px[i];
					const dy = py[j] - py[i];
					const d2 = dx * dx + dy * dy;
					if (d2 >= thr2) continue;
					const a = 1 - Math.sqrt(d2) / LINK;
					const f = shade((px[i] + px[j]) / 2);
					const pulsing = (i * 7 + j * 3) % 11 === 0;
					lines.set(n++, {
						a: [px[i], py[i]],
						b: [px[j], py[j]],
						width: 1,
						color: tone('param', 0.9 * f, 0.55 * a * f),
						pulseSpeed: pulsing ? 0.12 * motion : 0,
						pulseCount: 1,
						pulseGain: pulsing ? 4 * motion : 0
					});
				}
			}
			lines.count = n;

			lines.upload();
			glows.upload();
			dots.upload();
		},
		draw(pass) {
			lines.draw(pass);
			glows.draw(pass);
			dots.draw(pass);
		},
		dispose() {
			lines.destroy();
			glows.destroy();
			dots.destroy();
		}
	};
	return scene;
};

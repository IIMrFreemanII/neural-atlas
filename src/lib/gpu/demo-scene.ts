import { Camera2D } from './camera.ts';
import { tone } from './colors.ts';
import { Lines, bezier } from './primitives/lines.ts';
import { Points } from './primitives/points.ts';
import type { Label, Scene, SceneContext, SceneFactory } from './types.ts';

type Node = { x: number; y: number };

const INPUT_X = -1.4;
const HIDDEN_X = 0;
const OUTPUT_X = 1.4;

export const createDemoScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let lines: Lines;
	let glows: Points;
	let nodes: Points;

	const inputs: Node[] = [-0.7, 0, 0.7].map((y) => ({ x: INPUT_X, y }));
	const hidden: Node[] = [-0.825, -0.275, 0.275, 0.825].map((y) => ({ x: HIDDEN_X, y }));
	const output: Node = { x: OUTPUT_X, y: 0 };

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			lines = new Lines(c, { capacity: 640, blend: 'additive' });
			glows = new Points(c, { capacity: 32, blend: 'additive' });
			nodes = new Points(c, { capacity: 32, blend: 'alpha' });

			const motion = c.reducedMotion ? 0 : 1;
			const edgeColor = tone('param', 1.3, 0.55);
			let n = 0;
			const edge = (a: Node, b: Node) => {
				const dx = b.x - a.x;
				const pts = bezier(
					[a.x, a.y],
					[a.x + dx * 0.5, a.y],
					[b.x - dx * 0.5, b.y],
					[b.x, b.y],
					24
				);
				n = lines.setPolyline(n, pts, {
					width: 1.5,
					color: edgeColor,
					pulseSpeed: 0.22 * motion,
					pulseCount: 2,
					pulseGain: 3 * motion
				});
			};
						for (const a of inputs) for (const b of hidden) edge(a, b);
			for (const a of hidden) edge(a, output);
			lines.count = n;
			lines.upload();

			let g = 0;
			let p = 0;
			const all = [...inputs, ...hidden];
			for (const nd of all) {
				glows.set(g++, { x: nd.x, y: nd.y, radius: 26, color: tone('data', 2, 0.55), softness: 1 });
				nodes.set(p++, { x: nd.x, y: nd.y, radius: 11, color: tone('data', 1.2), ring: 2 });
			}
			glows.set(g++, { x: output.x, y: output.y, radius: 44, color: tone('grad', 2.5, 0.7), softness: 1 });
			nodes.set(p++, { x: output.x, y: output.y, radius: 13, color: tone('grad', 3.5) });
			glows.count = g;
			nodes.count = p;
			glows.upload();
			nodes.upload();
		},
		setParams() {},
		update(frame) {
			const v = ctx.view;
			camera.viewHeight = Math.max(2.6, 3.8 / v.aspect);
			v.setViewProj(camera.matrix(v.aspect));

			const label = (id: string, x: number, y: number, text: string): Label => {
				const [px, py] = v.project(x, y);
				return { id, x: px, y: py + 14, text, tone: 'muted' };
			};
			ctx.labels.set([
				label('inputs', INPUT_X, -0.7, 'inputs'),
				label('hidden', HIDDEN_X, -0.825, 'hidden'),
				label('output', OUTPUT_X, 0, 'output')
			]);
			ctx.emit('time', frame.time.toFixed(1));
		},
		draw(pass) {
			lines.draw(pass);
			glows.draw(pass);
			nodes.draw(pass);
		},
		dispose() {
			lines.destroy();
			glows.destroy();
			nodes.destroy();
		}
	};
	return scene;
};

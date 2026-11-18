import { Camera2D } from '../../gpu/camera.ts';
import type { Scene, SceneContext, SceneFactory } from '../../gpu/types.ts';
import { createMLP, forward } from '../../ml/mlp.ts';
import { fitCamera, lessonInsets } from '../fit.ts';
import { NetworkDiagram } from './network-diagram.ts';

const SIZES = [2, 4, 3, 1];
const fmt = (v: number) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));

export const createNetworkDemoScene: SceneFactory = () => {
	let ctx: SceneContext;
	let camera: Camera2D;
	let diagram: NetworkDiagram;
	const net = createMLP(SIZES, 'tanh', 'sigmoid', 7);
	let time = 0;

	const scene: Scene = {
		init(c) {
			ctx = c;
			camera = new Camera2D();
			diagram = new NetworkDiagram(c, SIZES);
		},
		setParams() {},
		update(frame) {
			time += frame.dt;
			const v = ctx.view;
			fitCamera(camera, v, diagram.rect, lessonInsets(v));
			v.setViewProj(camera.matrix(v.aspect));

			const ang = time * 0.5;
			const tr = forward(net, [0.9 * Math.cos(ang), 0.9 * Math.sin(ang)]);
			const last = SIZES.length - 1;
			for (let l = 0; l < SIZES.length; l++) {
				for (let i = 0; i < SIZES[l]; i++) {
					const s = diagram.node(l, i);
					s.value = tr.a[l][i];
					s.label = fmt(tr.a[l][i]);
					if (l === 0) s.name = i === 0 ? 'x₁' : 'x₂';
					if (l === last) s.name = 'ŷ';
				}
			}
			for (let l = 0; l < last; l++) {
				for (let i = 0; i < SIZES[l]; i++) {
					for (let j = 0; j < SIZES[l + 1]; j++) {
						const w = net.W[l][j * SIZES[l] + i];
						const e = diagram.edge(l, i, j);
						e.weight = w;
						e.signal = Math.abs(tr.a[l][i] * w);
					}
				}
			}
			diagram.commit();
			ctx.labels.set(diagram.labels(v));
		},
		draw(pass) {
			diagram.draw(pass);
		},
		dispose() {
			diagram.destroy();
		}
	};
	return scene;
};

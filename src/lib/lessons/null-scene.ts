import type { SceneFactory } from '../gpu/types.ts';

export const createNullScene: SceneFactory = () => ({
	init() {},
	setParams() {},
	update() {},
	draw() {},
	dispose() {}
});

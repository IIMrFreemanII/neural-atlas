import type { Camera2D } from '../gpu/camera.ts';
import type { Insets, View } from '../gpu/types.ts';

export type { Insets } from '../gpu/types.ts';

export interface WorldRect {
	x0: number;
	x1: number;
	y0: number;
	y1: number;
}

/**
 * The parts of a lesson stage covered by UI overlays, in CSS px:
 * legend and readout chips along the top, the control bar along the bottom.
 * Returns view.insets when the host set them, else these defaults.
 */
export function lessonInsets(
	view: Pick<View, 'width' | 'height'> & { insets?: Insets | null }
): Insets {
	if (view.insets) return view.insets;
	return {
		top: 64,
		right: 24,
		bottom: Math.min(170, view.height * 0.32),
		left: 24
	};
}

/**
 * Point a Camera2D so that `world` is fully visible and centered inside the stage area
 * that remains after removing `insets`. Call every frame from Scene.update (it is cheap),
 * then pass camera.matrix(view.aspect) to view.setViewProj.
 */
export function fitCamera(
	camera: Camera2D,
	view: Pick<View, 'width' | 'height'>,
	world: WorldRect,
	insets: Insets
): void {
	const availW = Math.max(1, view.width - insets.left - insets.right);
	const availH = Math.max(1, view.height - insets.top - insets.bottom);
	const worldW = Math.max(1e-6, world.x1 - world.x0);
	const worldH = Math.max(1e-6, world.y1 - world.y0);
	// CSS px per world unit
	const scale = Math.min(availW / worldW, availH / worldH);
	const dxCss = insets.left + availW / 2 - view.width / 2;
	const dyCss = insets.top + availH / 2 - view.height / 2;
	camera.viewHeight = Math.max(1, view.height) / scale;
	camera.center = [
		(world.x0 + world.x1) / 2 - dxCss / scale,
		(world.y0 + world.y1) / 2 + dyCss / scale
	];
}

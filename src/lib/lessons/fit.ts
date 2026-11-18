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

/**
 * For 3D scenes: the aspect ratio of the stage area left after removing `insets`.
 * Build the camera's projection with this aspect, then wrap it with insetMatrix().
 */
export function insetAspect(view: Pick<View, 'width' | 'height'>, insets: Insets): number {
	const w = Math.max(1, view.width - insets.left - insets.right);
	const h = Math.max(1, view.height - insets.top - insets.bottom);
	return w / h;
}

/**
 * For 3D scenes: a clip-space matrix that squeezes a full-stage projection into the stage area
 * left after removing `insets`, so the figure is centred in the part not covered by the card,
 * chips and control bar. Use as `view.setViewProj(mul(insetMatrix(view, ins), camera.matrix(insetAspect(view, ins))))`
 * (column-major, e.g. wgpu-matrix `mat4.multiply`). Picking with rayFromCss must use the same combined matrix.
 */
export function insetMatrix(view: Pick<View, 'width' | 'height'>, insets: Insets): Float32Array {
	const W = Math.max(1, view.width);
	const H = Math.max(1, view.height);
	const w = Math.max(1, W - insets.left - insets.right);
	const h = Math.max(1, H - insets.top - insets.bottom);
	const sx = w / W;
	const sy = h / H;
	const tx = ((insets.left + w / 2) / W) * 2 - 1;
	const ty = 1 - ((insets.top + h / 2) / H) * 2;
	// Column-major: x' = sx*x + tx*w, y' = sy*y + ty*w.
	return new Float32Array([sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, 1, 0, tx, ty, 0, 1]);
}

/**
 * Tiny grayscale raster helpers: soft-edged strokes on a square pixel grid.
 * Pixel (col, row) has its center at (col + 0.5, row + 0.5); row 0 is the top row. Values are 0..1.
 * The same stroke function draws the generated shapes and the learner's pointer strokes, so both look alike.
 */

/** Distance from point (px, py) to the segment a-b. */
export function segmentDistance(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
	const dx = bx - ax;
	const dy = by - ay;
	const len2 = dx * dx + dy * dy;
	let t = len2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
	t = Math.max(0, Math.min(1, t));
	return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/**
 * Stroke the segment a-b (pixel coordinates) into an n x n image.
 * A pixel's coverage is clamp(halfWidth + 0.5 - distance, 0, 1). Drawing keeps the larger of the old value and the
 * coverage; with erase = true it keeps the smaller of the old value and 1 - coverage.
 */
export function strokeSegment(
	data: Float32Array,
	n: number,
	ax: number,
	ay: number,
	bx: number,
	by: number,
	halfWidth: number,
	erase = false
): void {
	const r = halfWidth + 0.5;
	const c0 = Math.max(0, Math.floor(Math.min(ax, bx) - r));
	const c1 = Math.min(n - 1, Math.floor(Math.max(ax, bx) + r));
	const r0 = Math.max(0, Math.floor(Math.min(ay, by) - r));
	const r1 = Math.min(n - 1, Math.floor(Math.max(ay, by) + r));
	for (let row = r0; row <= r1; row++) {
		for (let col = c0; col <= c1; col++) {
			const d = segmentDistance(col + 0.5, row + 0.5, ax, ay, bx, by);
			const cover = Math.max(0, Math.min(1, r - d));
			if (cover <= 0) continue;
			const i = row * n + col;
			data[i] = erase ? Math.min(data[i], 1 - cover) : Math.max(data[i], cover);
		}
	}
}

/** Stroke a polyline given as [x, y] points (pixel coordinates). */
export function strokePath(
	data: Float32Array,
	n: number,
	points: ReadonlyArray<readonly [number, number]>,
	halfWidth: number
): void {
	for (let i = 0; i + 1 < points.length; i++) {
		strokeSegment(data, n, points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], halfWidth);
	}
}

export const SHAPE_KINDS = ['circle', 'square', 'triangle', 'cross'] as const;
export type ShapeKind = (typeof SHAPE_KINDS)[number];

/**
 * Outline of a shape as one or more polylines (pixel coordinates), centered at (cx, cy) with "radius" r
 * (distance from the center to a corner, or the circle's radius) and rotated by rot radians.
 * circle: a 28-gon. square: 4 corners. triangle: 3 corners. cross: two strokes crossing at the center.
 */
export function shapePaths(kind: ShapeKind, cx: number, cy: number, r: number, rot = 0): Array<Array<[number, number]>> {
	const at = (angle: number, radius = r): [number, number] => [
		cx + radius * Math.cos(angle + rot),
		cy + radius * Math.sin(angle + rot)
	];
	const ring = (sides: number, phase: number): Array<[number, number]> => {
		const pts: Array<[number, number]> = [];
		for (let i = 0; i <= sides; i++) pts.push(at(phase + (i / sides) * Math.PI * 2));
		return pts;
	};
	if (kind === 'circle') return [ring(28, 0)];
	if (kind === 'square') return [ring(4, Math.PI / 4)];
	if (kind === 'triangle') return [ring(3, -Math.PI / 2)];
	return [
		[at(0), at(Math.PI)],
		[at(Math.PI / 2), at(-Math.PI / 2)]
	];
}

/** A new n x n image (all 0) with the shape's outline stroked into it. */
export function drawShape(kind: ShapeKind, n: number, cx: number, cy: number, r: number, rot = 0, halfWidth = 0.7): Float32Array {
	const data = new Float32Array(n * n);
	for (const path of shapePaths(kind, cx, cy, r, rot)) strokePath(data, n, path, halfWidth);
	return data;
}

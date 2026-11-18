/**
 * Marching squares on a (cols x rows) grid of samples, values[row * cols + col].
 * Returns line segments in grid coordinates (fractional col, row), as [x0, y0, x1, y1].
 * The two ambiguous saddle cases are resolved with the cell-center average.
 */
export function contourSegments(
	values: Float32Array,
	cols: number,
	rows: number,
	level: number
): Array<[number, number, number, number]> {
	const out: Array<[number, number, number, number]> = [];
	for (let r = 0; r < rows - 1; r++) {
		for (let c = 0; c < cols - 1; c++) {
			const tl = values[r * cols + c];
			const tr = values[r * cols + c + 1];
			const bl = values[(r + 1) * cols + c];
			const br = values[(r + 1) * cols + c + 1];
			const idx =
				(tl >= level ? 8 : 0) | (tr >= level ? 4 : 0) | (br >= level ? 2 : 0) | (bl >= level ? 1 : 0);
			if (idx === 0 || idx === 15) continue;

			const top = (): [number, number] => [c + (level - tl) / (tr - tl), r];
			const bottom = (): [number, number] => [c + (level - bl) / (br - bl), r + 1];
			const left = (): [number, number] => [c, r + (level - tl) / (bl - tl)];
			const right = (): [number, number] => [c + 1, r + (level - tr) / (br - tr)];
			const seg = (a: [number, number], b: [number, number]) => out.push([a[0], a[1], b[0], b[1]]);

			switch (idx) {
				case 1:
				case 14:
					seg(left(), bottom());
					break;
				case 2:
				case 13:
					seg(bottom(), right());
					break;
				case 3:
				case 12:
					seg(left(), right());
					break;
				case 4:
				case 11:
					seg(top(), right());
					break;
				case 6:
				case 9:
					seg(top(), bottom());
					break;
				case 7:
				case 8:
					seg(top(), left());
					break;
				case 5: {
					// tr and bl high
					const centerHigh = (tl + tr + bl + br) / 4 >= level;
					if (centerHigh) {
						seg(left(), top());
						seg(bottom(), right());
					} else {
						seg(top(), right());
						seg(left(), bottom());
					}
					break;
				}
				case 10: {
					// tl and br high
					const centerHigh = (tl + tr + bl + br) / 4 >= level;
					if (centerHigh) {
						seg(top(), right());
						seg(left(), bottom());
					} else {
						seg(left(), top());
						seg(bottom(), right());
					}
					break;
				}
			}
		}
	}
	return out;
}

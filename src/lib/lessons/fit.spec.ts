import { describe, expect, it } from 'vitest';
import { projectToCss } from '../gpu/view.ts';
import { insetAspect, insetMatrix } from './fit.ts';

describe('insetMatrix', () => {
	const view = { width: 1000, height: 600 };
	const ins = { top: 60, right: 420, bottom: 140, left: 20 };

	it('maps the clip-space centre to the centre of the uncovered area', () => {
		const [x, y] = projectToCss(insetMatrix(view, ins), view.width, view.height, 0, 0, 0);
		expect(x).toBeCloseTo(20 + (1000 - 440) / 2);
		expect(y).toBeCloseTo(60 + (600 - 200) / 2);
	});

	it('maps the clip-space corners to the corners of the uncovered area', () => {
		const m = insetMatrix(view, ins);
		const [x0, y0] = projectToCss(m, view.width, view.height, -1, 1, 0);
		const [x1, y1] = projectToCss(m, view.width, view.height, 1, -1, 0);
		expect([x0, y0]).toEqual([expect.closeTo(20), expect.closeTo(60)]);
		expect([x1, y1]).toEqual([expect.closeTo(580), expect.closeTo(460)]);
	});

	it('gives the aspect of the uncovered area', () => {
		expect(insetAspect(view, ins)).toBeCloseTo(560 / 400);
	});
});

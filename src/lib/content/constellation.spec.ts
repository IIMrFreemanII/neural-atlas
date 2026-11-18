import { describe, expect, it } from 'vitest';
import { MAP_H, MAP_W, TRACK_LABELS, doneMask, links, nextUp, stars } from './constellation.ts';
import { allLessons, tracks } from './curriculum.ts';

/** Distance from point p to segment a-b. */
function segDist(p: [number, number], a: [number, number], b: [number, number]): number {
	const dx = b[0] - a[0];
	const dy = b[1] - a[1];
	const t = Math.min(1, Math.max(0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)));
	return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

describe('constellation layout', () => {
	it('has one star per lesson, in curriculum order', () => {
		expect(stars.length).toBe(allLessons.length);
		stars.forEach((s, i) => {
			expect(s.index).toBe(i);
			expect(s.key).toBe(`${allLessons[i].track}/${allLessons[i].id}`);
		});
	});

	it('keeps every star inside the map with room for its label', () => {
		for (const s of stars) {
			expect(s.x).toBeGreaterThanOrEqual(60);
			expect(s.x).toBeLessThanOrEqual(MAP_W - 40);
			expect(s.y).toBeGreaterThanOrEqual(50);
			expect(s.y).toBeLessThanOrEqual(MAP_H - 50);
		}
	});

	it('keeps stars at least 70 units apart', () => {
		for (let i = 0; i < stars.length; i++)
			for (let j = i + 1; j < stars.length; j++)
				expect(Math.hypot(stars[i].x - stars[j].x, stars[i].y - stars[j].y)).toBeGreaterThanOrEqual(70);
	});

	it('draws no track line within 30 units of a star it does not join (cross-track links are curves)', () => {
		for (const l of links.filter((x) => !x.cross))
			for (const s of stars) {
				if (s.index === l.from || s.index === l.to) continue;
				const a = stars[l.from];
				const b = stars[l.to];
				expect(segDist([s.x, s.y], [a.x, a.y], [b.x, b.y])).toBeGreaterThanOrEqual(30);
			}
	});

	it('links every lesson to the one before it in its track, plus four cross-track links', () => {
		const inTrack = allLessons.length - tracks.length;
		expect(links.filter((l) => !l.cross).length).toBe(inTrack);
		expect(links.filter((l) => l.cross).length).toBe(4);
		for (const l of links) expect(l.from).not.toBe(l.to);
	});

	it('has a label position for every track', () => {
		for (const t of tracks) expect(TRACK_LABELS[t.id]).toBeDefined();
	});
});

describe('progress on the map', () => {
	it('packs completed lessons into a bit mask', () => {
		expect(doneMask(() => false)).toBe(0);
		expect(doneMask((k) => k === stars[2].key)).toBe(1 << 2);
		expect(doneMask(() => true)).toBe(2 ** stars.length - 1);
	});

	it('suggests the first lesson, then follows the curriculum order', () => {
		expect(nextUp(0)).toBe(0);
		expect(nextUp(1)).toBe(1);
		expect(nextUp(2 ** stars.length - 1)).toBe(-1);
	});

	it('skips a lesson whose prerequisites are not complete', () => {
		// Everything in the first track except its second lesson: the third is done, so the suggestion is the second.
		const first = tracks[0].lessons.length;
		const mask = (2 ** first - 1) & ~(1 << 1);
		expect(nextUp(mask)).toBe(1);
		// With the first track complete, the next suggestion is the first lesson of the second track.
		expect(nextUp(2 ** first - 1)).toBe(first);
	});
});

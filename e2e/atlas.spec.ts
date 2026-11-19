import { expect, test, type Page } from '@playwright/test';
import { tracks, lessonHref } from '../src/lib/content/curriculum.ts';

const fundamentals = tracks[0].lessons;
const atlas = (page: Page) => page.getByRole('region', { name: 'Interactive learning atlas' });
const preview = (page: Page) => page.getByRole('complementary', { name: 'Topic and lesson preview' });

async function ready(page: Page) {
	await expect(page.locator('.scene-wrap .stage')).not.toHaveAttribute('data-status', 'loading');
	const status = await page.locator('.scene-wrap .stage').getAttribute('data-status');
	if (status === 'nogpu') test.skip(true, 'WebGPU is unavailable in this browser');
	expect(status).toBe('ready');
}

for (const theme of ['paper', 'night'] as const) {
	test(`the WebGPU atlas opens every lesson preview in ${theme}`, async ({ page }) => {
		const errors: string[] = [];
		page.on('pageerror', (error) => errors.push(String(error)));
		page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
		await page.addInitScript((theme) => localStorage.setItem('neural-atlas:settings:v1', JSON.stringify({ theme })), theme);
		await page.goto('/');
		await ready(page);
		await expect(atlas(page).getByRole('button', { name: /^Explore / })).toHaveCount(4);
		for (const [trackIndex, track] of tracks.entries()) {
			await page.getByRole('navigation', { name: 'Atlas regions' }).getByRole('button').nth(trackIndex).click();
			await expect(atlas(page)).toHaveAttribute('data-view', track.id);
			await expect(atlas(page).getByRole('button', { name: /^Preview / })).toHaveCount(track.lessons.length);
			await expect(preview(page).getByRole('heading')).toBeVisible();
			for (let i = 0; i < track.lessons.length; i++) {
				const lesson = track.lessons[i];
				await atlas(page).getByRole('button', { name: `Preview ${lesson.title}`, exact: true }).click();
				await expect(preview(page).getByRole('heading', { level: 2 })).toHaveText(lesson.title);
				await expect(preview(page).getByRole('link', { name: 'Start lesson' })).toHaveAttribute('href', lessonHref(lesson));
				await expect(preview(page).getByRole('img')).toHaveAttribute('src', `/previews/${track.id}/${lesson.id}.jpg`);
				await expect(page).toHaveURL(/\/$/);
				await expect(atlas(page)).toHaveAttribute('data-lesson', String(i));
				await expect(atlas(page)).toHaveAttribute('data-view', track.id);
				await expect(page.getByRole('navigation', { name: `${track.title} learning path` }).getByRole('button')).toHaveCount(track.lessons.length);
			}
			await preview(page).getByRole('button', { name: `Back to ${track.title} overview`, exact: true }).click();
			await expect(preview(page).getByRole('link', { name: 'Start lesson' })).toHaveCount(0);
			await preview(page).getByRole('button', { name: new RegExp(track.lessons[0].title) }).click();
			await expect(preview(page).getByRole('heading')).toHaveText(track.lessons[0].title);
		}
		await page.getByRole('group', { name: 'Theme' }).getByRole('button', { name: theme === 'paper' ? 'Night' : 'Paper', exact: true }).click();
		await ready(page);
		await expect(preview(page).getByRole('heading')).toHaveText('Convolution');
		await expect(atlas(page).getByRole('button', { name: /^Preview / })).toHaveCount(tracks[3].lessons.length);
		await page.getByRole('button', { name: 'Whole atlas' }).click();
		await expect(atlas(page)).toHaveAttribute('data-view', 'overview');
		await expect(preview(page)).toHaveCount(0);
		await expect(atlas(page).getByRole('button', { name: /^Explore / })).toHaveCount(4);
		expect(errors).toEqual([]);
	});
}

test('starting points lead to previews and Start enters the chosen lesson', async ({ page }) => {
	await page.goto('/');
	await page.getByRole('button', { name: 'New to AI' }).click();
	await expect(preview(page).getByRole('heading')).toHaveText('The neuron');
	await expect(preview(page).getByText('No prior knowledge needed')).toBeVisible();
	await page.getByRole('button', { name: 'I know the basics' }).click();
	await expect(preview(page).getByRole('heading')).toHaveText('Forward pass');
	await expect(preview(page).getByRole('link', { name: 'Activation functions' })).toHaveAttribute('href', lessonHref(fundamentals[1]));
	await preview(page).getByRole('link', { name: 'Start lesson' }).click();
	await expect(page).toHaveURL(/\/learn\/fundamentals\/forward-pass$/);
	await expect(page.getByRole('heading', { level: 1 })).toHaveText('Forward pass');
});

test('the atlas can be turned and reset without selecting a lesson', async ({ page }) => {
	await page.goto('/');
	await ready(page);
	await page.getByRole('button', { name: 'New to AI' }).click();
	const label = atlas(page).getByRole('button', { name: 'Preview Train a network', exact: true });
	await page.waitForTimeout(1100);
	const before = (await label.boundingBox())!;
	const canvas = (await atlas(page).locator('canvas').boundingBox())!;
	const x = canvas.x + canvas.width * .5;
	const y = canvas.y + canvas.height * .45;
	await page.mouse.move(x, y);
	await page.mouse.down();
	await page.mouse.move(x + 55, y + 24, { steps: 8 });
	await page.mouse.up();
	await expect.poll(async () => Math.abs((await label.boundingBox())!.x - before.x)).toBeGreaterThan(4);
	await expect(preview(page).getByRole('heading')).toHaveText('The neuron');
	await page.getByRole('button', { name: 'Reset camera', exact: true }).click();
	await expect.poll(async () => Math.abs((await label.boundingBox())!.x - before.x)).toBeLessThan(1);
});

test('the mobile atlas is usable with reduced motion and keyboard controls', async ({ page }) => {
	await page.setViewportSize({ width: 390, height: 844 });
	await page.emulateMedia({ reducedMotion: 'reduce' });
	await page.goto('/');
	await ready(page);
	await expect(page.locator('html')).toHaveAttribute('data-reduce-motion', '');
	await page.getByRole('button', { name: 'New to AI' }).focus();
	await page.keyboard.press('Enter');
	await expect(preview(page).getByRole('heading')).toBeFocused();
	await expect(preview(page).getByRole('heading')).toHaveText('The neuron');
	const target = atlas(page).getByRole('button', { name: 'Preview Gradient descent', exact: true });
	await target.focus();
	await page.keyboard.press('Enter');
	await expect(preview(page).getByRole('heading')).toHaveText('Gradient descent');
	await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBe(0);
	const box = (await atlas(page).boundingBox())!;
	for (const button of await atlas(page).getByRole('button').all()) {
		const bounds = (await button.boundingBox())!;
		expect(bounds.x).toBeGreaterThanOrEqual(box.x);
		expect(bounds.x + bounds.width).toBeLessThanOrEqual(box.x + box.width + 1);
	}
	const canvas = atlas(page).locator('canvas');
	await page.evaluate(() => document.fonts.ready);
	await canvas.scrollIntoViewIfNeeded();
	// Camera selection is applied on the next GPU frame; wait for that frame after scrolling.
	await page.evaluate(() => new Promise<void>((done) => requestAnimationFrame(() => requestAnimationFrame(() => done()))));
	await page.waitForTimeout(200);
	const frame = await canvas.screenshot();
	await page.waitForTimeout(150);
	const later = await canvas.screenshot();
	const maxChange = await page.evaluate(async ([before, after]) => {
		async function pixels(encoded: string) {
			const blob = await (await fetch(`data:image/png;base64,${encoded}`)).blob();
			const bitmap = await createImageBitmap(blob);
			const image = new OffscreenCanvas(bitmap.width, bitmap.height);
			const context = image.getContext('2d')!;
			context.drawImage(bitmap, 0, 0);
			bitmap.close();
			return context.getImageData(0, 0, image.width, image.height).data;
		}
		const [a, b] = await Promise.all([pixels(before), pixels(after)]);
		if (a.length !== b.length) return 255;
		let largest = 0;
		for (let i = 0; i < a.length; i++) largest = Math.max(largest, Math.abs(a[i] - b[i]));
		return largest;
	}, [frame.toString('base64'), later.toString('base64')]);
	// GPU compositing can round a few channels by one level even when geometry is stationary.
	expect(maxChange, 'Reduced-motion scene must remain still').toBeLessThanOrEqual(2);
});

test('lesson navigation remains available when WebGPU is unavailable', async ({ page }) => {
	await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { get: () => undefined }));
	await page.goto('/');
	await expect(page.locator('.scene-wrap .stage')).toHaveAttribute('data-status', 'nogpu');
	await expect(page.getByText('This visual needs WebGPU')).toBeVisible();
	await page.getByRole('navigation', { name: 'Atlas regions' }).getByRole('button', { name: /Neural networks/ }).click();
	await preview(page).getByRole('button', { name: /The neuron/ }).click();
	await expect(preview(page).getByRole('link', { name: 'Start lesson' })).toHaveAttribute('href', lessonHref(fundamentals[0]));
	await expect(page.locator('a.star')).toHaveCount(23);
});

for (const width of [390, 1024]) {
	test(`every region keeps its lesson markers readable at ${width}px`, async ({ page }) => {
		await page.setViewportSize({ width, height: 900 });
		await page.emulateMedia({ reducedMotion: 'reduce' });
		await page.goto('/');
		await ready(page);
		await page.evaluate(() => document.fonts.ready);
		for (const [index, track] of tracks.entries()) {
			await page.getByRole('navigation', { name: 'Atlas regions' }).getByRole('button').nth(index).click();
			const buttons = atlas(page).getByRole('button', { name: /^Preview / });
			await expect(buttons).toHaveCount(track.lessons.length);
			await atlas(page).scrollIntoViewIfNeeded();
			await page.evaluate(() => new Promise<void>((done) => requestAnimationFrame(() => requestAnimationFrame(() => done()))));
			const stage = (await atlas(page).boundingBox())!;
			const boxes = await buttons.evaluateAll((items) => items.map((item) => {
				const r = item.getBoundingClientRect();
				return { x: r.x, y: r.y, right: r.right, bottom: r.bottom, label: item.textContent };
			}));
			for (const [i, box] of boxes.entries()) {
				expect(box.x, box.label ?? '').toBeGreaterThanOrEqual(stage.x);
				expect(box.right, box.label ?? '').toBeLessThanOrEqual(stage.x + stage.width + 1);
				expect(box.y).toBeGreaterThanOrEqual(stage.y);
				expect(box.bottom).toBeLessThanOrEqual(stage.y + stage.height);
				for (const other of boxes.slice(i + 1)) {
					const overlap = Math.min(box.right, other.right) > Math.max(box.x, other.x) && Math.min(box.bottom, other.bottom) > Math.max(box.y, other.y);
					expect(overlap, `${box.label} overlaps ${other.label}`).toBe(false);
				}
			}
			await buttons.last().focus();
			await page.keyboard.press('Enter');
			await expect(preview(page).getByRole('heading')).toHaveText(track.lessons.at(-1)!.title);
		}
	});
}

test('completion belongs to the right lesson when switching regions', async ({ page }) => {
	await page.addInitScript(() => localStorage.setItem('neural-atlas:progress:v1', JSON.stringify({
		done: { 'classic-ml/decision-boundaries': true, 'llm/attention': true }, step: {}, solved: {}
	})));
	await page.goto('/');
	await ready(page);
	for (const [index, track] of tracks.entries()) {
		await page.getByRole('navigation', { name: 'Atlas regions' }).getByRole('button').nth(index).click();
		await atlas(page).getByRole('button', { name: `Preview ${track.lessons[1].title}`, exact: true }).click();
		await expect(preview(page).getByText('Completed · revisit anytime')).toHaveCount(index === 1 ? 1 : 0);
	}
	await page.getByRole('navigation', { name: 'Atlas regions' }).getByRole('button').nth(2).click();
	await atlas(page).getByRole('button', { name: 'Preview Self-attention', exact: true }).click();
	await expect(preview(page).getByText('Completed · revisit anytime')).toBeVisible();
	await expect(preview(page).getByRole('link', { name: 'Embedding space' })).toHaveAttribute('href', '/learn/llm/embeddings');
});

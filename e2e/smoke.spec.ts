import { expect, test, type Page } from '@playwright/test';
import { allLessons, lessonHref } from '../src/lib/content/curriculum.ts';
import { questions } from '../src/lib/lessons/fundamentals/neuron/questions.ts';

/**
 * Smoke test: every page loads without console errors, and every lesson's visual starts, draws
 * something and survives a walk through all of its steps.
 *
 * Where the test browser has no WebGPU the visuals cannot start. The lesson tests then check that the
 * "needs WebGPU" notice is shown instead, and say so in an annotation.
 */

/** Collects console errors and uncaught exceptions from the page. */
function watchErrors(page: Page): string[] {
	const errors: string[] = [];
	page.on('console', (m) => {
		if (m.type() === 'error') errors.push(m.text());
	});
	page.on('pageerror', (e) => errors.push(String(e)));
	// Name the file behind a "Failed to load resource" console message.
	page.on('response', (r) => {
		if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`);
	});
	return errors;
}

/** Waits until every stage on the page has left 'loading', then returns their statuses. */
async function stageStatuses(page: Page): Promise<string[]> {
	const stages = page.locator('.stage[data-status]');
	await expect(stages.first()).toBeVisible();
	await expect(page.locator('.stage[data-status="loading"]')).toHaveCount(0);
	return stages.evaluateAll((els) => els.map((el) => el.getAttribute('data-status') ?? ''));
}

/** Share of the lesson canvas that is clearly brighter than the background, 0..1. */
async function litFraction(page: Page): Promise<number> {
	// Hide everything in the stage cell except the canvas, so HTML labels and controls do not count.
	const style = await page.addStyleTag({
		content: '#lesson-stage *:not(canvas):not(:has(canvas)) { visibility: hidden !important; }'
	});
	const png = (await page.locator('#lesson-stage canvas').screenshot()).toString('base64');
	await style.evaluate((el) => el.remove());
	return page.evaluate(async (b64) => {
		const blob = await (await fetch(`data:image/png;base64,${b64}`)).blob();
		const bitmap = await createImageBitmap(blob);
		const c = new OffscreenCanvas(bitmap.width, bitmap.height);
		const g = c.getContext('2d')!;
		g.drawImage(bitmap, 0, 0);
		const d = g.getImageData(0, 0, c.width, c.height).data;
		let lit = 0;
		for (let i = 0; i < d.length; i += 4) if (Math.max(d[i], d[i + 1], d[i + 2]) > 48) lit++;
		return lit / (d.length / 4);
	}, png);
}

const PAGES: [path: string, heading: string][] = [
	['/', 'See how machines learn'],
	['/learn', 'Lessons'],
	['/glossary', 'Glossary']
];

for (const [path, heading] of PAGES) {
	test(`page ${path} loads`, async ({ page }) => {
		const errors = watchErrors(page);
		await page.goto(path);
		await expect(page.getByRole('heading', { level: 1 })).toHaveText(heading);
		if (path === '/') expect(await stageStatuses(page)).not.toContain('error');
		expect(errors).toEqual([]);
	});
}

test('/playground sends you to the training lesson', async ({ page }) => {
	await page.goto('/playground');
	await expect(page).toHaveURL(/\/learn\/fundamentals\/train-a-network$/);
});

test('the build has a page for every address, so a link straight to one answers 200', async ({
	request
}) => {
	const paths = [...PAGES.map(([p]) => p), '/playground'];
	for (const meta of allLessons) if (meta.status === 'ready') paths.push(lessonHref(meta));
	for (const p of paths) expect((await request.get(p)).status(), p).toBe(200);
	expect((await request.get('/learn/no-such-track/no-such-lesson')).status()).toBe(404);
});

test('the home map links to every lesson', async ({ page }) => {
	await page.goto('/');
	for (const meta of allLessons) {
		await expect(page.locator(`a.star[href="${lessonHref(meta)}"]`)).toHaveCount(1);
	}
});

for (const meta of allLessons) {
	test(`lesson ${meta.track}/${meta.id}`, async ({ page }) => {
		const errors = watchErrors(page);
		await page.goto(lessonHref(meta));
		await expect(page.getByRole('heading', { level: 1 })).toHaveText(meta.title);

		const [status] = await stageStatuses(page);
		if (status === 'nogpu') {
			test.info().annotations.push({ type: 'no WebGPU', description: 'visual not checked' });
			await expect(page.getByText('This visual needs WebGPU')).toBeVisible();
			// The only expected console output is the engine saying WebGPU is missing.
			expect(errors.filter((e) => !/webgpu/i.test(e))).toEqual([]);
			return;
		}
		expect(status).toBe('ready');

		// The first step draws something.
		await expect.poll(() => litFraction(page), { timeout: 15_000 }).toBeGreaterThan(0.002);

		// Walk to the last page (steps, then questions) with the keyboard; the Next button is replaced by a link there.
		const next = page.locator('button[aria-keyshortcuts="ArrowRight"]');
		const rail = page.locator('.rail .label');
		for (let i = 0; i < 20 && (await next.count()) > 0; i++) {
			const before = await rail.textContent();
			await page.keyboard.press('ArrowRight');
			await expect(rail).not.toHaveText(before ?? '');
		}
		await expect(next).toHaveCount(0);
		await expect(page.locator('.stage[data-status]').first()).toHaveAttribute('data-status', 'ready');
		await expect.poll(() => litFraction(page), { timeout: 15_000 }).toBeGreaterThan(0.002);

		expect(errors).toEqual([]);
	});
}

test('questions can be answered and are remembered', async ({ page }) => {
	const errors = watchErrors(page);
	await page.goto('/learn/fundamentals/neuron');
	await expect(page.getByRole('heading', { level: 1 })).toHaveText('The neuron');

	const label = page.locator('.rail .label');
	const card = page.locator('.question');
	const [q1, , q3] = questions;
	if (q1.kind !== 'choice' || q3.kind !== 'task') throw new Error('unexpected question kinds');

	await page.getByRole('button', { name: 'Go to question 1' }).click();
	await expect(label).toHaveText('Question 1 of 3');
	await expect(card).toHaveAttribute('data-solved', 'false');

	const wrongIndex = q1.options.findIndex((o) => !o.correct);
	const rightIndex = q1.options.findIndex((o) => o.correct);
	await card.locator(`[data-option="${wrongIndex}"]`).click();
	await expect(page.getByText(q1.options[wrongIndex].why)).toBeVisible();
	await expect(card).toHaveAttribute('data-solved', 'false');

	await card.locator(`[data-option="${rightIndex}"]`).click();
	await expect(card).toHaveAttribute('data-solved', 'true');
	await expect(page.getByText(q1.options[rightIndex].why)).toBeVisible();

	await page.reload();
	await expect(label).toHaveText('Question 1 of 3');
	await expect(card).toHaveAttribute('data-solved', 'true');

	// The task: set the bias with the only visible slider. No WebGPU is needed for this.
	await page.getByRole('button', { name: 'Go to question 3' }).click();
	await expect(label).toHaveText('Question 3 of 3');
	await expect(card).toHaveAttribute('data-solved', 'false');
	const slider = page.getByRole('slider');
	// fill() sends no pointer or key event, so touch the slider first, as a learner would.
	await slider.click();
	await slider.fill(String(q3.example?.params?.b));
	await expect(card).toHaveAttribute('data-solved', 'true');
	await expect(page.getByText(q3.solved.slice(0, 30))).toBeVisible();

	// The two solved questions show on the lesson list and on the home map.
	const solvedText = '2 of 3 questions solved';
	const row = 'a[href="/learn/fundamentals/neuron"]';
	await page.goto('/learn');
	await expect(page.locator(`${row} .pips`)).toHaveAttribute('aria-label', solvedText);
	await expect(page.locator(`${row} .pip.solved`)).toHaveCount(2);
	await expect(page.getByText('2 of 21 questions solved')).toBeVisible();
	await page.goto('/');
	await expect(page.locator(`a.star[href="/learn/fundamentals/neuron"]`)).toHaveAttribute(
		'aria-label',
		new RegExp(solvedText)
	);
	await expect(page.getByText('2 of 60 questions solved')).toBeVisible();

	expect(errors.filter((e) => !/webgpu/i.test(e))).toEqual([]);
});

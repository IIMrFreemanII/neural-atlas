import { expect, test, type Page } from '@playwright/test';
import { allLessons, lessonHref, questionTotal, tracks } from '../src/lib/content/curriculum.ts';
import { STEPS } from '../src/lib/content/steps-index.ts';
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

/**
 * Share of the lesson canvas that differs clearly from the stage background, 0..1.
 *
 * The canvas is transparent over a paper page (dark ink on light paper, or the reverse in Night), so "lit"
 * means "unlike the background", not "bright". The background is the most common colour once each channel is
 * cut to 16 levels (the mean of that bucket); a pixel counts when any channel is more than 40 away from it,
 * which the faint paper grid (alpha 0.09 at most) never reaches in either theme.
 */
async function litFraction(page: Page): Promise<number> {
	// Hide everything on the page except the canvas and the paper under it, so HTML labels, the narration
	// card and the controls do not count.
	const style = await page.addStyleTag({
		content:
			'#lesson-stage *:not(canvas):not(:has(canvas)) { visibility: hidden !important; }' +
			'.area > :not(.paper):not(.grid):not(#lesson-stage) { visibility: hidden !important; }'
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
		const n = d.length / 4;
		// The background: the mean colour of the most populated 16-level bucket.
		const count = new Map<number, number>();
		for (let i = 0; i < d.length; i += 4) {
			const k = ((d[i] >> 4) << 8) | ((d[i + 1] >> 4) << 4) | (d[i + 2] >> 4);
			count.set(k, (count.get(k) ?? 0) + 1);
		}
		let best = 0;
		let bestN = -1;
		for (const [k, v] of count) {
			if (v > bestN) {
				best = k;
				bestN = v;
			}
		}
		let sr = 0;
		let sg = 0;
		let sb = 0;
		for (let i = 0; i < d.length; i += 4) {
			const k = ((d[i] >> 4) << 8) | ((d[i + 1] >> 4) << 4) | (d[i + 2] >> 4);
			if (k !== best) continue;
			sr += d[i];
			sg += d[i + 1];
			sb += d[i + 2];
		}
		const bg = [sr / bestN, sg / bestN, sb / bestN];
		let lit = 0;
		for (let i = 0; i < d.length; i += 4) {
			if (Math.max(Math.abs(d[i] - bg[0]), Math.abs(d[i + 1] - bg[1]), Math.abs(d[i + 2] - bg[2])) > 40) lit++;
		}
		return lit / n;
	}, png);
}

/**
 * Least lit share that counts as "the scene drew something". A blank canvas measures 0 (the paper grain stays
 * under the colour threshold); a sparse first step, such as the pale token boxes of llm/attention, measures
 * about 0.0018, so the bar sits below that.
 */
const LIT_MIN = 0.001;

/** Opens the narration card (and so the step rail) before the page loads, whatever an earlier run left behind. */
async function unfoldCard(page: Page): Promise<void> {
	await page.addInitScript(() => {
		try {
			localStorage.setItem('neural-atlas:card-folded:v1', '0');
		} catch {
			/* storage unavailable */
		}
	});
}

const PAGES: [path: string, heading: string][] = [
	['/', 'See how machines learn'],
	['/learn', 'Lessons'],
	['/glossary', 'Glossary'],
	['/playground', 'Playground']
];

for (const [path, heading] of PAGES) {
	test(`page ${path} loads`, async ({ page }) => {
		const errors = watchErrors(page);
		await page.goto(path);
		await expect(page.getByRole('heading', { level: 1 })).toHaveText(heading);
		// The home page is an SVG map with no stage; the others that have one must not have failed.
		if (path === '/') expect(await page.locator('a.star').count()).toBeGreaterThan(0);
		else if (path === '/playground') expect(await stageStatuses(page)).not.toContain('error');
		expect(errors).toEqual([]);
	});
}

test('the playground trains a network', async ({ page }) => {
	const errors = watchErrors(page);
	await page.goto('/playground');
	await expect(page).toHaveTitle('Playground · Neural Atlas');
	await expect(page.getByRole('heading', { level: 1 })).toHaveText('Playground');

	const [status] = await stageStatuses(page);
	if (status === 'nogpu') {
		test.info().annotations.push({ type: 'no WebGPU', description: 'visual not checked' });
		await expect(page.getByText('This visual needs WebGPU')).toBeVisible();
		expect(errors.filter((e) => !/webgpu/i.test(e))).toEqual([]);
		return;
	}
	expect(status).toBe('ready');

	const steps = page.locator('.readouts li', { hasText: 'Steps' }).locator('.value');
	const stepCount = async () => Number((await steps.textContent()) ?? 0) || 0;
	await page.getByRole('button', { name: 'Train', exact: true }).click();
	await expect(page.getByRole('button', { name: 'Pause', exact: true })).toBeVisible();
	await expect.poll(stepCount, { timeout: 15_000 }).toBeGreaterThan(0);
	await page.getByRole('button', { name: 'Pause', exact: true }).click();
	await expect(page.getByRole('button', { name: 'Train', exact: true })).toBeVisible();

	expect(errors).toEqual([]);
});

test('the build has a page for every address, so a link straight to one answers 200', async ({
	request
}) => {
	const paths = PAGES.map(([p]) => p);
	for (const meta of allLessons) if (meta.status === 'ready') paths.push(lessonHref(meta));
	for (const p of paths) expect((await request.get(p)).status(), p).toBe(200);
	expect((await request.get('/learn/no-such-track/no-such-lesson')).status()).toBe(404);
});

test('each page names itself in its HTML, for link previews', async ({ request }) => {
	for (const meta of allLessons.filter((l) => l.status === 'ready')) {
		const html = await (await request.get(lessonHref(meta))).text();
		const head = `${meta.track}/${meta.id}`;
		expect(html, head).toContain(`<title>${meta.title} · Neural Atlas</title>`);
		expect(html, head).toContain(`<meta property="og:title" content="${meta.title} · Neural Atlas" />`);
		// Its link-preview image (scripts/capture-previews.ts takes them; rerun it after adding a lesson).
		const image = await request.get(`/previews/${meta.track}/${meta.id}.jpg`);
		expect(image.status(), `${head} preview image`).toBe(200);
	}
	expect((await request.get('/previews/home.jpg')).status()).toBe(200);
	expect(await (await request.get('/glossary')).text()).toContain('<title>Glossary · Neural Atlas</title>');
});

test('the tab title follows navigation in the app', async ({ page }) => {
	await page.goto('/learn/fundamentals/neuron');
	await expect(page).toHaveTitle('The neuron · Neural Atlas');
	await page.getByRole('link', { name: 'Learn', exact: true }).click();
	await expect(page).toHaveTitle('Lessons · Neural Atlas');
	await page.locator('a.brand').click();
	await expect(page).toHaveTitle('Neural Atlas');
	expect(await page.locator('head title').count()).toBe(1);
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
		await unfoldCard(page);
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
		await expect.poll(() => litFraction(page), { timeout: 15_000 }).toBeGreaterThan(LIT_MIN);

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
		await expect.poll(() => litFraction(page), { timeout: 15_000 }).toBeGreaterThan(LIT_MIN);

		expect(errors).toEqual([]);
	});
}

test('questions can be answered and are remembered', async ({ page }) => {
	const errors = watchErrors(page);
	await unfoldCard(page);
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

	// The two solved questions show on the lesson list (with its track's total) and on the home map.
	const solvedText = '2 of 3 questions solved';
	const row = 'a[href="/learn/fundamentals/neuron"]';
	await page.goto('/learn');
	await expect(page.locator(`${row} .pips`)).toHaveAttribute('aria-label', solvedText);
	await expect(page.locator(`${row} .pip.solved`)).toHaveCount(2);
	const fundamentals = tracks.find((t) => t.id === 'fundamentals')!;
	await expect(page.getByText(`2 of ${questionTotal(fundamentals.lessons)} questions solved`)).toBeVisible();
	await page.goto('/');
	await expect(page.locator(`a.star[href="/learn/fundamentals/neuron"]`)).toHaveAttribute(
		'aria-label',
		new RegExp(solvedText)
	);
	await expect(page.getByText(`2 of ${questionTotal(allLessons)} questions solved`)).toBeVisible();

	expect(errors.filter((e) => !/webgpu/i.test(e))).toEqual([]);
});

// The shell of the Paper Atlas redesign: themes, settings, the Contents drawer, quick jump, the camera chip.

/** Stores settings before the page loads (and on every load after, so use it only where the start is all that matters). */
async function storeSettings(page: Page, settings: { theme?: 'paper' | 'night'; reducedMotion?: boolean }): Promise<void> {
	await page.addInitScript((s) => {
		try {
			localStorage.setItem('neural-atlas:settings:v1', JSON.stringify(s));
		} catch {
			/* storage unavailable */
		}
	}, settings);
}

const html = (page: Page) => page.locator('html');
const themeSwitch = (page: Page) => page.getByRole('group', { name: 'Theme' });

test('the theme switch works, persists, and can follow the system', async ({ page }) => {
	await page.emulateMedia({ colorScheme: 'light' });
	await page.goto('/learn');
	await expect(html(page)).toHaveAttribute('data-theme', 'paper');

	await themeSwitch(page).getByRole('button', { name: 'Night' }).click();
	await expect(html(page)).toHaveAttribute('data-theme', 'night');
	await page.reload();
	await expect(html(page)).toHaveAttribute('data-theme', 'night');

	// Following the system again: the theme comes from the colour scheme the browser reports.
	await page.getByRole('button', { name: 'Settings' }).click();
	await page.getByRole('checkbox', { name: 'Follow system theme' }).check();
	await page.emulateMedia({ colorScheme: 'dark' });
	await expect(html(page)).toHaveAttribute('data-theme', 'night');
	await page.emulateMedia({ colorScheme: 'light' });
	await expect(html(page)).toHaveAttribute('data-theme', 'paper');
});

test('a lesson draws in the Night theme', async ({ page }) => {
	const errors = watchErrors(page);
	await page.emulateMedia({ colorScheme: 'light' });
	await storeSettings(page, { theme: 'night' });
	await unfoldCard(page);
	await page.goto('/learn/fundamentals/neuron');
	await expect(html(page)).toHaveAttribute('data-theme', 'night');
	const [status] = await stageStatuses(page);
	if (status === 'nogpu') {
		test.info().annotations.push({ type: 'no WebGPU', description: 'visual not checked' });
		return;
	}
	expect(status).toBe('ready');
	await expect.poll(() => litFraction(page), { timeout: 15_000 }).toBeGreaterThan(LIT_MIN);
	expect(errors).toEqual([]);
});

test('the Contents drawer lists the lessons, opens one, and closes with Escape', async ({ page }) => {
	await page.goto('/learn/fundamentals/neuron');
	const drawer = page.locator('#contents-drawer');
	const kMeans = allLessons.find((l) => l.track === 'classic-ml' && l.id === 'k-means')!;
	const classic = tracks.find((t) => t.id === 'classic-ml')!;

	await page.getByRole('button', { name: 'Contents' }).click();
	await expect(drawer).toBeVisible();
	// Only the current lesson's track starts open; open another to see its lessons.
	await drawer.getByRole('button', { name: new RegExp(classic.title) }).click();
	await expect(drawer.locator(`a[href="${lessonHref(kMeans)}"]`)).toBeVisible();

	await drawer.locator(`a[href="${lessonHref(kMeans)}"]`).click();
	await expect(page).toHaveURL(new RegExp(`${lessonHref(kMeans)}$`));
	await expect(page.getByRole('heading', { level: 1 })).toHaveText(kMeans.title);
	await expect(drawer).toHaveCount(0);

	await page.getByRole('button', { name: 'Contents' }).click();
	await expect(drawer).toBeVisible();
	await page.keyboard.press('Escape');
	await expect(drawer).toHaveCount(0);
});

test('quick jump takes you to a step of a lesson', async ({ page }) => {
	await unfoldCard(page);
	await page.goto('/learn');
	// The shortcut is a window listener, so wait for the app to start before pressing it.
	await page.waitForLoadState('networkidle');
	const step = STEPS.find((s) => s.track === 'fundamentals' && s.lesson === 'neuron' && s.step === 2)!;

	await page.keyboard.press('ControlOrMeta+K');
	await expect(page.getByRole('dialog', { name: 'Quick jump' })).toBeVisible();
	await page.getByRole('combobox').fill(step.title);
	await expect(page.getByRole('option', { name: new RegExp(step.title) }).first()).toBeVisible();
	await page.keyboard.press('Enter');

	await expect(page).toHaveURL(/\/learn\/fundamentals\/neuron\?step=3$/);
	await expect(page.locator('.rail .label')).toHaveText('Step 3 of 7');
});

test('turning a 3D figure shows a chip that takes you back', async ({ page }) => {
	await unfoldCard(page);
	// Step 4 is the first with a 3D figure to turn; the earlier ones are flat.
	await page.goto('/learn/fundamentals/loss-landscape?step=4');
	const [status] = await stageStatuses(page);
	if (status === 'nogpu') {
		test.info().annotations.push({ type: 'no WebGPU', description: 'visual not checked' });
		return;
	}
	expect(status).toBe('ready');

	await expect(page.getByText('Drag to turn the figure')).toBeVisible();
	const chip = page.getByRole('button', { name: "Back to this step's view" });
	await expect(chip).toHaveCount(0);
	const box = (await page.locator('#lesson-stage canvas').boundingBox())!;
	// The figure sits in the right half of the stage (the left half is the flat view), clear of the card.
	const x = box.x + box.width * 0.58;
	const y = box.y + box.height * 0.5;
	await page.mouse.move(x, y);
	await page.mouse.down();
	for (let i = 1; i <= 8; i++) await page.mouse.move(x + i * 8, y + i * 10);
	await page.mouse.up();
	await expect(chip).toBeVisible();

	await chip.click();
	await expect(chip).toHaveCount(0);
});

test('reduced motion can be turned on, is remembered, and a lesson still starts', async ({ page }) => {
	await page.emulateMedia({ reducedMotion: 'no-preference' });
	await page.goto('/learn');
	expect(await html(page).getAttribute('data-reduce-motion')).toBeNull();

	await page.getByRole('button', { name: 'Settings' }).click();
	await page.getByRole('switch', { name: 'Reduce motion' }).click();
	await expect(html(page)).toHaveAttribute('data-reduce-motion', '');
	await page.reload();
	await expect(html(page)).toHaveAttribute('data-reduce-motion', '');

	await page.goto('/learn/fundamentals/neuron');
	await expect(html(page)).toHaveAttribute('data-reduce-motion', '');
	const [status] = await stageStatuses(page);
	if (status === 'nogpu') {
		test.info().annotations.push({ type: 'no WebGPU', description: 'visual not checked' });
		return;
	}
	expect(status).toBe('ready');
});

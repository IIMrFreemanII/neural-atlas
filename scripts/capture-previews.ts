/**
 * Takes the link-preview images: a 1200x630 screenshot of every ready lesson and of the home page, written to
 * static/previews/. Run it from the project root after `npm run build`, with Chrome installed (the scenes need
 * WebGPU):
 *
 *   node --experimental-strip-types scripts/capture-previews.ts
 *
 * Optional: a lesson key to take only that one, e.g. `llm/sampling`. Rerun it when a lesson's look changes.
 */
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { chromium } from '@playwright/test';
import { allLessons, lessonHref } from '../src/lib/content/curriculum.ts';

const PORT = 4373;
const ROOT = `http://localhost:${PORT}`;
const OUT = 'static/previews';
/** The size link previews expect. */
const SIZE = { width: 1200, height: 630 };
/**
 * A lesson's shot is its stage alone, so that the scene fills the preview. The lesson page is a full-bleed
 * stage under a 48px top bar (--topbar-h); the narration card floats over the stage (it is folded and hidden
 * for the shot), so this viewport makes the stage exactly SIZE.
 */
const TOPBAR_H = 48;
const LESSON_VIEWPORT = { width: SIZE.width, height: SIZE.height + TOPBAR_H };
/**
 * What a lesson shot starts from: the card folded, the Paper theme, and reduced motion so that no glide or
 * draw-in is still running when the shot is taken.
 */
const LESSON_STORAGE = {
	'neural-atlas:card-folded:v1': '1',
	'neural-atlas:settings:v1': JSON.stringify({ theme: 'paper', reducedMotion: true })
};
/**
 * Hides everything on the lesson page that floats over the picture: the legend, readouts, chips and control bar
 * (the stage cell's children other than the stage itself) and the folded tab (the area's children other than the
 * paper, its grid and the stage cell). The scene's own labels live inside .stage and stay.
 */
const LESSON_HIDE =
	'#lesson-stage > :not(.stage), .area > :not(.paper):not(.grid):not(#lesson-stage) { visibility: hidden !important; }';
/** The home page is shot whole, laid out at 1600x840 and scaled down to SIZE. */
const HOME_VIEWPORT = { width: 1600, height: 840 };
/** How long a scene runs before the shot, so that it has drawn and settled. */
const SETTLE_MS = 2500;

/**
 * The step to show, for lessons whose first step is not the best picture of them (1 = the first step).
 * Steps are reached with the right arrow key, as a learner would.
 */
const STEP: Record<string, number> = {
	'fundamentals/neuron': 6,
	'fundamentals/activations': 7,
	'fundamentals/loss-landscape': 7,
	'fundamentals/gradient-descent': 6,
	'classic-ml/linear-regression': 6,
	'classic-ml/k-means': 7,
	'classic-ml/decision-trees': 6,
	'llm/tokenization': 6,
	'llm/embeddings': 2,
	'llm/attention': 8,
	'llm/training': 6,
	'vision/vit': 5,
	'vision/autoencoder': 7,
	'vision/diffusion': 2
};

async function waitForServer(): Promise<void> {
	for (let i = 0; i < 100; i++) {
		try {
			if ((await fetch(ROOT)).ok) return;
		} catch {
			// Not up yet.
		}
		await new Promise((r) => setTimeout(r, 200));
	}
	throw new Error(`The preview server did not start on port ${PORT}`);
}

const only = process.argv[2];
const server = spawn('npx', ['vite', 'preview', '--port', String(PORT), '--strictPort'], { stdio: 'ignore' });
try {
	await waitForServer();
	const browser = await chromium.launch({ channel: 'chrome', args: ['--enable-unsafe-webgpu'] });
	// The home shot is Paper too, whatever the colour scheme of the machine running this.
	const homeContext = await browser.newContext({
		viewport: HOME_VIEWPORT,
		deviceScaleFactor: SIZE.width / HOME_VIEWPORT.width
	});
	await homeContext.addInitScript((items) => {
		for (const [k, v] of Object.entries(items)) localStorage.setItem(k, v);
	}, { 'neural-atlas:settings:v1': LESSON_STORAGE['neural-atlas:settings:v1'] });
	const home = await homeContext.newPage();
	const lessonContext = await browser.newContext({ viewport: LESSON_VIEWPORT });
	await lessonContext.addInitScript((items) => {
		for (const [k, v] of Object.entries(items)) localStorage.setItem(k, v);
	}, LESSON_STORAGE);
	const lesson = await lessonContext.newPage();
	const shots: [key: string, path: string][] = [['home', '/']];
	for (const meta of allLessons) {
		if (meta.status === 'ready') shots.push([`${meta.track}/${meta.id}`, lessonHref(meta)]);
	}
	for (const [key, path] of shots) {
		if (only && key !== only) continue;
		const isHome = key === 'home';
		const page = isHome ? home : lesson;
		await page.goto(ROOT + path);
		if (isHome) {
			// The home page is an SVG map with no stage: wait for its lesson links.
			await page.locator('a.star').first().waitFor();
			await page.mouse.move(0, 0);
			await page.waitForTimeout(SETTLE_MS);
		} else {
			await page.addStyleTag({ content: LESSON_HIDE });
		}
		const stage = page.locator('.stage[data-status]').first();
		if (!isHome) {
			await stage.waitFor();
			await page.waitForFunction(
				() => document.querySelector('.stage[data-status]')?.getAttribute('data-status') !== 'loading'
			);
			for (let s = 1; s < (STEP[key] ?? 1); s++) await page.keyboard.press('ArrowRight');
			await page.mouse.move(0, 0);
			await page.waitForTimeout(SETTLE_MS);
			const now = await stage.getAttribute('data-status');
			if (now !== 'ready') throw new Error(`${key}: the scene is "${now}", not ready`);
		}
		const file = `${OUT}/${key}.jpg`;
		mkdirSync(file.slice(0, file.lastIndexOf('/')), { recursive: true });
		let clip;
		if (!isHome) {
			const box = await stage.boundingBox();
			if (!box || Math.round(box.width) !== SIZE.width || Math.round(box.height) !== SIZE.height) {
				throw new Error(`${key}: the stage is ${box?.width}x${box?.height}, not ${SIZE.width}x${SIZE.height}`);
			}
			clip = { x: box.x, y: box.y, ...SIZE };
		}
		await page.screenshot({ path: file, type: 'jpeg', quality: 85, clip });
		console.log(`wrote ${file}`);
	}
	await browser.close();
} finally {
	server.kill();
}

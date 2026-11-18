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
 * A lesson's shot is its stage alone, so that the scene fills the preview. The lesson page puts 600px of side
 * panels beside the stage and its 52px top bar above it, so this viewport makes the stage exactly SIZE.
 */
const LESSON_VIEWPORT = { width: SIZE.width + 600, height: SIZE.height + 52 };
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
	'classic-ml/linear-regression': 6,
	'classic-ml/k-means': 7,
	'llm/embeddings': 2,
	'llm/attention': 8,
	'llm/training': 6,
	'vision/autoencoder': 7
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
	const home = await browser.newPage({ viewport: HOME_VIEWPORT, deviceScaleFactor: SIZE.width / HOME_VIEWPORT.width });
	const lesson = await browser.newPage({ viewport: LESSON_VIEWPORT });
	const shots: [key: string, path: string][] = [['home', '/']];
	for (const meta of allLessons) {
		if (meta.status === 'ready') shots.push([`${meta.track}/${meta.id}`, lessonHref(meta)]);
	}
	for (const [key, path] of shots) {
		if (only && key !== only) continue;
		const page = key === 'home' ? home : lesson;
		await page.goto(ROOT + path);
		const stage = page.locator('.stage[data-status]').first();
		await stage.waitFor();
		const status = await stage.getAttribute('data-status');
		await page.waitForFunction(
			() => document.querySelector('.stage[data-status]')?.getAttribute('data-status') !== 'loading'
		);
		for (let s = 1; s < (STEP[key] ?? 1); s++) await page.keyboard.press('ArrowRight');
		await page.mouse.move(0, 0);
		await page.waitForTimeout(SETTLE_MS);
		const now = await stage.getAttribute('data-status');
		if (now !== 'ready') throw new Error(`${key}: the scene is "${now}" (was "${status}"), not ready`);
		const file = `${OUT}/${key}.jpg`;
		mkdirSync(file.slice(0, file.lastIndexOf('/')), { recursive: true });
		let clip;
		if (page === lesson) {
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

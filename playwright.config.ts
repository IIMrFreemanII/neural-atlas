import { defineConfig } from '@playwright/test';

// Not Vite's default preview port (4173), so a server already on that port is never mistaken for this site.
const PORT = 4273;

/**
 * Smoke tests run against the built site. They use the Chrome installed on this machine, because it is
 * the browser most likely to have WebGPU; set PW_CHANNEL=chromium to use Playwright's own browser
 * (after `npx playwright install chromium`).
 */
export default defineConfig({
	testDir: 'e2e',
	fullyParallel: true,
	workers: 4,
	reporter: 'list',
	timeout: 60_000,
	use: {
		baseURL: `http://localhost:${PORT}`,
		viewport: { width: 1280, height: 800 },
		channel: process.env.PW_CHANNEL ?? 'chrome',
		launchOptions: { args: ['--enable-unsafe-webgpu'] }
	},
	webServer: {
		command: `npm run build && npm run preview -- --port ${PORT} --strictPort`,
		port: PORT,
		reuseExistingServer: false,
		timeout: 180_000
	}
});

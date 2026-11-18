import { defineConfig } from 'vitest/config';
import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';

// The path the site is served from. Empty for the root of a domain; `npm run deploy` sets it to the
// repository's name, because GitHub Pages serves a project at https://<user>.github.io/<repository>/.
declare const process: { env: Record<string, string | undefined> };
const base = (process.env.BASE_PATH ?? '') as '' | `/${string}`;
// The site's origin, e.g. https://<user>.github.io, which `npm run deploy` sets. The prerendered pages use it
// for the full addresses that link previews need; without it they have no preview image.
const origin = process.env.SITE_ORIGIN || undefined;

export default defineConfig({
	plugins: [
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			// Every page is prerendered; GitHub Pages answers any other address with 404.html, which is the
			// app, so it shows its own "not found" page.
			adapter: adapter({ fallback: '404.html' }),
			paths: { base, origin }
		})
	],
	test: {
		expect: { requireAssertions: true },
		projects: [
			{
				extends: './vite.config.ts',
				test: {
					name: 'server',
					environment: 'node',
					include: ['src/**/*.{test,spec}.{js,ts}'],
					exclude: ['src/**/*.svelte.{test,spec}.{js,ts}']
				}
			}
		]
	}
});

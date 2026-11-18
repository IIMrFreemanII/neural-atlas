<script lang="ts">
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { findLesson } from '../../content/curriculum.ts';
	import { settings } from '../../state/settings.svelte.ts';
	import { ui } from '../../state/ui.svelte.ts';
	import Icon from './Icon.svelte';
	import SettingsMenu from './SettingsMenu.svelte';

	const links = [
		{ href: resolve('/learn'), label: 'Learn' },
		{ href: resolve('/playground'), label: 'Playground' },
		{ href: resolve('/glossary'), label: 'Glossary' }
	];

	const path = $derived(page.url.pathname);

	function active(href: string): boolean {
		return path === href || path.startsWith(href + '/');
	}

	// On a lesson page the bar also shows the Contents button and where the lesson sits.
	const current = $derived(
		page.route.id === '/learn/[track]/[lesson]'
			? findLesson(page.params.track ?? '', page.params.lesson ?? '')
			: null
	);

	let settingsOpen = $state(false);
	let gear: HTMLButtonElement | undefined = $state();
	let mod = $state('⌘K');

	$effect(() => {
		mod = /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘K' : 'Ctrl K';
	});
</script>

<header class="bar">
	<a class="brand" href={resolve('/')}>
		<svg width="22" height="22" viewBox="0 0 20 20" fill="none" aria-hidden="true">
			<circle cx="2.2" cy="5.4" r="1.3" fill="currentColor" />
			<circle cx="2.2" cy="14.6" r="1.3" fill="currentColor" />
			<path d="M3.4 6.1 7.5 8.6M3.4 13.9 7.5 11.4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" />
			<circle cx="10.2" cy="10" r="3.1" stroke="currentColor" stroke-width="1.6" />
			<path d="M13.3 10H16" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" />
			<circle cx="17.7" cy="10" r="1.7" fill="var(--data)" />
		</svg>
		<span>Neural Atlas</span>
	</a>

	{#if current}
		<button
			type="button"
			class="chip-btn"
			data-contents-toggle
			aria-expanded={ui.contentsOpen}
			aria-controls="contents-drawer"
			onclick={() => ui.toggleContents()}
		>
			<Icon name="menu" size={15} />
			Contents
		</button>
		<div class="crumb">
			<span>{current.track.title}</span>
			<Icon name="chevron-right" size={12} />
			<b>{current.meta.title}</b>
		</div>
	{/if}

	<nav aria-label="Main">
		{#each links as l (l.href)}
			<a href={l.href} class:active={active(l.href)} aria-current={active(l.href) ? 'page' : undefined}>
				{l.label}
			</a>
		{/each}
	</nav>

	<span class="spacer"></span>

	<button
		type="button"
		class="search-pill"
		aria-label="Search"
		aria-haspopup="dialog"
		aria-keyshortcuts="Meta+K Control+K"
		onclick={() => ui.setQuickJumpOpen(true)}
	>
		<Icon name="search" size={14} />
		<span>Search</span>
		<kbd>{mod}</kbd>
	</button>

	<div class="theme-switch" role="group" aria-label="Theme">
		<button type="button" aria-pressed={settings.theme === 'paper'} onclick={() => settings.setTheme('paper')}>
			<Icon name="sun" size={12} />Paper
		</button>
		<button type="button" aria-pressed={settings.theme === 'night'} onclick={() => settings.setTheme('night')}>
			<Icon name="moon" size={12} />Night
		</button>
	</div>

	<button
		type="button"
		class="icon-btn"
		class:open={settingsOpen}
		bind:this={gear}
		aria-label="Settings"
		aria-haspopup="dialog"
		aria-expanded={settingsOpen}
		onclick={() => (settingsOpen = !settingsOpen)}
	>
		<Icon name="gear" size={15} />
	</button>

	<SettingsMenu open={settingsOpen} anchor={gear} onclose={() => (settingsOpen = false)} />
</header>

<style>
	.bar {
		position: sticky;
		top: 0;
		z-index: 30;
		height: var(--topbar-h);
		display: flex;
		align-items: center;
		gap: 18px;
		padding: 0 20px;
		background: color-mix(in srgb, var(--pg) 90%, transparent);
		border-bottom: 1px solid var(--rule);
		font-size: 13px;
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 8px;
		font-family: var(--font-display);
		font-size: 19px;
		font-weight: 500;
		letter-spacing: -0.01em;
		white-space: nowrap;
		color: var(--ink);
	}
	nav {
		display: flex;
		gap: 18px;
	}
	nav a {
		color: var(--ink-2);
		padding-bottom: 2px;
	}
	nav a:hover {
		color: var(--ink);
	}
	nav a.active {
		color: var(--ink);
		box-shadow: inset 0 -2px 0 var(--ink);
	}
	.spacer {
		flex: 1;
	}
	.chip-btn {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		height: 30px;
		padding: 0 12px;
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		background: var(--sheet);
		color: var(--ink);
		font: inherit;
		font-weight: 500;
		cursor: pointer;
	}
	.chip-btn[aria-expanded='true'] {
		border-color: var(--ink);
	}
	.crumb {
		display: flex;
		align-items: center;
		gap: 8px;
		min-width: 0;
		white-space: nowrap;
		color: var(--ink-2);
	}
	.crumb b {
		font-family: var(--font-display);
		font-size: 17px;
		font-weight: 500;
		color: var(--ink);
	}
	.search-pill {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		height: 30px;
		min-width: 168px;
		padding: 0 6px 0 10px;
		border: 1px solid var(--rule);
		border-radius: 4px;
		background: color-mix(in srgb, var(--sheet) 60%, transparent);
		color: var(--ink-3);
		font: inherit;
		cursor: pointer;
	}
	.search-pill:hover {
		border-color: var(--rule-2);
		color: var(--ink-2);
	}
	kbd {
		margin-left: auto;
		font-family: var(--font-mono);
		font-size: 10.5px;
		padding: 1px 5px;
		border: 1px solid var(--rule-2);
		border-radius: 3px;
		background: var(--sheet);
		color: var(--ink-2);
	}
	.theme-switch {
		display: inline-flex;
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		overflow: hidden;
		font-size: 12px;
	}
	.theme-switch button {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 4px 10px;
		border: 0;
		background: transparent;
		color: var(--ink-2);
		font: inherit;
		cursor: pointer;
	}
	.theme-switch button + button {
		border-left: 1px solid var(--rule);
	}
	.theme-switch button[aria-pressed='true'] {
		background: var(--ink);
		color: var(--pg);
	}
	.icon-btn {
		display: inline-grid;
		place-items: center;
		width: 30px;
		height: 30px;
		padding: 0;
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		background: var(--sheet);
		color: var(--ink-2);
		cursor: pointer;
	}
	.icon-btn.open {
		border-color: var(--ink);
		color: var(--ink);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--ink) 10%, transparent);
	}
</style>

<script lang="ts">
	import { page } from '$app/state';

	const links = [
		{ href: '/learn', label: 'Learn' },
		{ href: '/playground', label: 'Playground' },
		{ href: '/glossary', label: 'Glossary' }
	];

	const path = $derived(page.url.pathname);

	function active(href: string): boolean {
		return path === href || path.startsWith(href + '/');
	}
</script>

<header class="bar">
	<div class="left">
		<a class="brand" href="/">
			<svg width="22" height="22" viewBox="0 0 32 32" fill="none" aria-hidden="true">
				<circle cx="16" cy="16" r="11" stroke="var(--c-data)" stroke-width="2.5" />
				<circle cx="16" cy="16" r="3.2" fill="var(--c-data)" />
			</svg>
			<span>Neural Atlas</span>
		</a>
	</div>
	<nav aria-label="Main">
		{#each links as l (l.href)}
			<a href={l.href} class:active={active(l.href)} aria-current={active(l.href) ? 'page' : undefined}>
				{l.label}
			</a>
		{/each}
	</nav>
</header>

<style>
	.bar {
		position: sticky;
		top: 0;
		z-index: 30;
		height: var(--topbar-h);
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0 16px;
		background: var(--bg-panel);
		border-bottom: 1px solid var(--line);
	}
	.left {
		display: flex;
		align-items: center;
		gap: 8px;
		min-width: 0;
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 10px;
		font-family: var(--font-display);
		font-size: 17px;
		font-weight: 500;
		white-space: nowrap;
		border-radius: var(--r-sm);
	}
	nav {
		display: flex;
		gap: 4px;
	}
	nav a {
		padding: 6px 10px;
		border-radius: var(--r-sm);
		font-size: 14px;
		color: var(--c-text-2);
		transition: color 0.15s;
	}
	nav a:hover,
	nav a.active {
		color: var(--c-text);
	}
</style>

<script lang="ts">
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { settings } from '../../state/settings.svelte.ts';

	const links = [
		{ href: resolve('/learn'), label: 'Learn' },
		{ href: resolve('/playground'), label: 'Playground' },
		{ href: resolve('/glossary'), label: 'Glossary' }
	];

	const path = $derived(page.url.pathname);

	function active(href: string): boolean {
		return path === href || path.startsWith(href + '/');
	}
</script>

<header class="bar">
	<div class="left">
		<a class="brand" href={resolve('/')}>
			<svg width="22" height="22" viewBox="0 0 32 32" fill="none" aria-hidden="true">
				<circle cx="16" cy="16" r="11" stroke="var(--c-data)" stroke-width="2.5" />
				<circle cx="16" cy="16" r="3.2" fill="var(--c-data)" />
			</svg>
			<span>Neural Atlas</span>
		</a>
	</div>
	<div class="right">
		<nav aria-label="Main">
			{#each links as l (l.href)}
				<a href={l.href} class:active={active(l.href)} aria-current={active(l.href) ? 'page' : undefined}>
					{l.label}
				</a>
			{/each}
		</nav>
		<div class="motion">
			<span class="lbl" id="motion-label">Reduce motion</span>
			<button
				type="button"
				role="switch"
				class="switch"
				class:on={settings.reducedMotion}
				aria-checked={settings.reducedMotion}
				aria-labelledby="motion-label"
				title="Stops decorative animation and makes transitions instant"
				onclick={() => settings.setReducedMotion(!settings.reducedMotion)}
			><span class="knob"></span></button>
		</div>
	</div>
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
	.right {
		display: flex;
		align-items: center;
		gap: 20px;
	}
	nav {
		display: flex;
		gap: 4px;
	}
	.motion {
		display: flex;
		align-items: center;
		gap: 8px;
		padding-left: 20px;
		border-left: 1px solid var(--line);
	}
	.lbl {
		font-size: 12px;
		color: var(--c-text-2);
		white-space: nowrap;
	}
	.switch {
		position: relative;
		flex: none;
		width: 30px;
		height: 18px;
		padding: 0;
		border: 1px solid var(--line-strong);
		border-radius: var(--r-pill);
		background: transparent;
		cursor: pointer;
		transition:
			background 0.15s,
			border-color 0.15s;
	}
	.knob {
		position: absolute;
		top: 2px;
		left: 2px;
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--c-text-2);
		transition:
			transform 0.15s,
			background 0.15s;
	}
	.switch.on {
		background: color-mix(in srgb, var(--c-data) 25%, transparent);
		border-color: var(--c-data);
	}
	.switch.on .knob {
		transform: translateX(12px);
		background: var(--c-data);
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

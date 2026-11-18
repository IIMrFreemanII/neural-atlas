<script lang="ts">
	import { page } from '$app/state';
	import { resolve } from '$app/paths';

	const notFound = $derived(page.status === 404);
</script>

<svelte:head><title>{notFound ? 'Page not found' : `Error ${page.status}`} · Neural Atlas</title></svelte:head>

<main>
	<div class="sheet">
		<p class="code num">{page.status}</p>
		{#if notFound}
			<h1>Page not found</h1>
			<p class="msg">There is nothing at this address. The map has every lesson, and the list has them in order.</p>
		{:else}
			<h1>Something went wrong</h1>
			<p class="msg">{page.error?.message ?? 'An unexpected error stopped this page from loading.'}</p>
		{/if}
		<div class="actions">
			<a class="btn primary" href={resolve('/')}>Back to the map</a>
			<a class="btn" href={resolve('/learn')}>Browse lessons</a>
		</div>
	</div>
</main>

<style>
	main {
		display: grid;
		place-items: center;
		min-height: calc(100vh - var(--topbar-h));
		padding: 48px 24px;
	}
	.sheet {
		display: grid;
		gap: 12px;
		justify-items: start;
		width: min(520px, 100%);
		padding: 32px;
		background-color: var(--sheet);
		background-image: var(--grain);
		border: 1px solid var(--rule);
		border-radius: 6px;
		box-shadow: var(--shadow);
	}
	.code {
		font-family: var(--font-display);
		font-style: italic;
		font-size: 44px;
		line-height: 1;
		color: var(--data);
	}
	h1 {
		font-family: var(--font-display);
		font-weight: 500;
		font-size: 38px;
		line-height: 1.05;
		letter-spacing: -0.02em;
	}
	.msg {
		color: var(--ink-2);
	}
	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: 12px;
		margin-top: 8px;
	}
</style>

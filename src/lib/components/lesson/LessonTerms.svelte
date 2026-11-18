<script lang="ts">
	import { tick } from 'svelte';
	import { resolve } from '$app/paths';
	import { termsForLesson } from '../../content/glossary.ts';
	import { settings } from '../../state/settings.svelte.ts';

	interface Props {
		lessonKey: string;
	}

	let { lessonKey }: Props = $props();

	const terms = $derived(termsForLesson(lessonKey));
	let openId: string | null = $state(null);
	const open = $derived(terms.find((e) => e.id === openId) ?? null);

	let panel: HTMLDivElement | undefined = $state();

	async function toggle(id: string): Promise<void> {
		openId = openId === id ? null : id;
		if (openId === null) return;
		await tick();
		panel?.scrollIntoView({ block: 'nearest', behavior: settings.reducedMotion ? 'auto' : 'smooth' });
	}

	// A different lesson means a different list: close whatever was open.
	$effect(() => {
		void lessonKey;
		openId = null;
	});
</script>

{#if terms.length > 0}
	<section class="terms" aria-labelledby="lesson-terms-h">
		<h3 id="lesson-terms-h">Terms in this lesson</h3>
		<ul class="chips">
			{#each terms as e (e.id)}
				<li>
					<button
						type="button"
						class="chip"
						style:--tone={e.tone ? `var(--c-${e.tone})` : 'var(--c-text-2)'}
						aria-expanded={openId === e.id}
						onclick={() => toggle(e.id)}>{e.term}</button
					>
				</li>
			{/each}
		</ul>
		<div class="panel" bind:this={panel} class:filled={open} aria-live="polite">
			{#if open}
				<p>{open.definition}</p>
				<a href="{resolve('/glossary')}#{open.id}">Open in glossary</a>
			{/if}
		</div>
	</section>
{/if}

<style>
	.terms {
		margin-top: 20px;
		padding-top: 20px;
		border-top: 1px solid var(--line);
	}
	h3 {
		margin-bottom: 10px;
		font-family: var(--font-body, inherit);
		font-size: 12px;
		font-weight: 500;
		color: var(--c-muted);
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.chip {
		padding: 3px 10px;
		border: 1px solid var(--line-strong);
		border-radius: var(--r-pill);
		background: transparent;
		font: inherit;
		font-size: 12px;
		color: var(--tone);
		cursor: pointer;
	}
	.chip:hover {
		background: var(--glass);
	}
	.chip[aria-expanded='true'] {
		background: var(--glass-strong);
		border-color: var(--tone);
	}
	.chip:focus-visible {
		outline: 1px solid var(--c-data);
		outline-offset: 2px;
	}
	.panel.filled {
		margin-top: 12px;
		padding: 10px 12px;
		border-radius: var(--r-md);
		background: var(--glass);
	}
	.panel p {
		margin: 0 0 6px;
		font-size: 13px;
		color: var(--c-text-2);
	}
	.panel a {
		font-size: 13px;
		color: var(--c-data);
	}
</style>

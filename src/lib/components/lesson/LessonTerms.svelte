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
						style:--tone={e.tone ? `var(--c-${e.tone})` : 'var(--ink-2)'}
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
		padding-top: 16px;
		border-top: 1px solid var(--rule);
	}
	h3 {
		margin: 0 0 10px;
		font-family: var(--font-body);
		font-size: 12px;
		font-weight: 400;
		color: var(--ink-2);
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
		padding: 2px 8px;
		border: 1px solid var(--rule);
		border-radius: 3px;
		background: transparent;
		font: inherit;
		font-size: 12px;
		color: var(--ink-2);
		cursor: pointer;
	}
	.chip:hover {
		color: var(--ink);
		border-color: var(--rule-2);
	}
	.chip[aria-expanded='true'] {
		color: var(--ink);
		background: color-mix(in srgb, var(--tone) 12%, transparent);
		border-color: var(--tone);
	}
	.panel.filled {
		margin-top: 12px;
		padding: 10px 12px;
		border: 1px solid var(--rule);
		border-radius: 4px;
		background: color-mix(in srgb, var(--ink) 4%, transparent);
	}
	.panel p {
		margin: 0 0 6px;
		font-size: 13px;
		color: var(--ink-2);
	}
	.panel a {
		font-size: 13px;
		color: var(--data);
	}
</style>

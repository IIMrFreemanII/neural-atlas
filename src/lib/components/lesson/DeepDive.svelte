<script lang="ts">
	import katex from 'katex';
	import 'katex/dist/katex.min.css';
	import type { DeepDive } from '../../lessons/types.ts';
	import Icon from '../shell/Icon.svelte';

	interface Props {
		dive: DeepDive;
	}

	let { dive }: Props = $props();

	let open = $state(false);
	let picked = $state<'math' | 'code'>('math');

	const tabs = $derived.by(() => {
		const t: ('math' | 'code')[] = [];
		if (dive.math?.length) t.push('math');
		if (dive.code) t.push('code');
		return t;
	});
	const tab = $derived(tabs.includes(picked) ? picked : tabs[0]);
	const rendered = $derived(
		(dive.math ?? []).map((m) => ({
			html: katex.renderToString(m.tex, { displayMode: true, throwOnError: false }),
			caption: m.caption
		}))
	);
	const uid = $props.id();
</script>

<section class="dive">
	<button
		type="button"
		class="head"
		aria-expanded={open}
		aria-controls="{uid}-body"
		onclick={() => (open = !open)}
	>
		<span>Go deeper</span>
		<span class="chev" class:open><Icon name="chevron-right" /></span>
	</button>
	{#if open}
		<div class="body" id="{uid}-body">
			{#if dive.note}<p class="note">{dive.note}</p>{/if}
			{#if tabs.length > 1}
				<div class="tabs" role="tablist" aria-label="Deep dive">
					{#each tabs as t (t)}
						<button
							type="button"
							role="tab"
							id="{uid}-tab-{t}"
							aria-selected={tab === t}
							aria-controls="{uid}-panel"
							onclick={() => (picked = t)}>{t === 'math' ? 'Math' : 'Code'}</button
						>
					{/each}
				</div>
			{/if}
			<div
				id="{uid}-panel"
				role={tabs.length > 1 ? 'tabpanel' : undefined}
				aria-labelledby={tabs.length > 1 && tab ? `${uid}-tab-${tab}` : undefined}
			>
				{#if tab === 'math'}
					{#each rendered as m, i (i)}
						<figure>
							<div class="math">{@html m.html}</div>
							{#if m.caption}<figcaption>{m.caption}</figcaption>{/if}
						</figure>
					{/each}
				{:else if tab === 'code' && dive.code}
					<pre><code>{dive.code.source}</code></pre>
				{/if}
			</div>
		</div>
	{/if}
</section>

<style>
	.dive {
		margin-top: 20px;
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		transition: background-color 0.15s;
	}
	.head {
		display: flex;
		width: 100%;
		align-items: center;
		justify-content: space-between;
		padding: 10px 12px;
		background: transparent;
		border: 0;
		border-radius: 4px;
		font-size: 14px;
		font-weight: 500;
		color: var(--ink);
		cursor: pointer;
	}
	.head:hover {
		background: color-mix(in srgb, var(--ink) 4%, transparent);
	}
	.chev {
		display: inline-flex;
		color: var(--ink-2);
		transition: transform 0.15s;
	}
	.chev.open {
		transform: rotate(90deg);
	}
	.body {
		padding: 0 14px 14px;
		border-top: 1px solid var(--rule);
		padding-top: 12px;
	}
	.note {
		font-size: 14px;
		color: var(--ink-2);
		margin-bottom: 12px;
	}
	.tabs {
		display: flex;
		gap: 4px;
		margin-bottom: 12px;
		border-bottom: 1px solid var(--rule);
	}
	.tabs button {
		padding: 6px 10px;
		margin-bottom: -1px;
		background: transparent;
		border: 0;
		border-bottom: 2px solid transparent;
		font-size: 13px;
		color: var(--ink-2);
		cursor: pointer;
	}
	.tabs button:hover {
		color: var(--ink);
	}
	.tabs button[aria-selected='true'] {
		color: var(--ink);
		border-bottom-color: var(--ink);
	}
	figure {
		margin: 0 0 12px;
	}
	.math {
		overflow-x: auto;
		overflow-y: hidden;
	}
	.math :global(.katex-display) {
		margin: 0.4em 0;
	}
	figcaption {
		font-size: 12px;
		color: var(--ink-2);
	}
	pre {
		margin: 0;
		padding: 12px;
		overflow-x: auto;
		background: color-mix(in srgb, var(--ink) 5%, var(--sheet));
		border: 1px solid var(--rule);
		border-radius: 4px;
		font-size: 12.5px;
		line-height: 1.55;
	}
	code {
		font-family: var(--font-mono);
	}
</style>

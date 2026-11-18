<script lang="ts">
	import type { Tone } from '../../gpu/types.ts';
	import { parseBody } from '../../lessons/markup.ts';

	interface Props {
		text: string;
		onhighlight?: (tone: Tone | null) => void;
		/** The tone currently highlighted (from a term, a legend item or the stage). Terms of that tone light up. */
		highlight?: Tone | null;
	}

	let { text, onhighlight, highlight = null }: Props = $props();

	const paragraphs = $derived(parseBody(text));
</script>

{#each paragraphs as para, i (i)}
	<p>
		{#each para as run, j (j)}
			{#if run.kind === 'bold'}
				<strong>{run.text}</strong>
			{:else if run.kind === 'term'}
				<button
					type="button"
					class="term"
					class:lit={highlight === run.tone}
					data-term
					data-tone={run.tone}
					style:--tone="var(--c-{run.tone})"
					onpointerenter={() => onhighlight?.(run.tone)}
					onpointerleave={() => onhighlight?.(null)}
					onfocus={() => onhighlight?.(run.tone)}
					onblur={() => onhighlight?.(null)}>{run.text}</button
				>
			{:else}
				{run.text}
			{/if}
		{/each}
	</p>
{/each}

<style>
	p {
		margin: 0 0 14px;
		font-size: 15px;
		line-height: 1.65;
		color: var(--ink);
	}
	strong {
		font-weight: 600;
	}
	.term {
		display: inline;
		padding: 0;
		border: 0;
		border-radius: 2px;
		background: transparent;
		font: inherit;
		font-weight: 500;
		color: var(--tone);
		text-decoration: underline;
		text-decoration-color: color-mix(in srgb, var(--tone) 45%, transparent);
		text-decoration-thickness: 2px;
		text-underline-offset: 3px;
		cursor: default;
		transition:
			background-color 0.15s,
			box-shadow 0.15s;
	}
	.term.lit {
		background: color-mix(in srgb, var(--tone) 14%, transparent);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--tone) 14%, transparent);
		text-decoration-color: var(--tone);
	}
</style>

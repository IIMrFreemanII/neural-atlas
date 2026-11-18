<script lang="ts">
	import type { Tone } from '../../gpu/types.ts';
	import { parseBody } from '../../lessons/markup.ts';

	interface Props {
		text: string;
		onhighlight?: (tone: Tone | null) => void;
	}

	let { text, onhighlight }: Props = $props();

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
					data-term
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
		color: var(--c-text-2);
		margin-bottom: 12px;
	}
	strong {
		font-weight: 600;
		color: var(--c-text);
	}
	.term {
		display: inline;
		padding: 0;
		border: 0;
		background: transparent;
		font: inherit;
		color: var(--tone);
		text-decoration: underline dotted;
		text-underline-offset: 3px;
		cursor: help;
	}
</style>

<script lang="ts">
	import type { Tone } from '../../gpu/types.ts';

	interface Props {
		items: { tone: Tone; label: string }[];
	}

	let { items }: Props = $props();
</script>

{#if items.length}
	<ul class="legend">
		{#each items as item (item.label)}
			<li class="glass">
				<span class="dot" style:background="var(--c-{item.tone})"></span>
				{item.label}
			</li>
		{/each}
	</ul>
{/if}

<style>
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	li {
		display: flex;
		align-items: center;
		gap: 8px;
		height: 28px;
		padding: 0 12px;
		border-radius: var(--r-pill);
		font-size: 12px;
		color: var(--c-text-2);
	}
	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
	}
</style>

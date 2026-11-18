<script lang="ts">
	import type { Readout } from '../../lessons/types.ts';

	interface Props {
		readouts: Readout[];
		values: Record<string, number | string>;
	}

	let { readouts, values }: Props = $props();

	function format(r: Readout, v: number | string | undefined): string {
		if (v === undefined) return '–';
		return typeof v === 'number' ? v.toFixed(r.digits ?? 2) : v;
	}
</script>

{#if readouts.length}
	<ul class="readouts">
		{#each readouts as r (r.key)}
			<li>
				<span class="label">{r.label}</span>
				<span class="value num" style:color="var(--c-{r.tone ?? 'text'})">{format(r, values[r.key])}</span>
			</li>
		{/each}
	</ul>
{/if}

<style>
	.readouts {
		display: flex;
		margin: 0;
		padding: 0;
		list-style: none;
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		background-color: var(--sheet);
		background-image: var(--grain);
		box-shadow: var(--shadow);
	}
	li {
		display: grid;
		min-width: 96px;
		padding: 6px 14px;
	}
	li + li {
		border-left: 1px solid var(--rule);
	}
	.label {
		font-size: 12px;
		color: var(--ink-2);
	}
	.value {
		font-size: 18px;
		font-weight: 500;
		line-height: 1.4;
	}
</style>

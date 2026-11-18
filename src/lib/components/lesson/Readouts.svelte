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
			<li class="glass">
				<span class="label">{r.label}</span>
				<span class="value num" style:color="var(--c-{r.tone ?? 'text'})">{format(r, values[r.key])}</span>
			</li>
		{/each}
	</ul>
{/if}

<style>
	.readouts {
		display: flex;
		flex-wrap: wrap;
		justify-content: flex-end;
		gap: 6px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	li {
		display: flex;
		align-items: baseline;
		gap: 8px;
		height: 28px;
		padding: 0 12px;
		align-items: center;
		border-radius: var(--r-pill);
		font-size: 12px;
	}
	.label {
		color: var(--c-text-2);
	}
	.value {
		font-size: 13px;
	}
</style>

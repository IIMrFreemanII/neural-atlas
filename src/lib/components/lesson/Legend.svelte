<script lang="ts">
	import type { Tone } from '../../gpu/types.ts';

	interface Props {
		items: { tone: Tone; label: string }[];
		/** The tone currently highlighted. Its items are bold, the others dim. */
		highlight?: Tone | null;
		onhighlight?: (tone: Tone | null) => void;
	}

	let { items, highlight = null, onhighlight }: Props = $props();
</script>

{#if items.length}
	<ul class="legend">
		{#each items as item (item.label)}
			<li class:on={highlight === item.tone} class:dim={highlight !== null && highlight !== item.tone}>
				<button
					type="button"
					data-tone={item.tone}
					style:--c="var(--c-{item.tone})"
					onpointerenter={() => onhighlight?.(item.tone)}
					onpointerleave={() => onhighlight?.(null)}
					onfocus={() => onhighlight?.(item.tone)}
					onblur={() => onhighlight?.(null)}
				>
					<i aria-hidden="true"></i>{item.label}
				</button>
			</li>
		{/each}
	</ul>
{/if}

<style>
	.legend {
		display: grid;
		gap: 3px;
		margin: 0;
		padding: 0;
		list-style: none;
		font-size: 12.5px;
		line-height: 1.5;
	}
	button {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 1px 6px 1px 0;
		border: 0;
		border-radius: 3px;
		background: transparent;
		font: inherit;
		color: var(--ink-2);
		cursor: default;
		transition: opacity 0.15s;
	}
	i {
		flex: none;
		width: 18px;
		height: 3px;
		border-radius: 2px;
		background: var(--c);
	}
	.on button {
		color: var(--ink);
		font-weight: 600;
	}
	.on i {
		height: 5px;
	}
	.dim button {
		opacity: 0.45;
	}
</style>

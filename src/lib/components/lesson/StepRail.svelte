<script lang="ts">
	interface Props {
		count: number;
		index: number;
		titles: string[];
		onselect: (i: number) => void;
	}

	let { count, index, titles, onselect }: Props = $props();
</script>

<div class="rail">
	<p class="label num">Step {index + 1} of {count}</p>
	<div class="segments">
		{#each titles as title, i (i)}
			<button
				type="button"
				class="seg"
				class:done={i < index}
				class:current={i === index}
				aria-label="Go to step {i + 1}: {title}"
				aria-current={i === index ? 'step' : undefined}
				onclick={() => onselect(i)}
			></button>
		{/each}
	</div>
</div>

<style>
	.rail {
		padding: 0 20px;
	}
	.label {
		font-size: 12px;
		color: var(--c-muted);
		margin-bottom: 8px;
	}
	.segments {
		display: flex;
		gap: 4px;
	}
	.seg {
		position: relative;
		flex: 1;
		height: 3px;
		padding: 0;
		border: 0;
		border-radius: 2px;
		background: var(--line-strong);
		cursor: pointer;
	}
	/* Larger hit area without changing the visual height. */
	.seg::after {
		content: '';
		position: absolute;
		inset: -8px 0;
	}
	.seg.done {
		background: var(--c-data);
	}
	.seg.current {
		background: var(--c-text);
	}
</style>

<script lang="ts">
	interface Props {
		count: number;
		index: number;
		titles: string[];
		onselect: (i: number) => void;
		/** Number of question pages after the steps. */
		questions?: number;
		/** Whether each question is solved. */
		solved?: boolean[];
	}

	let { count, index, titles, onselect, questions = 0, solved = [] }: Props = $props();

	const onQuestion = $derived(index >= count);
</script>

<div class="rail">
	<p class="label num">
		{#if onQuestion}Question {index - count + 1} of {questions}{:else}Step {index + 1} of {count}{/if}
	</p>
	<div class="segments">
		{#each titles as title, i (i)}
			<button
				type="button"
				class="seg"
				class:done={i < index || onQuestion}
				class:current={i === index}
				aria-label="Go to step {i + 1}: {title}"
				aria-current={i === index ? 'step' : undefined}
				onclick={() => onselect(i)}
			></button>
		{/each}
		{#each { length: questions } as _, j (j)}
			{@const isSolved = solved[j] === true}
			<button
				type="button"
				class="seg"
				class:first-question={j === 0}
				class:solved={isSolved}
				class:current={count + j === index}
				aria-label="Go to question {j + 1}{isSolved ? ', solved' : ''}"
				aria-current={count + j === index ? 'step' : undefined}
				onclick={() => onselect(count + j)}
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
	.seg.first-question {
		margin-left: 8px;
	}
	.seg.solved {
		background: var(--c-ok);
	}
	.seg.current {
		background: var(--c-text);
	}
</style>

<script lang="ts">
	interface Props {
		count: number;
		/** Current page: a step, a question (count + j), or count + questions for the lesson-complete page. */
		index: number;
		titles: string[];
		onselect: (i: number) => void;
		/** Number of question pages after the steps. */
		questions?: number;
		/** Whether each question is solved. */
		solved?: boolean[];
	}

	let { count, index, titles, onselect, questions = 0, solved = [] }: Props = $props();

	const complete = $derived(index >= count + questions);
	const onQuestion = $derived(index >= count && !complete);
	const solvedCount = $derived(solved.filter(Boolean).length);
</script>

<div class="rail">
	<div class="ruler">
		{#each titles as title, i (i)}
			<button
				type="button"
				class="seg"
				class:seen={i < index || complete}
				class:now={i === index}
				aria-label="Go to step {i + 1}: {title}"
				aria-current={i === index ? 'step' : undefined}
				onclick={() => onselect(i)}
			></button>
		{/each}
		{#each { length: questions } as _, j (j)}
			{@const isSolved = solved[j] === true}
			<button
				type="button"
				class="q"
				class:solved={isSolved}
				class:cur={count + j === index}
				aria-label="Go to question {j + 1}{isSolved ? ', solved' : ''}"
				aria-current={count + j === index ? 'step' : undefined}
				onclick={() => onselect(count + j)}
			></button>
		{/each}
	</div>
	<div class="meta">
		<p class="label">
			{#if complete}Lesson complete{:else if onQuestion}Question {index - count + 1} of {questions}{:else}Step {index + 1} of {count}{/if}
		</p>
		{#if questions > 0}<p class="solved-count">{solvedCount} of {questions} questions solved</p>{/if}
	</div>
</div>

<style>
	.rail {
		display: grid;
		gap: 6px;
	}
	.ruler {
		display: flex;
		align-items: center;
		gap: 4px;
	}
	.seg {
		position: relative;
		flex: 1;
		height: 4px;
		padding: 0;
		border: 0;
		border-radius: 1px;
		background: var(--rule-2);
		cursor: pointer;
	}
	.seg.seen {
		background: var(--ink-2);
	}
	.seg.now {
		height: 6px;
		background: var(--tone, var(--data));
	}
	/* Larger hit area without changing the visual height. */
	.seg::after {
		content: '';
		position: absolute;
		inset: -9px 0;
	}
	.q {
		position: relative;
		flex: none;
		box-sizing: border-box;
		width: 9px;
		height: 9px;
		margin: 0 0 0 5px;
		padding: 0;
		border: 1.5px solid var(--rule-2);
		border-radius: 0;
		background: transparent;
		transform: rotate(45deg);
		cursor: pointer;
	}
	.q + .q {
		margin-left: 6px;
	}
	.q::after {
		content: '';
		position: absolute;
		inset: -6px;
	}
	.q.solved {
		background: var(--ok);
		border-color: var(--ok);
	}
	.q.cur {
		border-color: var(--tone, var(--data));
		border-width: 2px;
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--tone, var(--data)) 20%, transparent);
	}
	.meta {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		font-size: 12px;
		color: var(--ink-2);
	}
	p {
		margin: 0;
	}
	.label {
		font-family: var(--font-body);
		font-variant-numeric: tabular-nums;
	}
</style>

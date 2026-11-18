<script lang="ts">
	import { progress } from '../../state/progress.svelte.ts';

	interface Props {
		/** '<track>/<id>'. */
		lessonKey: string;
		/** How many questions the lesson has. Nothing is drawn for 0. */
		count: number;
	}

	let { lessonKey, count }: Props = $props();

	const solved = $derived(Array.from({ length: count }, (_, j) => progress.isSolved(lessonKey, j)));
	const text = $derived(`${solved.filter(Boolean).length} of ${count} questions solved`);
</script>

<!-- One bar per question, in the lesson's order, like the question segments of the step rail. -->
{#if count > 0}
	<span class="pips" role="img" aria-label={text} title={text}>
		{#each solved as s, j (j)}<span class="pip" class:solved={s}></span>{/each}
	</span>
{/if}

<style>
	.pips {
		flex: none;
		display: inline-flex;
		gap: 2px;
	}
	.pip {
		width: 8px;
		height: 3px;
		border-radius: 2px;
		background: var(--line-strong);
	}
	.pip.solved {
		background: var(--c-ok);
	}
</style>

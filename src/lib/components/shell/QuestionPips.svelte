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

<!-- One diamond per question, in the lesson's order, like the question marks of the step rail. -->
{#if count > 0}
	<span class="pips" role="img" aria-label={text} title={text}>
		{#each solved as s, j (j)}<span class="pip" class:solved={s}></span>{/each}
	</span>
{/if}

<style>
	.pips {
		flex: none;
		display: inline-flex;
		gap: 4px;
	}
	/* A small ink diamond per question, filled once solved. */
	.pip {
		width: 7px;
		height: 7px;
		border: 1.5px solid var(--rule-2);
		transform: rotate(45deg);
	}
	.pip.solved {
		background: var(--tone, var(--ink-2));
		border-color: var(--tone, var(--ink-2));
	}
</style>

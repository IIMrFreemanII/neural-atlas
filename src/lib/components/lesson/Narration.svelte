<script lang="ts">
	import type { Tone } from '../../gpu/types.ts';
	import type { LessonDef, LessonMeta, Track } from '../../lessons/types.ts';
	import { lessonHref } from '../../content/curriculum.ts';
	import { parseBody } from '../../lessons/markup.ts';
	import DeepDive from './DeepDive.svelte';
	import StepRail from './StepRail.svelte';

	interface Props {
		lesson: LessonDef;
		track: Track;
		next: LessonMeta | null;
		stepIndex: number;
		onstep: (i: number) => void;
		onhighlight: (tone: Tone | null) => void;
	}

	let { lesson, track, next, stepIndex, onstep, onhighlight }: Props = $props();

	const step = $derived(lesson.steps[stepIndex]);
	const paragraphs = $derived(parseBody(step.body));
	const last = $derived(stepIndex === lesson.steps.length - 1);

	let scroller: HTMLDivElement | undefined = $state();
	$effect(() => {
		void stepIndex;
		if (scroller) scroller.scrollTop = 0;
	});
</script>

<aside class="narration" aria-label="Lesson narration">
	<header>
		<p class="track" style:color="var(--c-{track.tone})">{track.title}</p>
		<h1>{lesson.title}</h1>
	</header>
	<StepRail
		count={lesson.steps.length}
		index={stepIndex}
		titles={lesson.steps.map((s) => s.title)}
		onselect={onstep}
	/>
	<div class="scroll" bind:this={scroller}>
		<h2>{step.title}</h2>
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
							onpointerenter={() => onhighlight(run.tone)}
							onpointerleave={() => onhighlight(null)}
							onfocus={() => onhighlight(run.tone)}
							onblur={() => onhighlight(null)}>{run.text}</button
						>
					{:else}
						{run.text}
					{/if}
				{/each}
			</p>
		{/each}
		{#key stepIndex}
			{#if step.deepDive}<DeepDive dive={step.deepDive} />{/if}
		{/key}
	</div>
	<footer>
		<button type="button" class="btn" disabled={stepIndex === 0} onclick={() => onstep(stepIndex - 1)}>
			Back
		</button>
		{#if !last}
			<button type="button" class="btn primary" onclick={() => onstep(stepIndex + 1)}>Next</button>
		{:else if next}
			<a class="btn primary" href={lessonHref(next)}>Next lesson</a>
		{:else}
			<a class="btn primary" href="/learn">All lessons</a>
		{/if}
	</footer>
</aside>

<style>
	.narration {
		display: flex;
		flex-direction: column;
		min-height: 0;
		background: var(--bg-panel);
		border-left: 1px solid var(--line);
	}
	header {
		padding: 20px 20px 16px;
	}
	.track {
		font-size: 12px;
		margin-bottom: 2px;
	}
	h1 {
		font-size: 20px;
		line-height: 1.25;
	}
	.scroll {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		padding: 20px 20px 24px;
	}
	h2 {
		font-size: 17px;
		margin-bottom: 12px;
	}
	.scroll p {
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
	footer {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		padding: 14px 20px;
		background: var(--bg-panel);
		border-top: 1px solid var(--line);
	}
</style>

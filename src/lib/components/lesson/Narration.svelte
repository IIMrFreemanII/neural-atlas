<script lang="ts">
	import type { Tone } from '../../gpu/types.ts';
	import type { LessonDef, LessonMeta, Track } from '../../lessons/types.ts';
	import { resolve } from '$app/paths';
	import { lessonLink } from '../../link.ts';
	import DeepDive from './DeepDive.svelte';
	import LessonTerms from './LessonTerms.svelte';
	import Prose from './Prose.svelte';
	import QuestionCard from './QuestionCard.svelte';
	import StepRail from './StepRail.svelte';

	interface Props {
		lesson: LessonDef;
		track: Track;
		next: LessonMeta | null;
		stepIndex: number;
		/** Whether each question is solved, by question index. */
		solved: boolean[];
		onsolve: (questionIndex: number) => void;
		onstep: (i: number) => void;
		onhighlight: (tone: Tone | null) => void;
	}

	let { lesson, track, next, stepIndex, solved, onsolve, onstep, onhighlight }: Props = $props();

	const stepCount = $derived(lesson.steps.length);
	const questions = $derived(lesson.questions ?? []);
	/** Index of the open question, or -1 on a step page. */
	const qIndex = $derived(stepIndex - stepCount);
	const step = $derived(qIndex < 0 ? lesson.steps[stepIndex] : null);
	const question = $derived(qIndex >= 0 ? questions[qIndex] : null);
	const last = $derived(stepIndex === stepCount + questions.length - 1);

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
		count={stepCount}
		index={stepIndex}
		titles={lesson.steps.map((s) => s.title)}
		onselect={onstep}
		questions={questions.length}
		{solved}
	/>
	<div class="scroll" bind:this={scroller}>
		{#if step}
			<h2>{step.title}</h2>
			<Prose text={step.body} {onhighlight} />
			{#key stepIndex}
				{#if step.deepDive}<DeepDive dive={step.deepDive} />{/if}
			{/key}
		{:else if question}
			{#key qIndex}
				<QuestionCard
					{question}
					solved={solved[qIndex] === true}
					onsolve={() => onsolve(qIndex)}
					{onhighlight}
				/>
			{/key}
		{/if}
		<LessonTerms lessonKey={`${lesson.track}/${lesson.id}`} />
	</div>
	<footer>
		<button
			type="button"
			class="btn"
			disabled={stepIndex === 0}
			title="Shortcut: left arrow"
			aria-keyshortcuts="ArrowLeft"
			onclick={() => onstep(stepIndex - 1)}
		>
			<span class="key" aria-hidden="true">←</span> Back
		</button>
		{#if !last}
			<button
				type="button"
				class="btn primary"
				title="Shortcut: right arrow"
				aria-keyshortcuts="ArrowRight"
				onclick={() => onstep(stepIndex + 1)}
			>
				{stepIndex === stepCount - 1 ? 'Questions' : 'Next'} <span class="key" aria-hidden="true">→</span>
			</button>
		{:else if next}
			<a class="btn primary" href={lessonLink(next)}>Next lesson</a>
		{:else}
			<a class="btn primary" href={resolve('/learn')}>All lessons</a>
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
	.key {
		font-family: var(--font-mono);
		opacity: 0.7;
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

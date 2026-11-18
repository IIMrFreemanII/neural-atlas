<script lang="ts">
	import { tick } from 'svelte';
	import type { Tone } from '../../gpu/types.ts';
	import type { LessonDef, LessonMeta, Track } from '../../lessons/types.ts';
	import { resolve } from '$app/paths';
	import { lessonLink } from '../../link.ts';
	import { settings } from '../../state/settings.svelte.ts';
	import Icon from '../shell/Icon.svelte';
	import DeepDive from './DeepDive.svelte';
	import LessonComplete from './LessonComplete.svelte';
	import LessonTerms from './LessonTerms.svelte';
	import Prose from './Prose.svelte';
	import QuestionCard from './QuestionCard.svelte';
	import StepRail from './StepRail.svelte';

	interface Props {
		lesson: LessonDef;
		track: Track;
		next: LessonMeta | null;
		/** Current page: a step, a question, or one past the last question for the lesson-complete page. */
		stepIndex: number;
		/** Whether each question is solved, by question index. */
		solved: boolean[];
		/** The tone currently highlighted; matching terms light up. */
		highlight: Tone | null;
		/** The card is folded into a tab at the right edge. */
		folded: boolean;
		onfold: (folded: boolean) => void;
		onsolve: (questionIndex: number) => void;
		onstep: (i: number) => void;
		onhighlight: (tone: Tone | null) => void;
	}

	let {
		lesson,
		track,
		next,
		stepIndex,
		solved,
		highlight,
		folded,
		onfold,
		onsolve,
		onstep,
		onhighlight
	}: Props = $props();

	const stepCount = $derived(lesson.steps.length);
	const questions = $derived(lesson.questions ?? []);
	const pageCount = $derived(stepCount + questions.length);
	const complete = $derived(stepIndex >= pageCount);
	/** Index of the open question, or -1 on a step page. */
	const qIndex = $derived(stepIndex - stepCount);
	const step = $derived(!complete && qIndex < 0 ? lesson.steps[stepIndex] : null);
	const question = $derived(!complete && qIndex >= 0 ? questions[qIndex] : null);
	const lastPage = $derived(stepIndex === pageCount - 1);
	const solvedCount = $derived(solved.filter(Boolean).length);

	/** What the folded tab shows: the position in the lesson. */
	const tabLabel = $derived(
		complete
			? 'Done'
			: qIndex >= 0
				? `Q ${qIndex + 1} / ${questions.length}`
				: `${stepIndex + 1} / ${stepCount}`
	);
	const tabName = $derived(
		complete ? 'Lesson complete' : qIndex >= 0 ? `Question ${qIndex + 1}` : (step?.title ?? '')
	);

	let scroller: HTMLDivElement | undefined = $state();
	$effect(() => {
		void stepIndex;
		if (scroller) scroller.scrollTop = 0;
	});

	let hideBtn: HTMLButtonElement | undefined = $state();
	let tabBtn: HTMLButtonElement | undefined = $state();

	async function setFolded(b: boolean) {
		onfold(b);
		await tick();
		(b ? tabBtn : hideBtn)?.focus();
	}
</script>

<aside
	class="card"
	class:folded
	class:still={settings.reducedMotion}
	style:--tone="var(--c-{track.tone})"
	aria-label="Lesson narration"
	inert={folded}
>
	<header>
		<div class="title">
			<div>
				<p class="track">{track.title}</p>
				<h1>{lesson.title}</h1>
			</div>
			<button
				type="button"
				class="hide-btn"
				bind:this={hideBtn}
				aria-label="Hide the narration"
				aria-expanded={!folded}
				onclick={() => setFolded(true)}
			>
				<Icon name="chevron-right" size={14} />
			</button>
		</div>
		<StepRail
			count={stepCount}
			index={stepIndex}
			titles={lesson.steps.map((s) => s.title)}
			onselect={onstep}
			questions={questions.length}
			{solved}
		/>
	</header>
	<div class="scroll" bind:this={scroller}>
		{#if complete}
			<LessonComplete
				steps={lesson.steps.map((s) => s.title)}
				{solvedCount}
				questionCount={questions.length}
				{next}
			/>
		{:else if step}
			<h2>{step.title}</h2>
			<Prose text={step.body} {onhighlight} {highlight} />
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
					{highlight}
				/>
			{/key}
		{/if}
		{#if !complete}<LessonTerms lessonKey={`${lesson.track}/${lesson.id}`} />{/if}
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
		{#if !complete}
			<button
				type="button"
				class="btn primary"
				title="Shortcut: right arrow"
				aria-keyshortcuts="ArrowRight"
				onclick={() => onstep(stepIndex + 1)}
			>
				{lastPage ? 'Finish' : stepIndex === stepCount - 1 ? 'Questions' : 'Next'}
				<span class="key" aria-hidden="true">→</span>
			</button>
		{:else if next}
			<a class="btn primary" href={lessonLink(next)}>Next lesson <span class="key" aria-hidden="true">→</span></a>
		{:else}
			<a class="btn primary" href={resolve('/learn')}>All lessons</a>
		{/if}
	</footer>
</aside>

<button
	type="button"
	class="tab"
	class:folded
	class:still={settings.reducedMotion}
	bind:this={tabBtn}
	aria-label="Show the narration: {tabName}"
	inert={!folded}
	onclick={() => setFolded(false)}
>
	<Icon name="chevron-left" size={14} />
	<span class="v"><b>{tabLabel}</b>{tabName}</span>
</button>

<style>
	/* The card floats over the stage, 16px from the top, right and bottom edges. Folding slides it out of view. */
	.card {
		position: absolute;
		top: 16px;
		right: 16px;
		bottom: 16px;
		z-index: 3;
		display: grid;
		grid-template-rows: auto minmax(0, 1fr) auto;
		width: 380px;
		overflow: hidden;
		border: 1px solid var(--rule);
		border-radius: 6px;
		background-color: var(--sheet);
		background-image: var(--grain);
		box-shadow: var(--shadow);
		transition:
			transform 0.3s ease,
			visibility 0s linear 0s;
	}
	.card.folded {
		transform: translateX(calc(100% + 24px));
		visibility: hidden;
		transition:
			transform 0.3s ease,
			visibility 0s linear 0.3s;
	}
	.card.still,
	.card.still.folded {
		transition: none;
	}
	header {
		display: grid;
		gap: 14px;
		padding: 16px 20px 14px 24px;
		border-bottom: 1px solid var(--rule);
	}
	.title {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 12px;
	}
	.track {
		margin: 0 0 2px;
		font-size: 12.5px;
		font-weight: 500;
		color: var(--tone);
	}
	h1 {
		margin: 0;
		font-family: var(--font-display);
		font-weight: 500;
		font-size: 22px;
		line-height: 1.2;
		letter-spacing: -0.01em;
	}
	.hide-btn {
		display: grid;
		flex: none;
		place-items: center;
		width: 28px;
		height: 28px;
		padding: 0;
		border: 1px solid var(--rule);
		border-radius: 4px;
		background: transparent;
		color: var(--ink-2);
		cursor: pointer;
	}
	.hide-btn:hover {
		color: var(--ink);
		border-color: var(--rule-2);
	}
	.scroll {
		min-height: 0;
		overflow-y: auto;
		padding: 22px 24px 24px;
	}
	h2 {
		margin: 0 0 14px;
		font-family: var(--font-display);
		font-weight: 500;
		font-size: 26px;
		line-height: 1.15;
		letter-spacing: -0.01em;
		text-wrap: balance;
	}
	.key {
		font-family: var(--font-mono);
		opacity: 0.7;
	}
	footer {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		padding: 14px 20px;
		border-top: 1px solid var(--rule);
	}

	/* The folded card: a tab on the right edge that still says where the learner is. */
	.tab {
		position: absolute;
		top: 16px;
		right: 0;
		z-index: 3;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 12px;
		box-sizing: border-box;
		width: 44px;
		padding: 14px 0;
		border: 1px solid var(--rule);
		border-right: 0;
		border-radius: 6px 0 0 6px;
		background-color: var(--sheet);
		background-image: var(--grain);
		box-shadow: var(--shadow);
		color: var(--ink-2);
		font: inherit;
		cursor: pointer;
		transform: translateX(100%);
		visibility: hidden;
		transition:
			transform 0.3s ease,
			visibility 0s linear 0.3s;
	}
	.tab.folded {
		transform: none;
		visibility: visible;
		transition:
			transform 0.3s ease,
			visibility 0s linear 0s;
	}
	.tab.still,
	.tab.still.folded {
		transition: none;
	}
	.tab:hover {
		color: var(--ink);
	}
	.v {
		max-height: 60vh;
		overflow: hidden;
		writing-mode: vertical-rl;
		font-size: 13px;
		color: var(--ink);
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.v b {
		display: inline-block;
		margin-bottom: 8px;
		font-family: var(--font-mono);
		font-weight: 500;
		color: var(--data);
	}
</style>

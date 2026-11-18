<script lang="ts">
	import { asset } from '$app/paths';
	import type { LessonMeta } from '../../lessons/types.ts';
	import Icon from '../shell/Icon.svelte';

	interface Props {
		/** Titles of the steps the learner went through. */
		steps: string[];
		/** Questions solved and questions in the lesson. */
		solvedCount: number;
		questionCount: number;
		next: LessonMeta | null;
	}

	let { steps, solvedCount, questionCount, next }: Props = $props();

	/** The next lesson's plate, a static file under /previews (base-path aware, as on the lesson list). */
	const plate = (m: LessonMeta) => asset(`previews/${m.track}/${m.id}.jpg` as Parameters<typeof asset>[0]);

	const nextFacts = $derived(
		next
			? [
					`${next.minutes} min`,
					next.questions > 0 ? `${next.questions} ${next.questions === 1 ? 'question' : 'questions'}` : null
				]
					.filter(Boolean)
					.join(' · ')
			: ''
	);
</script>

<section class="complete" aria-labelledby="complete-h">
	<div class="head">
		<svg viewBox="0 0 44 44" aria-hidden="true"
			><circle cx="22" cy="22" r="20" fill="none" stroke="var(--ok)" stroke-width="2" /><path
				d="m13 22.5 6 6 12-13"
				fill="none"
				stroke="var(--ok)"
				stroke-width="2.6"
				stroke-linecap="round"
				stroke-linejoin="round"
			/></svg
		>
		<div>
			<h2 id="complete-h">Lesson complete</h2>
			<p class="facts">
				{steps.length}
				{steps.length === 1 ? 'step' : 'steps'}{#if questionCount > 0}
					· {solvedCount} of {questionCount} questions solved{/if}
			</p>
		</div>
	</div>

	<ol class="covered" aria-label="Steps covered">
		{#each steps as title, i (i)}
			<li><Icon name="check" size={12} />{title}</li>
		{/each}
	</ol>

	{#if next}
		<div class="next">
			<img
				class="plate"
				alt=""
				src={plate(next)}
				loading="lazy"
			/>
			<div>
				<p class="up">Up next</p>
				<h3>{next.title}</h3>
				<p class="sum">{next.summary}</p>
				<p class="facts">{nextFacts}</p>
			</div>
		</div>
	{/if}
</section>

<style>
	.complete {
		display: grid;
		gap: 16px;
	}
	.head {
		display: flex;
		align-items: center;
		gap: 14px;
	}
	.head svg {
		flex: none;
		width: 44px;
		height: 44px;
	}
	h2 {
		margin: 0;
		font-family: var(--font-display);
		font-weight: 500;
		font-size: 26px;
		line-height: 1.15;
		letter-spacing: -0.01em;
	}
	p {
		margin: 0;
	}
	.facts {
		font-size: 12px;
		color: var(--ink-2);
	}
	.covered {
		display: grid;
		gap: 3px;
		margin: 0;
		padding: 0;
		list-style: none;
		font-size: 13.5px;
		line-height: 1.5;
		color: var(--ink-2);
	}
	.covered li {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.covered :global(svg) {
		flex: none;
		color: var(--ok);
	}
	.next {
		display: grid;
		grid-template-columns: 116px 1fr;
		gap: 14px;
		align-items: start;
		padding-top: 14px;
		border-top: 1px solid var(--rule);
	}
	.plate {
		display: block;
		width: 100%;
		aspect-ratio: 1200 / 630;
		object-fit: cover;
		border: 1px solid var(--rule);
		border-radius: 3px;
		background: var(--pg);
		filter: var(--plate-filter);
	}
	.up {
		font-size: 12px;
		font-weight: 500;
		color: var(--data);
	}
	h3 {
		margin: 0;
		font-family: var(--font-display);
		font-weight: 500;
		font-size: 20px;
		line-height: 1.2;
	}
	.sum {
		margin-top: 2px;
		font-size: 13px;
		line-height: 1.45;
		color: var(--ink-2);
	}
	.next .facts {
		margin-top: 4px;
		font-family: var(--font-mono);
	}
</style>

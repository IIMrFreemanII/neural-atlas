<script lang="ts">
	import ConstellationMap from '#lib/components/shell/ConstellationMap.svelte';
	import { resolve } from '$app/paths';
	import { allLessons, lessonKey, questionTotal, tracks } from '#lib/content/curriculum.ts';
	import { doneMask, nextUp } from '#lib/content/constellation.ts';
	import { lessonLink } from '#lib/link.ts';
	import { progress } from '#lib/state/progress.svelte.ts';

	const ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI'];

	const done = $derived(doneMask((k) => progress.isComplete(k)));
	const next = $derived(nextUp(done));
	const nextMeta = $derived(next >= 0 ? allLessons[next] : null);
	const fresh = $derived(progress.completedCount === 0);
	const completeCount = $derived(
		allLessons.filter((l) => progress.isComplete(lessonKey(l.track, l.id))).length
	);
	const solvedCount = $derived(
		allLessons.reduce((n, l) => n + progress.solvedCount(lessonKey(l.track, l.id), l.questions), 0)
	);
	const questionCount = questionTotal(allLessons);

	const readyMinutes = allLessons.filter((l) => l.status === 'ready').reduce((n, l) => n + l.minutes, 0);
	const hours = Math.max(1, Math.round(readyMinutes / 60));

	const neuron = allLessons.find((l) => l.track === 'fundamentals' && l.id === 'neuron') ?? allLessons[0];
	const startTrack = tracks.findIndex((t) => t.id === neuron.track);

	const nextKey = $derived(nextMeta ? lessonKey(nextMeta.track, nextMeta.id) : '');
	const trackRows = $derived(
		tracks.map((t, i) => ({
			t,
			n: ROMAN[i],
			done: t.lessons.filter((l) => progress.isComplete(lessonKey(l.track, l.id))).length
		}))
	);
</script>

<svelte:head><title>Neural Atlas</title></svelte:head>

<div class="home">
	<aside class="intro">
		<p class="eyebrow">An atlas of machine learning</p>
		<h1>See how machines learn</h1>
		<p class="lede">
			Interactive figures that take neural networks, classic ML and language models apart, one idea
			at a time.
		</p>

		<div class="rail" aria-hidden="true">
			{#each tracks as t (t.id)}
				<span class="group">
					{#each t.lessons as l (l.id)}
						{@const k = lessonKey(l.track, l.id)}
						<i
							class:next={k === nextKey}
							style:background={progress.isComplete(k) ? `var(--c-${t.tone})` : undefined}
						></i>
					{/each}
				</span>
			{/each}
		</div>
		<p class="count num">
			{completeCount} of {allLessons.length} complete · {solvedCount} of {questionCount} questions solved
		</p>

		{#if fresh}
			<div class="up-next">
				<p class="lab">Start here · Track {ROMAN[startTrack]}, lesson 1</p>
				<h2>{neuron.title}</h2>
				<p>{neuron.summary}</p>
				<p class="lab">{neuron.minutes} min · {neuron.questions} questions</p>
			</div>
			<p class="how">
				The whole atlas takes about {hours} hours. Each lesson is a live figure you can drag, with short
				steps and a few questions; start anywhere and your progress stays in this browser.
			</p>
		{:else}
			{#if nextMeta}
				<div class="up-next">
					<p class="lab">Up next</p>
					<h2>{nextMeta.title}</h2>
					<p>{nextMeta.summary}</p>
					<p class="lab">{nextMeta.minutes} min · {nextMeta.questions} questions</p>
				</div>
			{/if}
			<ul class="track-rows">
				{#each trackRows as r (r.t.id)}
					<li style="--c: var(--c-{r.t.tone})">
						<span class="numl">{r.n}</span>
						<span>{r.t.title}</span>
						<span class="bar"><span style:width="{(r.done / r.t.lessons.length) * 100}%"></span></span>
						<span class="cnt num">{r.done}/{r.t.lessons.length}</span>
					</li>
				{/each}
			</ul>
		{/if}

		<div class="actions">
			{#if fresh}
				<a class="btn primary" href={lessonLink(neuron)}>Start with the neuron</a>
			{:else if nextMeta}
				<a class="btn primary" href={lessonLink(nextMeta)}>Continue: {nextMeta.title}</a>
			{:else}
				<a class="btn primary" href={resolve('/learn')}>Browse lessons</a>
			{/if}
			{#if fresh || nextMeta}
				<a class="alt" href={resolve('/learn')}>Browse lessons as a list</a>
			{/if}
		</div>
	</aside>

	<section class="stage" aria-label="The map">
		<ConstellationMap />
	</section>
</div>

<style>
	.home {
		display: grid;
		grid-template-columns: 340px minmax(0, 1fr);
		gap: 28px;
		align-items: start;
		max-width: 1440px;
		margin: 0 auto;
		padding: 32px 28px 56px;
	}
	.intro {
		display: grid;
		gap: 12px;
		padding: 24px;
		background-color: var(--sheet);
		background-image: var(--grain);
		border: 1px solid var(--rule);
		border-radius: 6px;
		box-shadow: var(--shadow);
	}
	.eyebrow {
		font-size: 13px;
		font-weight: 500;
		color: var(--data);
	}
	h1 {
		font-family: var(--font-display);
		font-weight: 500;
		font-size: 42px;
		line-height: 1.02;
		letter-spacing: -0.02em;
		text-wrap: balance;
	}
	.lede {
		color: var(--ink-2);
		font-size: 15px;
	}
	.rail {
		display: flex;
		gap: 8px;
		margin-top: 6px;
	}
	.group {
		flex: 1;
		display: flex;
		gap: 2px;
	}
	.group i {
		flex: 1;
		height: 5px;
		background: var(--rule-2);
		border-radius: 1px;
	}
	.group i.next {
		background: transparent;
		box-shadow: inset 0 0 0 1.5px var(--data);
	}
	.count {
		font-size: 12px;
		color: var(--ink-2);
	}
	.up-next {
		display: grid;
		gap: 2px;
		padding: 12px 0;
		border-top: 1px solid var(--rule);
		border-bottom: 1px solid var(--rule);
	}
	.up-next .lab {
		font-size: 12px;
		color: var(--ink-2);
	}
	.up-next .lab:first-child {
		color: var(--data);
		font-weight: 500;
	}
	.up-next h2 {
		font-family: var(--font-display);
		font-weight: 500;
		font-size: 26px;
	}
	.up-next p:not(.lab) {
		font-size: 14.5px;
		color: var(--ink-2);
	}
	.how {
		font-size: 13.5px;
		color: var(--ink-2);
	}
	.track-rows {
		display: grid;
		gap: 2px;
		list-style: none;
		margin: 0;
		padding: 0;
	}
	.track-rows li {
		display: grid;
		grid-template-columns: 26px 1fr 56px 30px;
		gap: 10px;
		align-items: center;
		font-size: 13.5px;
		padding: 2px 0;
	}
	.numl {
		font-family: var(--font-display);
		font-style: italic;
		font-size: 17px;
		color: var(--c);
	}
	.bar {
		height: 4px;
		background: var(--rule-2);
		border-radius: 2px;
		overflow: hidden;
	}
	.bar span {
		display: block;
		height: 100%;
		background: var(--c);
	}
	.cnt {
		font-size: 11.5px;
		color: var(--ink-3);
		text-align: right;
	}
	.actions {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 12px 18px;
		margin-top: 4px;
	}
	.actions .btn {
		height: 40px;
		padding: 0 20px;
		font-size: 14.5px;
	}
	.alt {
		font-size: 13.5px;
		color: var(--ink-2);
		text-decoration: underline;
		text-underline-offset: 3px;
	}
	.alt:hover {
		color: var(--ink);
	}
	.stage {
		min-width: 0;
	}
</style>

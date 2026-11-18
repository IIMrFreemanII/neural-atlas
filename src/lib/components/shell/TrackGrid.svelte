<script lang="ts">
	import { asset } from '$app/paths';
	import { tracks, lessonKey, questionTotal } from '../../content/curriculum.ts';
	import { lessonLink } from '../../link.ts';
	import { progress } from '../../state/progress.svelte.ts';
	import { settings } from '../../state/settings.svelte.ts';
	import QuestionPips from './QuestionPips.svelte';

	/** The lesson's preview image. A static file, so `asset` (typed to the files that exist) gives its address. */
	const plate = (track: string, id: string) =>
		asset(`previews/${track}/${id}.jpg` as Parameters<typeof asset>[0]);

	const ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI'];

	/** Turn a lesson's plate toward the pointer, up to 4 degrees each way. */
	function tilt(e: PointerEvent): void {
		if (settings.reducedMotion) return;
		const row = e.currentTarget as HTMLElement;
		const r = row.querySelector('.plate-wrap')?.getBoundingClientRect();
		if (!r) return;
		const c = (v: number) => Math.max(0, Math.min(1, v)) - 0.5;
		const x = c((e.clientX - r.left) / r.width);
		const y = c((e.clientY - r.top) / r.height);
		row.style.setProperty('--ry', `${(x * 8).toFixed(1)}deg`);
		row.style.setProperty('--rx', `${(-y * 8).toFixed(1)}deg`);
	}
	function untilt(e: PointerEvent): void {
		const row = e.currentTarget as HTMLElement;
		row.style.removeProperty('--rx');
		row.style.removeProperty('--ry');
	}
</script>

<div class="chapters" class:still={settings.reducedMotion}>
	{#each tracks as track, ti (track.id)}
		{@const done = track.lessons.filter((l) => progress.isComplete(lessonKey(l.track, l.id))).length}
		{@const solved = track.lessons.reduce(
			(n, l) => n + progress.solvedCount(lessonKey(l.track, l.id), l.questions),
			0
		)}
		<section class="chapter" style:--tone="var(--c-{track.tone})">
			<div class="chapter-head">
				<span class="numl" aria-hidden="true">{ROMAN[ti]}</span>
				<h2>{track.title}</h2>
				<p>{track.blurb}</p>
				<p class="lab num">
					{done} of {track.lessons.length} complete · {solved} of {questionTotal(track.lessons)} questions solved
				</p>
			</div>
			<ol class="toc-list">
				{#each track.lessons as l, li (l.id)}
					{@const complete = progress.isComplete(lessonKey(l.track, l.id))}
					<li>
						{#if l.status === 'ready'}
							<a class="toc-row" class:done={complete} href={lessonLink(l)} onpointermove={tilt} onpointerleave={untilt}>
								<span class="plate-wrap">
									<img
										class="plate"
										alt=""
										loading="lazy"
										src={plate(l.track, l.id)}
									/>
								</span>
								<span class="meta">
									<span class="t"><span class="n">{ti + 1}.{li + 1}</span>{l.title}</span>
									<span class="s">{l.summary}</span>
								</span>
								<span class="side">
									<span class="min num">{l.minutes} min</span>
									<QuestionPips lessonKey={lessonKey(l.track, l.id)} count={l.questions} />
								</span>
							</a>
						{:else}
							<div class="toc-row soon">
								<span class="plate-wrap"><span class="plate blank"></span></span>
								<span class="meta">
									<span class="t"><span class="n">{ti + 1}.{li + 1}</span>{l.title}</span>
									<span class="s">{l.summary}</span>
								</span>
								<span class="side">
									<span class="tag">Soon</span>
									<span class="min num">{l.minutes} min</span>
								</span>
							</div>
						{/if}
					</li>
				{/each}
			</ol>
		</section>
	{/each}
</div>

<style>
	.chapters {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 48px 56px;
		align-items: start;
		max-width: 1280px;
		margin: 0 auto;
		padding: 0 48px;
	}
	.chapter {
		display: grid;
		gap: 14px;
	}
	.chapter-head {
		display: grid;
		grid-template-columns: 52px 1fr;
		gap: 0 12px;
		align-items: baseline;
	}
	.numl {
		font-family: var(--font-display);
		font-size: 44px;
		line-height: 1;
		font-style: italic;
		color: var(--tone);
	}
	h2 {
		font-family: var(--font-display);
		font-weight: 500;
		font-size: 27px;
		line-height: 1.15;
		color: var(--tone);
	}
	.chapter-head p {
		grid-column: 2;
		color: var(--ink-2);
		font-size: 14px;
	}
	.chapter-head .lab {
		margin-top: 2px;
		font-size: 12px;
	}
	.toc-list {
		list-style: none;
		margin: 0;
		padding: 0;
		border-top: 1px solid var(--rule);
	}
	.toc-list li {
		border-bottom: 1px solid var(--rule);
	}
	.toc-row {
		display: grid;
		grid-template-columns: 92px 1fr auto;
		gap: 14px;
		align-items: center;
		padding: 9px 0;
		transition: background-color 0.12s;
	}
	.meta {
		display: grid;
		gap: 1px;
		min-width: 0;
	}
	.t {
		display: flex;
		align-items: baseline;
		gap: 8px;
		font-size: 14.5px;
		font-weight: 500;
	}
	.t .n {
		font-family: var(--font-mono);
		font-size: 11.5px;
		font-weight: 400;
		color: var(--ink-3);
	}
	.done .t::after {
		content: '';
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--tone);
		transform: translateY(-2px);
	}
	.s {
		font-size: 12.5px;
		line-height: 1.4;
		color: var(--ink-2);
	}
	.side {
		display: grid;
		justify-items: end;
		gap: 4px;
	}
	.min {
		font-size: 12px;
		color: var(--ink-2);
	}
	.tag {
		font-size: 11.5px;
		font-weight: 500;
		color: var(--tone);
	}
	.soon {
		color: var(--ink-3);
	}

	/* The plate: today's link-preview image, turned into ink on paper by the theme's filter. */
	.plate-wrap {
		display: block;
		border-radius: 3px;
		transform: perspective(520px);
		transition:
			transform 0.16s ease-out,
			box-shadow 0.16s ease-out;
	}
	.plate {
		display: block;
		width: 100%;
		aspect-ratio: 1200 / 630;
		object-fit: cover;
		border-radius: 3px;
		border: 1px solid var(--rule);
		filter: var(--plate-filter);
		background: var(--pg);
	}
	.plate.blank {
		filter: none;
	}
	a.toc-row:hover .plate-wrap,
	a.toc-row:focus-visible .plate-wrap {
		transform: perspective(520px) translateY(-3px) rotateX(var(--rx, 0deg)) rotateY(var(--ry, 0deg)) scale(1.07);
		box-shadow: var(--lift);
		position: relative;
		z-index: 1;
	}
	a.toc-row:hover .t,
	a.toc-row:focus-visible .t {
		color: var(--tone);
	}
	a.toc-row:focus-visible {
		outline: 2px solid var(--tone);
		outline-offset: 2px;
	}
	.still .plate-wrap {
		transition: none;
	}
	.still .toc-row {
		transition: none;
	}
</style>

<script lang="ts">
	import { asset, resolve } from '$app/paths';
	import { tick } from 'svelte';
	import AtlasScene from './AtlasScene.svelte';
	import { allLessons, tracks, lessonKey, questionTotal } from '../../content/curriculum.ts';
	import { doneMask, nextUp } from '../../content/constellation.ts';
	import { lessonLink } from '../../link.ts';
	import { progress } from '../../state/progress.svelte.ts';
	import { settings } from '../../state/settings.svelte.ts';

	const regionNames = ['Neural networks', 'Classic machine learning', 'Language models', 'Vision & generation'];
	const motifs = ['From one neuron to a learning system', 'Find the structure in your data', 'Inside the machinery of language', 'From pixels to possibilities'];
	let region = $state<string | null>(null);
	let lesson = $state<number | null>(null);
	let audience = $state<'new' | 'experienced' | null>(null);
	let scene: AtlasScene | undefined = $state();
	let detailHeading: HTMLHeadingElement | undefined = $state();
	const selectedTrack = $derived(tracks.find((t) => t.id === region));
	const selectedLesson = $derived(lesson !== null ? selectedTrack?.lessons[lesson] ?? null : null);
	const completedMask = $derived(selectedTrack?.lessons.reduce((mask, item, i) => progress.isComplete(lessonKey(item.track, item.id)) ? mask | (1 << i) : mask, 0) ?? 0);
	const completed = $derived(allLessons.filter((l) => progress.isComplete(lessonKey(l.track, l.id))).length);
	const solved = $derived(allLessons.reduce((n, l) => n + progress.solvedCount(lessonKey(l.track, l.id), l.questions), 0));
	const nextIndex = $derived(nextUp(doneMask((key) => progress.isComplete(key))));
	const next = $derived(nextIndex >= 0 ? allLessons[nextIndex] : null);
	const hasProgress = $derived(completed > 0 || solved > 0 || allLessons.some((l) => progress.getStep(lessonKey(l.track, l.id)) > 0));
	const prerequisites = $derived(selectedLesson?.prerequisites.map((key) => allLessons.find((l) => lessonKey(l.track, l.id) === key)).filter((l) => l !== undefined) ?? []);
	const plate = (track: string, id: string) => asset(`previews/${track}/${id}.jpg` as Parameters<typeof asset>[0]);

	async function focusDetails() {
		await tick();
		detailHeading?.focus({ preventScroll: true });
	}
	function selectRegion(id: string) {
		region = id;
		lesson = null;
		void focusDetails();
	}
	function selectLesson(index: number) {
		if (!selectedTrack?.lessons[index]) return;
		lesson = index;
		void focusDetails();
	}
	function chooseStart(value: 'new' | 'experienced') {
		audience = value;
		region = 'fundamentals';
		selectLesson(value === 'new' ? 0 : 2);
	}
	function overview() {
		region = null;
		lesson = null;
		scene?.resetView();
	}
</script>

<main class="atlas-home">
	<header class="introduction">
		<div>
			<p class="eyebrow"><span></span> An atlas of machine learning</p>
			<h1>See how machines learn</h1>
			<p class="lede">Follow your curiosity. Turn an abstract idea into something you can see.</p>
		</div>
		<div class="entry">
			<p class="small-label">Choose your starting point</p>
			<div class="entry-buttons">
				<button class="btn primary" aria-pressed={audience === 'new'} onclick={() => chooseStart('new')}>New to AI <span aria-hidden="true">↗</span></button>
				<button class="btn" aria-pressed={audience === 'experienced'} onclick={() => chooseStart('experienced')}>I know the basics <span aria-hidden="true">↗</span></button>
			</div>
			{#if hasProgress && next}
				<a class="continue" href={lessonLink(next)}>Continue learning: {next.title} <span aria-hidden="true">→</span></a>
			{:else}
				<p class="entry-note">Or choose any region below. There’s no wrong way in.</p>
			{/if}
		</div>
	</header>

	<section class="atlas-frame" aria-label="Explore the learning roadmap">
		<div class="atlas-toolbar">
			<div class="location"><span class="coordinate">ATLAS / {selectedTrack ? String(tracks.indexOf(selectedTrack) + 1).padStart(2, '0') : '00'}</span><span>{selectedTrack?.title ?? 'The whole picture'}</span></div>
			<div class="view-controls">
				<button onclick={() => scene?.resetView()} aria-label="Reset camera">Reset view</button>
				<button class:current={!region} onclick={overview}>Whole atlas <span aria-hidden="true">↗</span></button>
			</div>
		</div>
		<div class="atlas-body" class:focused={region !== null}>
			<div class="scene-wrap">
				<AtlasScene bind:this={scene} {region} {lesson} completed={completedMask} reducedMotion={settings.reducedMotion} onregion={selectRegion} onlesson={selectLesson} />
			</div>
			{#if selectedTrack}
				<aside class="detail" aria-label="Topic and lesson preview" style:--accent="var(--c-{selectedTrack.tone})">
					{#if selectedLesson}
						<div class="preview-meta"><span>Lesson {lesson! + 1} / {selectedTrack.lessons.length}</span><button onclick={() => selectRegion(selectedTrack.id)} aria-label="Back to {selectedTrack.title} overview">All {selectedTrack.lessons.length} lessons ↗</button></div>
						<img class="preview-image" src={plate(selectedLesson.track, selectedLesson.id)} alt="Interactive figure from {selectedLesson.title}" />
						<div class="preview-copy">
							<p class="small-label">{progress.isComplete(lessonKey(selectedLesson.track, selectedLesson.id)) ? 'Completed · revisit anytime' : region === 'fundamentals' && audience === 'experienced' && lesson === 2 ? 'A starting point for developers' : lesson === 0 ? 'Your first building block' : 'Explore the idea'}</p>
							<h2 bind:this={detailHeading} tabindex="-1">{selectedLesson.title}</h2>
							<p>{selectedLesson.summary}</p>
							<p class="lesson-facts">{selectedLesson.minutes} minutes <span>·</span> {selectedLesson.questions} questions <span>·</span> Interactive</p>
							<div class="prerequisites"><span>Builds on</span>{#if prerequisites.length}{#each prerequisites as prerequisite}<a href={lessonLink(prerequisite)}>{prerequisite.title} ↗</a>{/each}{:else}<b>No prior knowledge needed</b>{/if}</div>
							<a class="btn primary start-lesson" href={lessonLink(selectedLesson)}>Start lesson <span aria-hidden="true">→</span></a>
						</div>
					{:else}
						<div class="region-copy">
							<p class="small-label">Region {tracks.indexOf(selectedTrack) + 1} · {selectedTrack.lessons.length} lessons</p>
							<h2 bind:this={detailHeading} tabindex="-1">{regionNames[tracks.indexOf(selectedTrack)]}</h2>
							<p>{selectedTrack.blurb}</p>
							<p class="path-caption">A recommended path, with room to wander. Choose any lesson to take a closer look.</p>
							<ol class="region-lessons">{#each selectedTrack.lessons as item, i}<li><button onclick={() => selectLesson(i)}><span class="step-num">{progress.isComplete(lessonKey(item.track, item.id)) ? '✓' : String(i + 1).padStart(2, '0')}</span><span>{item.title}</span><span aria-hidden="true">↗</span></button></li>{/each}</ol>
						</div>
					{/if}
				</aside>
			{/if}
		</div>
		<nav class="region-tabs" aria-label="Atlas regions">
			{#each tracks as track, i}
				<button class:active={region === track.id} aria-pressed={region === track.id} onclick={() => selectRegion(track.id)} style:--accent="var(--c-{track.tone})">
					<span class="region-number">0{i + 1}</span><span><strong>{regionNames[i]}</strong><small>{motifs[i]}</small></span><span class="region-arrow" aria-hidden="true">↗</span>
				</button>
			{/each}
		</nav>
	</section>

	{#if selectedTrack && selectedLesson}
		<nav class="lesson-path" aria-label="{selectedTrack.title} learning path" style:--path-count={selectedTrack.lessons.length} style:--accent="var(--c-{selectedTrack.tone})">
			{#each selectedTrack.lessons as item, i}
				<button aria-current={lesson === i ? 'step' : undefined} class:selected={lesson === i} onclick={() => selectLesson(i)}><span>{progress.isComplete(lessonKey(item.track, item.id)) ? '✓' : String(i + 1).padStart(2, '0')}</span>{item.title}</button>
			{/each}
		</nav>
	{/if}
	<div class="atlas-foot"><p><span class="progress-dot"></span>{completed} of {allLessons.length} complete <span>·</span> {solved} of {questionTotal(allLessons)} questions solved</p><a href={resolve('/learn')}>Browse lessons as a list <span aria-hidden="true">→</span></a></div>

	<section class="directory" aria-label="All lessons">
		<div class="directory-heading"><h2>Every idea has a way in.</h2><p>Start small, follow a path, or jump to what interests you.</p></div>
		<div class="directory-tracks">{#each tracks as track}<div style:--accent="var(--c-{track.tone})"><h3>{track.title}</h3>{#each track.lessons as item}{@const key = lessonKey(item.track, item.id)}<a class="star" href={lessonLink(item)} aria-label="{item.title}, {progress.isComplete(key) ? 'complete' : 'not started'}, {progress.solvedCount(key, item.questions)} of {item.questions} questions solved"><span>{item.title}</span><span>{progress.isComplete(key) ? '✓' : `${item.minutes} min`}</span></a>{/each}</div>{/each}</div>
	</section>
</main>

<style>
	:global(body:has(.atlas-home)) { min-width: 360px; }
	.atlas-home { max-width: 1600px; margin: auto; padding: 30px 32px 64px; }
	.introduction { display: flex; align-items: center; justify-content: space-between; gap: 24px; margin-bottom: 26px; }
	.eyebrow, .small-label { font-size: 11px; text-transform: uppercase; letter-spacing: .12em; font-weight: 600; }
	.eyebrow { display: flex; align-items: center; gap: 8px; color: var(--data); margin-bottom: 10px; }
	.eyebrow span, .progress-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--data); }
	h1 { font-size: clamp(40px, 4vw, 62px); line-height: 1.06; letter-spacing: -.035em; }
	.lede { margin-top: 10px; font-size: 14px; color: var(--ink-2); }
	.entry { flex-shrink: 0; }
	.entry .small-label { color: var(--ink-3); margin-bottom: 10px; }
	.entry-buttons { display: flex; gap: 8px; }
	.entry-buttons .btn { height: 40px; border-radius: 5px; gap: 20px; font-size: 13px; }
	.entry-note, .continue { display: block; font-size: 11px; margin-top: 9px; color: var(--ink-2); }
	.continue { color: var(--data); text-decoration: underline; text-underline-offset: 3px; }
	.atlas-frame { border: 1px solid var(--rule-2); border-radius: 8px; overflow: hidden; background: var(--pg); box-shadow: var(--shadow); }
	.atlas-toolbar { min-height: 48px; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 8px 18px; border-bottom: 1px solid var(--rule); background: var(--sheet); }
	.location { display: flex; gap: 18px; align-items: center; font-size: 12px; }
	.coordinate { color: var(--ink-3); font: 10px var(--font-mono); letter-spacing: .08em; }
	.view-controls { display: flex; gap: 6px; }
	.view-controls button { background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 5px 10px; font-size: 11px; cursor: pointer; }
	.view-controls button:hover, .view-controls button.current { background: var(--pg); border-color: var(--rule); }
	.atlas-body { display: grid; grid-template-columns: minmax(0, 1fr); }
	.atlas-body.focused { grid-template-columns: minmax(0, 1fr) 310px; }
	.scene-wrap { height: 500px; min-width: 0; background: radial-gradient(ellipse at 50% 45%, var(--sheet), transparent 72%); }
	.detail { border-left: 1px solid var(--rule); background: var(--sheet); }
	.preview-meta { display: flex; align-items: center; justify-content: space-between; padding: 12px 18px; font: 10px var(--font-mono); color: var(--ink-3); }
	.preview-meta button { border: 0; padding: 0; background: transparent; font-size: 10px; color: var(--data); cursor: pointer; }
	.preview-image { display: block; width: calc(100% - 36px); aspect-ratio: 2.4; object-fit: cover; margin: 0 18px; border: 1px solid var(--rule); border-radius: 4px; filter: var(--plate-filter); }
	.preview-copy, .region-copy { padding: 18px; }
	.detail .small-label { color: var(--accent); font-size: 9px; }
	.detail h2 { font-size: 30px; letter-spacing: -.025em; line-height: 1.1; margin: 8px 0 12px; }
	.detail h2:focus { outline: none; }
	.detail p:not(.small-label) { font-size: 13px; line-height: 1.65; color: var(--ink-2); }
	.detail p.lesson-facts { font-size: 10px; margin-top: 14px; }
	.lesson-facts span { padding: 0 4px; color: var(--ink-3); }
	.prerequisites { display: grid; gap: 2px; font-size: 11px; border-top: 1px solid var(--rule); padding-top: 12px; margin-top: 14px; }
	.prerequisites span { color: var(--ink-3); }
	.prerequisites b { font-weight: 400; }
	.prerequisites a { color: var(--data); }
	.start-lesson { width: 100%; justify-content: space-between; margin-top: 16px; height: 40px; }
	.detail .path-caption { margin-top: 14px; font-size: 11px; }
	.region-lessons { list-style: none; padding: 0; margin: 14px 0 0; }
	.region-lessons button { display: grid; grid-template-columns: 22px 1fr 12px; gap: 8px; width: 100%; text-align: left; padding: 9px 0; background: transparent; border: 0; border-top: 1px solid var(--rule); font-size: 12px; cursor: pointer; align-items: center; }
	.region-lessons button:hover { color: var(--accent); }
	.step-num { font: 10px var(--font-mono); color: var(--accent); }
	.region-tabs { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); background: var(--sheet); border-top: 1px solid var(--rule); }
	.region-tabs button { min-width: 0; display: flex; gap: 12px; align-items: center; text-align: left; padding: 18px; border: 0; border-right: 1px solid var(--rule); background: transparent; cursor: pointer; }
	.region-tabs button:last-child { border-right: 0; }
	.region-tabs button:hover, .region-tabs button.active { background: color-mix(in srgb, var(--accent) 6%, var(--sheet)); box-shadow: inset 0 2px var(--accent); }
	.region-number { color: var(--accent); font: italic 24px var(--font-display); }
	.region-tabs strong { display: block; font-size: 12px; font-weight: 500; }
	.region-tabs small { display: block; color: var(--ink-3); font-size: 9px; margin-top: 3px; }
	.region-arrow { margin-left: auto; color: var(--accent); }
	.atlas-foot { display: flex; justify-content: space-between; gap: 16px; margin: 18px 0; font-size: 11px; color: var(--ink-2); }
	.atlas-foot p span { margin-right: 8px; }
	.atlas-foot p span:not(.progress-dot) { margin-left: 8px; }
	.atlas-foot a { text-decoration: underline; text-underline-offset: 4px; }
	.lesson-path { display: grid; grid-template-columns: repeat(var(--path-count, 7), 1fr); margin-top: 16px; border: 1px solid var(--rule); border-radius: 5px; overflow: hidden; }
	.lesson-path button { display: flex; flex-direction: column; gap: 6px; text-align: left; font-size: 11px; line-height: 1.4; padding: 12px; border: 0; border-right: 1px solid var(--rule); background: var(--sheet); cursor: pointer; }
	.lesson-path button:last-child { border-right: 0; }
	.lesson-path button span { color: var(--accent); font: 10px var(--font-mono); }
	.lesson-path button.selected { box-shadow: inset 0 2px var(--accent); background: color-mix(in srgb, var(--accent) 7%, var(--sheet)); }
	.directory { margin-top: 56px; border-top: 1px solid var(--rule-2); padding-top: 28px; }
	.directory-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 26px; }
	.directory h2 { font-size: 32px; letter-spacing: -.02em; }
	.directory-heading p { color: var(--ink-2); font-size: 12px; }
	.directory-tracks { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 28px; }
	.directory h3 { font-size: 21px; color: var(--accent); margin-bottom: 10px; }
	.star { display: flex; justify-content: space-between; gap: 10px; border-top: 1px solid var(--rule); padding: 10px 0; font-size: 12px; }
	.star:hover { color: var(--accent); }
	.star span:last-child { white-space: nowrap; font-size: 10px; color: var(--ink-3); }
	@media (min-width: 1440px) { .scene-wrap { height: 540px; } }
	@media (max-width: 1100px) {
		.atlas-home { padding: 24px; }
		h1 { font-size: 42px; }
		.lede { max-width: 44ch; font-size: 12px; }
		.region-tabs button { padding: 14px 12px; gap: 8px; }
		.region-tabs small { display: none; }
		.atlas-body.focused { grid-template-columns: minmax(0, 1fr) 290px; }
	}
	@media (max-width: 900px) {
		.introduction { align-items: flex-start; flex-direction: column; gap: 22px; }
		.atlas-body.focused { grid-template-columns: minmax(0, 1fr); }
		.detail { border-left: 0; border-top: 1px solid var(--rule); }
		.preview-image { max-width: 360px; }
		.region-lessons { display: grid; grid-template-columns: 1fr 1fr; column-gap: 22px; }
		.lesson-path { grid-template-columns: repeat(4, 1fr); }
		.directory-tracks { grid-template-columns: 1fr 1fr; }
		.directory-heading { display: block; }
		.directory-heading p { margin-top: 8px; }
	}
	@media (max-width: 620px) {
		:global(body:has(.atlas-home) .bar) { padding: 0 10px; gap: 6px; }
		:global(body:has(.atlas-home) .bar nav) { display: none; }
		:global(body:has(.atlas-home) .bar .brand) { font-size: 16px; gap: 4px; }
		:global(body:has(.atlas-home) .bar .brand svg) { width: 17px; }
		.atlas-home { padding: 24px 12px 40px; }
		h1 { font-size: 40px; max-width: 10ch; }
		.lede { font-size: 13px; max-width: 33ch; }
		.introduction { margin-bottom: 20px; }
		.eyebrow { font-size: 9px; }
		.entry-buttons .btn { gap: 12px; padding: 0 12px; }
		.atlas-toolbar { align-items: flex-start; padding: 10px; flex-direction: column; gap: 8px; }
		.location { font-size: 11px; gap: 12px; }
		.view-controls { align-self: flex-end; }
		.scene-wrap { height: 440px; }
		.region-tabs { grid-template-columns: repeat(2, minmax(0, 1fr)); }
		.region-tabs button { border-bottom: 1px solid var(--rule); }
		.region-tabs strong { font-size: 11px; }
		.region-number { font-size: 21px; }
		.atlas-foot { flex-direction: column; font-size: 10px; gap: 10px; }
		.region-lessons, .directory-tracks { grid-template-columns: 1fr; }
		.lesson-path { grid-template-columns: repeat(2, 1fr); }
		.directory { margin-top: 32px; }
		.directory h2 { font-size: 30px; }
	}
</style>

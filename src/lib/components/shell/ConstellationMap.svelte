<script lang="ts">
	import Stage from '../lesson/Stage.svelte';
	import { allLessons, lessonHref, lessonKey, tracks } from '../../content/curriculum.ts';
	import { MAP_H, MAP_W, TRACK_LABELS, doneMask, nextUp, stars } from '../../content/constellation.ts';
	import { createConstellationScene } from '../../lessons/home/constellation-scene.ts';
	import { progress } from '../../state/progress.svelte.ts';
	import { settings } from '../../state/settings.svelte.ts';

	type Side = 'below' | 'above' | 'left' | 'right';

	/** Where a lesson's name sits relative to its star. Anything missing is 'below'. */
	const SIDES: Record<string, Side> = {
		'classic-ml/linear-regression': 'above',
		'classic-ml/decision-boundaries': 'right',
		'classic-ml/k-means': 'right',
		'fundamentals/activations': 'above',
		'fundamentals/forward-pass': 'above',
		'fundamentals/loss-landscape': 'right',
		'fundamentals/train-a-network': 'right',
		'fundamentals/neuron': 'left',
		'llm/sampling': 'left',
		'llm/tokenization': 'above',
		'llm/embeddings': 'right',
		'vision/cnn-features': 'right'
	};

	const trackById = new Map(tracks.map((t) => [t.id, t]));
	const titleByKey = new Map(allLessons.map((l) => [lessonKey(l.track, l.id), l.title]));

	let hover = $state(-1);
	const nameW = $state<number[]>([]);
	const done = $derived(doneMask((k) => progress.isComplete(k)));
	const next = $derived(nextUp(done));
	const completeCount = $derived(stars.filter((s) => (done & (1 << s.index)) !== 0).length);
	const params = $derived({ done, hover, next });

	const hovered = $derived(hover >= 0 ? stars[hover] : null);
	const hoveredMeta = $derived(hover >= 0 ? allLessons[hover] : null);
	const hoveredTrack = $derived(hoveredMeta ? trackById.get(hoveredMeta.track) : undefined);
	const hoveredStatus = $derived(
		hover < 0 ? '' : (done & (1 << hover)) !== 0 ? 'Complete' : hover === next ? 'Start here' : 'Not started'
	);
	const cardStyle = $derived.by(() => {
		if (!hovered) return '';
		const top = hovered.y > MAP_H * 0.6;
		const toRight = hovered.x <= MAP_W * 0.65;
		const x = (hovered.x / MAP_W) * 100;
		const y = (hovered.y / MAP_H) * 100;
		// 22px from the star, or further when the name sits on that side, so the card never covers it.
		const side = SIDES[hovered.key] ?? 'below';
		const w = nameW[hovered.index] ?? 80;
		let gap = 22;
		if (side === 'above' || side === 'below') gap = Math.max(22, w / 2 + 8);
		else if (side === (toRight ? 'right' : 'left')) gap = 20 + w + 8;
		const h = toRight ? `left: calc(${x}% + ${gap}px);` : `right: calc(${100 - x}% + ${gap}px);`;
		const v = top ? `bottom: calc(${100 - y}% - 18px);` : `top: calc(${y}% - 18px);`;
		return h + v;
	});
</script>

<section class="atlas" aria-labelledby="atlas-title">
	<header>
		<h2 id="atlas-title">The map</h2>
		<p>Every star is a lesson and every line shows what builds on what. Lessons you finish glow.</p>
		<p class="count num">{completeCount} of {allLessons.length} complete</p>
	</header>

	<div class="map">
		<div class="sky">
			{#key settings.reducedMotion}<Stage createScene={createConstellationScene} {params} />{/key}
		</div>

		{#each tracks as track (track.id)}
			<span
				class="track-name"
				aria-hidden="true"
				style:left="{(TRACK_LABELS[track.id][0] / MAP_W) * 100}%"
				style:top="{(TRACK_LABELS[track.id][1] / MAP_H) * 100}%"
				style:color="var(--c-{track.tone})">{track.title}</span
			>
		{/each}

		{#each allLessons as meta, i (meta.id + meta.track)}
			{@const s = stars[i]}
			{@const complete = (done & (1 << i)) !== 0}
			<a
				class="star {SIDES[s.key] ?? 'below'}"
				class:complete
				href={lessonHref(meta)}
				style:left="{(s.x / MAP_W) * 100}%"
				style:top="{(s.y / MAP_H) * 100}%"
				aria-label="{meta.title}, {trackById.get(meta.track)?.title}, {complete ? 'complete' : 'not started'}"
				onpointerenter={() => (hover = i)}
				onpointerleave={() => (hover = -1)}
				onfocus={() => (hover = i)}
				onblur={() => (hover = -1)}
			>
				<span class="core"></span>
				<span class="name" bind:clientWidth={nameW[i]}>{meta.title}</span>
			</a>
		{/each}

		{#if hoveredMeta && hoveredTrack}
			<div class="card glass" style={cardStyle} aria-hidden="true">
				<p class="track" style:color="var(--c-{hoveredTrack.tone})">{hoveredTrack.title}</p>
				<p class="title">{hoveredMeta.title}</p>
				<p class="summary">{hoveredMeta.summary}</p>
				<p class="meta num">{hoveredMeta.minutes} min · {hoveredStatus}</p>
				{#if hoveredMeta.prerequisites.length > 0}
					<p class="builds">
						Builds on: {hoveredMeta.prerequisites.map((k) => titleByKey.get(k) ?? k).join(', ')}
					</p>
				{/if}
			</div>
		{/if}
	</div>

	<p class="list-link"><a href="/learn">Browse lessons as a list</a></p>
</section>

<style>
	.atlas {
		max-width: 1120px;
		margin: 0 auto;
		padding: 40px 24px 80px;
	}
	header {
		margin-bottom: 20px;
	}
	h2 {
		font-size: 26px;
	}
	header p {
		color: var(--c-text-2);
		font-size: 15px;
		margin-top: 6px;
	}
	header .count {
		font-size: 12px;
		color: var(--c-muted);
		margin-top: 10px;
	}
	.map {
		position: relative;
		width: 100%;
		max-width: 1072px;
		aspect-ratio: 1000 / 520;
		margin: 0 auto;
	}
	.sky {
		position: absolute;
		inset: 0;
		pointer-events: none;
	}
	.track-name {
		position: absolute;
		transform: translate(-50%, -50%);
		font-family: var(--font-display);
		font-size: 13px;
		letter-spacing: 0.04em;
		white-space: nowrap;
		opacity: 0.8;
		pointer-events: none;
		text-shadow: 0 0 4px var(--bg-void), 0 0 8px var(--bg-void);
	}
	.star {
		position: absolute;
		width: 36px;
		height: 36px;
		transform: translate(-50%, -50%);
		border-radius: 50%;
		display: block;
	}
	.core {
		position: absolute;
		left: 50%;
		top: 50%;
		width: 3px;
		height: 3px;
		margin: -1.5px 0 0 -1.5px;
		border-radius: 50%;
		background: #f4f6ff;
	}
	.name {
		position: absolute;
		font-size: 12px;
		color: var(--c-text-2);
		white-space: nowrap;
		pointer-events: none;
		text-shadow:
			0 0 3px var(--bg-void),
			0 0 6px var(--bg-void),
			0 0 10px var(--bg-void);
	}
	.below .name {
		top: 100%;
		left: 50%;
		transform: translateX(-50%);
		margin-top: -6px;
	}
	.above .name {
		bottom: 100%;
		left: 50%;
		transform: translateX(-50%);
		margin-bottom: -6px;
	}
	.left .name {
		right: 100%;
		top: 50%;
		transform: translateY(-50%);
		margin-right: -4px;
	}
	.right .name {
		left: 100%;
		top: 50%;
		transform: translateY(-50%);
		margin-left: 2px;
	}
	.star:hover .name,
	.star:focus-visible .name,
	.star.complete .name {
		color: var(--c-text);
	}
	.card {
		position: absolute;
		z-index: 2;
		width: 260px;
		padding: 14px;
		pointer-events: none;
		background: rgba(10, 12, 20, 0.88);
	}
	.card p {
		margin: 0;
	}
	.card .track {
		font-size: 12px;
	}
	.card .title {
		font-family: var(--font-display);
		font-size: 16px;
		margin-top: 4px;
	}
	.card .summary {
		font-size: 13px;
		color: var(--c-text-2);
		margin-top: 6px;
	}
	.card .meta {
		font-size: 12px;
		color: var(--c-muted);
		margin-top: 8px;
	}
	.card .builds {
		font-size: 12px;
		color: var(--c-muted);
		margin-top: 4px;
	}
	.list-link {
		text-align: center;
		margin-top: 20px;
		font-size: 13px;
	}
	.list-link a {
		color: var(--c-text-2);
		text-decoration: underline;
		text-underline-offset: 3px;
	}
	.list-link a:hover {
		color: var(--c-text);
	}
</style>

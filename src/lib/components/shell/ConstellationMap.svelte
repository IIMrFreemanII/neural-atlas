<script lang="ts">
	import { allLessons, lessonKey, tracks } from '../../content/curriculum.ts';
	import { lessonLink } from '../../link.ts';
	import { MAP_H, MAP_W, TRACK_LABELS, doneMask, links, nextUp, stars } from '../../content/constellation.ts';
	import { progress } from '../../state/progress.svelte.ts';
	import { settings } from '../../state/settings.svelte.ts';

	/** Lessons whose name runs to the left of the dot. Every other name runs to the right. */
	const LEFT = new Set([
		'fundamentals/neuron',
		'fundamentals/gradient-descent',
		'llm/transformer-block',
		'llm/sampling',
		'llm/training',
		'vision/cnn-features',
		'vision/vit',
		'vision/autoencoder',
		'vision/diffusion'
	]);

	/** How the four lessons that build on another track are joined: a curve bowed toward this point. */
	const BOWS: Record<string, [number, number]> = {
		'fundamentals/neuron>vision/convolution': [520, 560],
		'fundamentals/train-a-network>vision/cnn-features': [600, 560],
		'fundamentals/forward-pass>llm/transformer-block': [640, 320],
		'fundamentals/train-a-network>llm/training': [700, 440]
	};

	const trackById = new Map(tracks.map((t) => [t.id, t]));
	const titleByKey = new Map(allLessons.map((l) => [lessonKey(l.track, l.id), l.title]));
	const tone = (track: string) => `var(--c-${trackById.get(track as never)?.tone ?? 'data'})`;

	/** Resting tilt of the sheet and how far the pointer may turn it, in degrees. */
	const REST_X = 10;
	const SWING = 3;

	let hover = $state(-1);
	let rx = $state(REST_X);
	let ry = $state(0);

	const done = $derived(doneMask((k) => progress.isComplete(k)));
	const next = $derived(nextUp(done));
	/** Solved questions of each lesson, by star index. */
	const solved = $derived(allLessons.map((l) => progress.solvedCount(lessonKey(l.track, l.id), l.questions)));

	const hovered = $derived(hover >= 0 ? stars[hover] : null);
	const hoveredMeta = $derived(hover >= 0 ? allLessons[hover] : null);
	const hoveredTrack = $derived(hoveredMeta ? trackById.get(hoveredMeta.track) : undefined);
	const hoveredStatus = $derived(
		hover < 0 ? '' : (done & (1 << hover)) !== 0 ? 'Complete' : hover === next ? 'Start here' : 'Not started'
	);
	const cardStyle = $derived.by(() => {
		if (!hovered) return '';
		const top = hovered.y > MAP_H * 0.6;
		const toRight = hovered.x <= MAP_W * 0.6;
		const x = (hovered.x / MAP_W) * 100;
		const y = (hovered.y / MAP_H) * 100;
		// 22px from the dot, or further when the name runs that way, so the card never covers it.
		const w = (hoveredMeta?.title.length ?? 10) * 7;
		const nameRight = !LEFT.has(hovered.key);
		const gap = nameRight === toRight ? 20 + w + 8 : 22;
		const h = toRight ? `left: calc(${x}% + ${gap}px);` : `right: calc(${100 - x}% + ${gap}px);`;
		const v = top ? `bottom: calc(${100 - y}% - 18px);` : `top: calc(${y}% - 18px);`;
		return h + v;
	});

	function onmove(e: PointerEvent) {
		if (settings.reducedMotion) return;
		const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
		const x = (e.clientX - r.left) / r.width - 0.5;
		const y = (e.clientY - r.top) / r.height - 0.5;
		const c = (v: number) => Math.max(-SWING, Math.min(SWING, v));
		ry = c(x * 2 * SWING);
		rx = REST_X - c(y * 2 * SWING);
	}
	function onleave() {
		rx = REST_X;
		ry = 0;
		hover = -1;
	}

	const COLS = 'ABCDEF'.split('');
	/** The sheet's edge, inside the map's drawing space, and the size of one grid cell. */
	const EDGE = 16;
	const COLW = (MAP_W - 2 * EDGE) / COLS.length;
	const ROWH = (MAP_H - 2 * EDGE) / 4;
</script>

<div class="atlas">
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div class="map" onpointermove={onmove} onpointerleave={onleave}>
		<div class="map3d">
			<div class="tilt" class:still={settings.reducedMotion} style="--rx: {rx}deg; --ry: {ry}deg">
				<!-- The sheet lies on the desk. -->
				<svg class="lay lay0" viewBox="0 0 {MAP_W} {MAP_H}" aria-hidden="true">
					<rect class="mapsheet" x={EDGE} y={EDGE} width={MAP_W - 2 * EDGE} height={MAP_H - 2 * EDGE} stroke="var(--rule-2)" />
					{#each COLS as c, i (c)}
						{@const lx = EDGE + i * COLW}
						<text x={lx + COLW / 2} y={EDGE + 16} text-anchor="middle" class="tick">{c}</text>
						{#if i > 0}
							<line x1={lx} y1={EDGE} x2={lx} y2={EDGE + 8} stroke="var(--rule-2)" />
							<line x1={lx} y1={MAP_H - EDGE - 8} x2={lx} y2={MAP_H - EDGE} stroke="var(--rule-2)" />
						{/if}
					{/each}
					{#each [0, 1, 2, 3] as j (j)}
						{@const ly = EDGE + j * ROWH}
						{#if j > 0}
							<line x1={EDGE} y1={ly} x2={EDGE + 8} y2={ly} stroke="var(--rule-2)" />
							<line x1={MAP_W - EDGE - 8} y1={ly} x2={MAP_W - EDGE} y2={ly} stroke="var(--rule-2)" />
						{/if}
						<text x={MAP_W - EDGE - 14} y={ly + ROWH / 2} text-anchor="end" class="tick">{j + 1}</text>
					{/each}
				</svg>

				<!-- Links, names and dots float just above it. -->
				<svg class="lay lay1" viewBox="0 0 {MAP_W} {MAP_H}" role="group" aria-label="Map of all lessons">
					{#each links as l, i (i)}
						{@const a = stars[l.from]}
						{@const b = stars[l.to]}
						{@const both = (done & (1 << l.from)) !== 0 && (done & (1 << l.to)) !== 0}
						{@const bow = BOWS[`${a.key}>${b.key}`]}
						{#if l.cross}
							<path
								d="M{a.x} {a.y} {bow ? `Q${bow[0]} ${bow[1]}` : 'L'} {b.x} {b.y}"
								fill="none"
								stroke="var(--ink-3)"
								stroke-width="1.2"
								stroke-dasharray="3 5"
							/>
						{:else}
							<line
								x1={a.x}
								y1={a.y}
								x2={b.x}
								y2={b.y}
								stroke={tone(a.track)}
								stroke-width={both ? 2 : 1.3}
								stroke-opacity={both ? 0.9 : 0.45}
							/>
						{/if}
					{/each}

					{#each tracks as track (track.id)}
						<text
							class="track-name"
							x={TRACK_LABELS[track.id][0]}
							y={TRACK_LABELS[track.id][1]}
							text-anchor={TRACK_LABELS[track.id][2]}
							fill="var(--c-{track.tone})"
							aria-hidden="true">{track.title}</text
						>
					{/each}

					{#each allLessons as meta, i (meta.id + meta.track)}
						{@const s = stars[i]}
						{@const complete = (done & (1 << i)) !== 0}
						{@const isNext = i === next}
						{@const t = tone(meta.track)}
						{@const left = LEFT.has(s.key)}
						{@const lx = s.x + (left ? -14 : 14)}
						{@const anchor = left ? 'end' : 'start'}
						<a
							class="star"
							class:complete
							class:next={isNext}
							href={lessonLink(meta)}
							aria-label="{meta.title}, {trackById.get(meta.track)?.title}, {complete
								? 'complete'
								: 'not started'}{meta.questions > 0 ? `, ${solved[i]} of ${meta.questions} questions solved` : ''}"
							onpointerenter={() => (hover = i)}
							onpointerleave={() => (hover = -1)}
							onfocus={() => (hover = i)}
							onblur={() => (hover = -1)}
						>
							<circle class="hit" cx={s.x} cy={s.y} r="16" fill="transparent" />
							<circle class="focus-ring" cx={s.x} cy={s.y} r="12" fill="none" stroke={t} stroke-width="1.5" />
							{#if isNext}
								<circle class="pulse" cx={s.x} cy={s.y} r="14" fill="none" stroke={t} stroke-width="1.5" />
							{/if}
							<circle
								class="dotc"
								cx={s.x}
								cy={s.y}
								r="6.5"
								fill={complete ? t : 'var(--sheet)'}
								stroke={t}
								stroke-width="1.6"
							/>
							<text
								class="name"
								class:strong={isNext}
								x={lx}
								y={s.y + 4.5}
								text-anchor={anchor}
								fill={complete || isNext ? 'var(--ink)' : 'var(--ink-2)'}>{meta.title}</text
							>
							{#if isNext}
								<text class="sub" x={lx} y={s.y + 21} text-anchor={anchor} fill={t}
									>{progress.completedCount === 0 ? 'Start here' : 'Up next'} · {meta.minutes} min</text
								>
							{/if}
						</a>
					{/each}
				</svg>
			</div>
		</div>

		{#if hoveredMeta && hoveredTrack}
			<div class="card" style={cardStyle} aria-hidden="true">
				<p class="track" style:color="var(--c-{hoveredTrack.tone})">{hoveredTrack.title}</p>
				<p class="title">{hoveredMeta.title}</p>
				<p class="summary">{hoveredMeta.summary}</p>
				<p class="meta num">{hoveredMeta.minutes} min · {hoveredStatus}</p>
				{#if hoveredMeta.questions > 0}
					<p class="questions num" class:all={solved[hover] === hoveredMeta.questions}>
						{solved[hover]} of {hoveredMeta.questions} questions solved
					</p>
				{/if}
				{#if hoveredMeta.prerequisites.length > 0}
					<p class="builds">
						Builds on: {hoveredMeta.prerequisites.map((k) => titleByKey.get(k) ?? k).join(', ')}
					</p>
				{/if}
			</div>
		{/if}
	</div>

	<ul class="key">
		<li>
			<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="5" fill="var(--ink-2)" /></svg>
			Finished
		</li>
		<li>
			<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
				<circle cx="7" cy="7" r="6" fill="none" stroke="var(--data)" stroke-width="1.2" />
				<circle cx="7" cy="7" r="3" fill="var(--data)" />
			</svg>
			Start here or up next
		</li>
		<li>
			<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="5" fill="var(--sheet)" stroke="var(--ink-2)" stroke-width="1.4" /></svg>
			Not started
		</li>
		<li>
			<svg width="18" height="14" viewBox="0 0 18 14" aria-hidden="true"><line x1="1" y1="7" x2="17" y2="7" stroke="var(--ink-3)" stroke-width="1.3" stroke-dasharray="3 4" /></svg>
			Builds on another track
		</li>
	</ul>
</div>

<style>
	.atlas {
		width: 100%;
	}
	.map {
		position: relative;
		/* Fit by width and by height: the page padding and the key under the map are 150px. */
		width: min(100%, calc((100vh - var(--topbar-h) - 150px) * 884 / 732));
		aspect-ratio: 884 / 732;
		margin: 0 auto;
	}
	.map3d {
		position: absolute;
		inset: 0;
		perspective: 1500px;
		perspective-origin: 50% 20%;
	}
	.tilt {
		position: absolute;
		inset: 0;
		transform-style: preserve-3d;
		transform: scale(0.96) rotateX(var(--rx, 10deg)) rotateY(var(--ry, 0deg));
		transform-origin: 50% 60%;
		transition: transform 0.5s cubic-bezier(0.2, 0.7, 0.2, 1);
	}
	.tilt.still {
		transition: none;
	}
	.lay {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		overflow: visible;
	}
	.lay0 {
		pointer-events: none;
		filter: drop-shadow(0 22px 22px color-mix(in srgb, var(--cast) calc(var(--cast-a) * 130%), transparent));
	}
	.lay1 {
		transform: translateZ(22px);
	}
	.mapsheet {
		fill: var(--sheet);
	}
	.tick {
		font-family: var(--font-mono);
		font-size: 10px;
		fill: var(--ink-3);
	}
	.track-name {
		font-family: var(--font-display);
		font-style: italic;
		font-size: 21px;
		stroke: var(--sheet);
		stroke-width: 5px;
		paint-order: stroke;
		stroke-linejoin: round;
		pointer-events: none;
	}
	.star {
		cursor: pointer;
		outline: none;
		transition: transform 0.16s ease-out;
	}
	.star:hover,
	.star:focus-visible {
		transform: translateY(-3px);
	}
	.star:hover .dotc,
	.star:focus-visible .dotc {
		filter: drop-shadow(0 3px 2px color-mix(in srgb, var(--cast) 40%, transparent));
	}
	.name {
		font-family: var(--font-body);
		font-size: 14px;
		stroke: var(--sheet);
		stroke-width: 4px;
		paint-order: stroke;
		stroke-linejoin: round;
	}
	.sub {
		font-family: var(--font-body);
		font-size: 12.5px;
		font-weight: 500;
		stroke: var(--sheet);
		stroke-width: 4px;
		paint-order: stroke;
		stroke-linejoin: round;
	}
	.name.strong {
		font-weight: 600;
	}
	.star:hover .name,
	.star:focus-visible .name {
		font-weight: 600;
	}
	.focus-ring {
		opacity: 0;
	}
	.star:focus-visible .focus-ring {
		opacity: 1;
	}
	.pulse {
		animation: pulse 2.4s ease-in-out infinite;
	}
	.still .pulse {
		animation: none;
	}
	@keyframes pulse {
		0%,
		100% {
			opacity: 0.35;
		}
		50% {
			opacity: 1;
		}
	}
	.card {
		position: absolute;
		z-index: 2;
		width: 260px;
		padding: 14px;
		pointer-events: none;
		background-color: var(--sheet);
		background-image: var(--grain);
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		box-shadow: var(--shadow);
	}
	.card p {
		margin: 0;
	}
	.card .track {
		font-size: 12px;
		font-weight: 500;
	}
	.card .title {
		font-family: var(--font-display);
		font-size: 18px;
		margin-top: 4px;
	}
	.card .summary {
		font-size: 13px;
		color: var(--ink-2);
		margin-top: 6px;
	}
	.card .meta,
	.card .questions,
	.card .builds {
		font-size: 12px;
		color: var(--ink-3);
		margin-top: 4px;
	}
	.card .meta {
		margin-top: 8px;
	}
	.card .questions.all {
		color: var(--ok);
	}
	.key {
		display: flex;
		flex-wrap: wrap;
		justify-content: center;
		gap: 6px 22px;
		list-style: none;
		margin: 18px 0 0;
		padding: 0;
		font-size: 12.5px;
		color: var(--ink-2);
	}
	.key li {
		display: flex;
		align-items: center;
		gap: 8px;
	}
</style>

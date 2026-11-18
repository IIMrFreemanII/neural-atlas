<script lang="ts">
	import { untrack } from 'svelte';
	import type { Tone } from '../../gpu/types.ts';
	import type { LessonDef, LessonMeta, Params, Track } from '../../lessons/types.ts';
	import { lessonKey } from '../../content/curriculum.ts';
	import { progress } from '../../state/progress.svelte.ts';
	import Icon from '../shell/Icon.svelte';
	import ControlBar from './ControlBar.svelte';
	import Legend from './Legend.svelte';
	import Narration from './Narration.svelte';
	import Readouts from './Readouts.svelte';
	import Sidebar from './Sidebar.svelte';
	import Stage from './Stage.svelte';

	interface Props {
		lesson: LessonDef;
		meta: LessonMeta;
		track: Track;
		next: LessonMeta | null;
	}

	let { lesson, meta, track, next }: Props = $props();

	// The parent re-creates this component per lesson ({#key}), so props are read once on purpose.
	const key = untrack(() => lessonKey(meta.track, meta.id));
	const TWEEN_MS = 600;

	const discrete = new Set<string>(untrack(() => lesson.discrete ?? []));
	for (const c of untrack(() => lesson.controls)) {
		if (c.kind === 'toggle' || c.kind === 'select') discrete.add(c.key);
	}

	const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
	const targetFor = (i: number): Params => ({ ...lesson.defaults, ...lesson.steps[i].params });
	const clamp = (i: number) => Math.min(Math.max(0, Math.floor(i) || 0), lesson.steps.length - 1);

	let stepIndex = $state(untrack(() => clamp(progress.getStep(key))));
	let params = $state.raw<Params>(untrack(() => targetFor(stepIndex)));
	let highlight = $state<Tone | null>(null);
	let readoutValues = $state<Record<string, number | string>>({});
	let stage: { command(id: string): void } | undefined = $state();
	let collapsed = $state(false);

	if (untrack(() => stepIndex === lesson.steps.length - 1)) progress.complete(key);

	// Tweening
	let tween: { start: number; items: Map<string, [number, number]> } | null = null;
	let raf = 0;

	const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

	function stopTween() {
		cancelAnimationFrame(raf);
		raf = 0;
		tween = null;
	}

	function tick(now: number) {
		if (!tween) return;
		const t = Math.min(1, (now - tween.start) / TWEEN_MS);
		const e = ease(t);
		const next = { ...params };
		for (const [k, [a, b]] of tween.items) next[k] = t >= 1 ? b : a + (b - a) * e;
		params = next;
		if (t >= 1 || tween.items.size === 0) {
			tween = null;
			raf = 0;
		} else {
			raf = requestAnimationFrame(tick);
		}
	}

	function applyStep(i: number) {
		stopTween();
		const target = targetFor(i);
		const instant = reducedMotion();
		const items = new Map<string, [number, number]>();
		const next: Params = { ...params };
		for (const [k, v] of Object.entries(target)) {
			if (instant || discrete.has(k) || next[k] === undefined || next[k] === v) next[k] = v;
			else items.set(k, [next[k], v]);
		}
		params = next;
		if (items.size) {
			tween = { start: performance.now(), items };
			raf = requestAnimationFrame(tick);
		}
	}

	function goTo(i: number) {
		const n = clamp(i);
		if (n === stepIndex) return;
		stepIndex = n;
		progress.setStep(key, n);
		if (n === lesson.steps.length - 1) progress.complete(key);
		applyStep(n);
	}

	function setParam(k: string, v: number) {
		tween?.items.delete(k);
		params = { ...params, [k]: v };
	}

	$effect(() => stopTween);

	const visibleControls = $derived(
		lesson.steps[stepIndex].controls === undefined
			? lesson.controls
			: lesson.controls.filter((c) =>
					lesson.steps[stepIndex].controls!.includes(c.kind === 'action' ? c.id : c.key)
				)
	);

	function onkeydown(e: KeyboardEvent) {
		if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
		if (e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
		const t = e.target as HTMLElement | null;
		if (t && t.closest('input, select, textarea, button:not([data-term])')) return;
		e.preventDefault();
		goTo(stepIndex + (e.key === 'ArrowRight' ? 1 : -1));
	}

</script>

<svelte:window {onkeydown} />

<div class="layout" class:collapsed>
	<div class="side">
		<Sidebar currentKey={key} onclose={() => (collapsed = true)} />
	</div>

	<div class="stage-cell">
		<Stage
			bind:this={stage}
			createScene={lesson.createScene}
			{params}
			{highlight}
			onreadout={(k, v) => (readoutValues = { ...readoutValues, [k]: v })}
		/>
		<div class="tl">
			{#if collapsed}
				<button
					type="button"
					class="reopen glass"
					aria-label="Show lesson list"
					onclick={() => (collapsed = false)}
				>
					<Icon name="chevron-right" />
				</button>
			{/if}
			<Legend items={lesson.legend ?? []} />
		</div>
		<div class="tr"><Readouts readouts={lesson.readouts ?? []} values={readoutValues} /></div>
		{#if visibleControls.length}
			<div class="bottom">
				<ControlBar
					controls={visibleControls}
					{params}
					onparam={setParam}
					onaction={(id) => stage?.command(id)}
				/>
			</div>
		{/if}
	</div>

	<Narration
		{lesson}
		{track}
		{next}
		{stepIndex}
		onstep={goTo}
		onhighlight={(t) => (highlight = t)}
	/>
</div>

<style>
	/* Desktop-only layout: lesson list | stage | narration, filling the window below the top bar. */
	.layout {
		display: grid;
		height: calc(100vh - var(--topbar-h));
		grid-template-columns: 240px minmax(0, 1fr) 360px;
		grid-template-rows: minmax(0, 1fr);
		overflow: hidden;
	}
	.layout.collapsed {
		grid-template-columns: 0 minmax(0, 1fr) 360px;
	}
	.side {
		min-width: 0;
		overflow: hidden;
	}
	.collapsed .side {
		visibility: hidden;
	}
	.stage-cell {
		position: relative;
		min-height: 0;
		overflow: hidden;
		background: var(--bg-void);
	}
	.tl,
	.tr,
	.bottom {
		position: absolute;
		z-index: 2;
		pointer-events: none;
	}
	.tl {
		top: 12px;
		left: 12px;
		right: 50%;
		display: flex;
		align-items: flex-start;
		gap: 8px;
	}
	.tr {
		top: 12px;
		right: 12px;
		left: 50%;
	}
	.bottom {
		bottom: 16px;
		left: 0;
		right: 0;
		display: flex;
		justify-content: center;
	}
	.bottom :global(.bar),
	.reopen {
		pointer-events: auto;
	}
	.reopen {
		display: grid;
		place-items: center;
		flex: none;
		width: 28px;
		height: 28px;
		padding: 0;
		color: var(--c-text-2);
		cursor: pointer;
	}
	.reopen:hover {
		color: var(--c-text);
		background: var(--glass-strong);
	}
</style>

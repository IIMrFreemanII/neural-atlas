<script lang="ts">
	import { untrack } from 'svelte';
	import type { Tone } from '../../gpu/types.ts';
	import type { LessonDef, LessonMeta, Params, Track } from '../../lessons/types.ts';
	import { lessonKey } from '../../content/curriculum.ts';
	import { initialTexts, questionControls, questionParams } from '../../lessons/questions.ts';
	import { pageLegend, pageReadouts } from '../../lessons/chips.ts';
	import { progress } from '../../state/progress.svelte.ts';
	import { settings } from '../../state/settings.svelte.ts';
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

	const S = untrack(() => lesson.steps.length);
	const Q = untrack(() => lesson.questions?.length ?? 0);
	/** Page i is step i for i < S, else question i - S. */
	const targetFor = (i: number): Params =>
		i < S
			? { ...lesson.defaults, ...lesson.steps[i].params }
			: questionParams(lesson, lesson.questions![i - S]);
	const clamp = (i: number) => Math.min(Math.max(0, Math.floor(i) || 0), S + Q - 1);

	let stepIndex = $state(untrack(() => clamp(progress.getStep(key))));
	let params = $state.raw<Params>(untrack(() => targetFor(stepIndex)));
	let highlight = $state<Tone | null>(null);
	let readoutValues = $state<Record<string, number | string>>({});
	let stage: { command(id: string): void } | undefined = $state();
	/** Current value of each 'text' control, by id. Kept here so it survives steps that hide the control. */
	let texts = $state<Record<string, string>>({});

	function setText(id: string, value: string) {
		texts = { ...texts, [id]: value };
		stage?.command(`${id}:${value}`);
	}
	let collapsed = $state(false);

	// The Stage is re-created when the motion setting changes, so its scene starts from the lesson's
	// initial text again and has emitted nothing yet.
	$effect(() => {
		void settings.reducedMotion;
		texts = {};
		readoutValues = {};
	});

	if (untrack(() => stepIndex >= S - 1)) progress.complete(key);

	// Tweening
	let tween: { start: number; items: Map<string, [number, number]> } | null = null;
	/** Mirrors `tween !== null`, so the task goal can wait for the tween to end. */
	let tweening = $state(false);
	let raf = 0;

	const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

	function stopTween() {
		cancelAnimationFrame(raf);
		raf = 0;
		tween = null;
		tweening = false;
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
			tweening = false;
			raf = 0;
		} else {
			raf = requestAnimationFrame(tick);
		}
	}

	function applyStep(i: number) {
		stopTween();
		const target = targetFor(i);
		const instant = settings.reducedMotion;
		const items = new Map<string, [number, number]>();
		const next: Params = { ...params };
		for (const [k, v] of Object.entries(target)) {
			if (instant || discrete.has(k) || next[k] === undefined || next[k] === v) next[k] = v;
			else items.set(k, [next[k], v]);
		}
		params = next;
		if (items.size) {
			tween = { start: performance.now(), items };
			tweening = true;
			raf = requestAnimationFrame(tick);
		}
	}

	function goTo(i: number) {
		const n = clamp(i);
		if (n === stepIndex) return;
		stepIndex = n;
		progress.setStep(key, n);
		if (n >= S - 1) progress.complete(key);
		applyStep(n);
	}

	function setParam(k: string, v: number) {
		tween?.items.delete(k);
		params = { ...params, [k]: v };
	}

	$effect(() => stopTween);

	/** Scene -> UI values. A key of the form 'param:<key>' writes that lesson param (so its slider follows the scene). */
	function onreadout(k: string, v: number | string) {
		if (k.startsWith('param:')) {
			const pk = k.slice(6);
			if (typeof v === 'number' && Number.isFinite(v) && params[pk] !== v) setParam(pk, v);
			return;
		}
		readoutValues = { ...readoutValues, [k]: v };
	}

	const question = $derived(stepIndex >= S ? lesson.questions![stepIndex - S] : null);
	const visibleControls = $derived(
		question
			? questionControls(lesson, question)
			: lesson.steps[stepIndex].controls === undefined
				? lesson.controls
				: lesson.controls.filter((c) =>
						lesson.steps[stepIndex].controls!.includes('id' in c ? c.id : c.key)
					)
	);

	const page = $derived(question ?? lesson.steps[stepIndex]);
	const visibleLegend = $derived(pageLegend(lesson, page));
	const visibleReadouts = $derived(pageReadouts(lesson, page));

	const solved = $derived(
		(lesson.questions ?? []).map((_, j) => progress.isSolved(key, j))
	);

	// Task questions: solved the first time the goal holds after the learner touched the stage.
	let touched = $state(false);
	$effect(() => {
		void stepIndex;
		touched = false;
	});
	let stageCell: HTMLDivElement | undefined = $state();
	$effect(() => {
		const el = stageCell;
		if (!el) return;
		const touch = () => {
			touched = true;
		};
		el.addEventListener('pointerdown', touch, true);
		el.addEventListener('keydown', touch, true);
		return () => {
			el.removeEventListener('pointerdown', touch, true);
			el.removeEventListener('keydown', touch, true);
		};
	});
	/**
	 * The goal has to hold this long before it counts, so dragging a slider across the target on the way
	 * to somewhere else does not solve the task.
	 */
	const HOLD_MS = 400;
	let hold: ReturnType<typeof setTimeout> | undefined;
	$effect(() => {
		const q = question;
		const j = stepIndex - S;
		let reached = false;
		if (q && q.kind === 'task' && touched && !tweening && !untrack(() => progress.isSolved(key, j))) {
			try {
				reached = q.goal({ params, readouts: readoutValues, texts: { ...initialTexts(lesson), ...texts } });
			} catch {
				reached = false;
			}
		}
		if (!reached) {
			clearTimeout(hold);
			hold = undefined;
		} else if (hold === undefined) {
			hold = setTimeout(() => {
				hold = undefined;
				progress.solve(key, j);
			}, HOLD_MS);
		}
	});
	$effect(() => () => clearTimeout(hold));

	/** The action the space bar presses: the first visible one with a play icon. */
	const mainAction = $derived(
		visibleControls.find((c) => c.kind === 'action' && c.icon === 'play') as
			| Extract<(typeof visibleControls)[number], { kind: 'action' }>
			| undefined
	);

	/**
	 * Lesson shortcuts. Left and right arrows change the step unless the focused element uses them itself
	 * (a slider or a text field). Space presses the main action unless the focused element is one that
	 * space already activates or types into.
	 */
	function onkeydown(e: KeyboardEvent) {
		if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
		const t = e.target as HTMLElement | null;
		if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
			if (t?.closest('input, select, textarea, [contenteditable]')) return;
			e.preventDefault();
			goTo(stepIndex + (e.key === 'ArrowRight' ? 1 : -1));
		} else if (e.key === ' ') {
			if (!mainAction) return;
			if (t?.closest('input:not([type="range"]), select, textarea, button, a, summary, [contenteditable]'))
				return;
			e.preventDefault();
			touched = true;
			stage?.command(mainAction.id);
		}
	}

</script>

<svelte:window {onkeydown} />

<a class="skip-link" href="#lesson-stage">Skip to the lesson</a>

<div class="layout" class:collapsed>
	<div class="side">
		<Sidebar currentKey={key} onclose={() => (collapsed = true)} />
	</div>

	<div class="stage-cell" id="lesson-stage" tabindex="-1" bind:this={stageCell}>
		{#key settings.reducedMotion}
			<Stage
				bind:this={stage}
				createScene={lesson.createScene}
				{params}
				{highlight}
				onreadout={onreadout}
			/>
		{/key}
		<div class="top">
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
				<Legend items={visibleLegend} />
			</div>
			<div class="tr"><Readouts readouts={visibleReadouts} values={readoutValues} /></div>
		</div>
		{#if visibleControls.length}
			<div class="bottom">
				<ControlBar
					controls={visibleControls}
					{params}
					onparam={setParam}
					{texts}
					ontext={setText}
					onaction={(id) => stage?.command(id)}
					spaceAction={mainAction?.id}
				/>
			</div>
		{/if}
	</div>

	<Narration
		{lesson}
		{track}
		{next}
		{stepIndex}
		{solved}
		onsolve={(j) => progress.solve(key, j)}
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
		outline: none;
		position: relative;
		min-height: 0;
		overflow: hidden;
		background: var(--bg-void);
	}
	.top,
	.bottom {
		position: absolute;
		z-index: 2;
		pointer-events: none;
	}
	/* Legend on the left, readouts on the right, each as wide as its chips; they wrap only when the two meet. */
	.top {
		top: 12px;
		left: 12px;
		right: 12px;
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 12px;
	}
	.tl {
		display: flex;
		align-items: flex-start;
		gap: 8px;
		flex: 0 1 auto;
		min-width: 0;
	}
	.tr {
		flex: 0 1 auto;
		min-width: 0;
		margin-left: auto;
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

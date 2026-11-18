<script lang="ts">
	import { untrack } from 'svelte';
	import { page } from '$app/state';
	import type { Insets, Tone } from '../../gpu/types.ts';
	import type { LessonDef, LessonMeta, Params, Track } from '../../lessons/types.ts';
	import { lessonKey } from '../../content/curriculum.ts';
	import { initialTexts, questionControls, questionParams } from '../../lessons/questions.ts';
	import { pageLegend, pageReadouts } from '../../lessons/chips.ts';
	import { progress } from '../../state/progress.svelte.ts';
	import { settings } from '../../state/settings.svelte.ts';
	import { ui } from '../../state/ui.svelte.ts';
	import ControlBar from './ControlBar.svelte';
	import Legend from './Legend.svelte';
	import Narration from './Narration.svelte';
	import Readouts from './Readouts.svelte';
	import Stage from './Stage.svelte';
	import ViewChip from './ViewChip.svelte';

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
	/** The pages are the steps, then the questions; one more virtual page after them says the lesson is complete. */
	const clampPage = (i: number) => Math.min(Math.max(0, Math.floor(i) || 0), S + Q - 1);
	const clamp = (i: number) => Math.min(Math.max(0, Math.floor(i) || 0), S + Q);

	/** A `?step=N` link opens step N; otherwise the learner's saved place. */
	function startPage(): number {
		const raw = page.url.searchParams.get('step');
		if (raw !== null && /^\d+$/.test(raw)) {
			const n = Number(raw);
			if (n >= 1 && n <= S) return n - 1;
		}
		return clampPage(progress.getStep(key));
	}

	let stepIndex = $state(untrack(startPage));
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
	/** The learner has turned a 3D figure away from the step's own view (reported by the scene as 'view:moved'). */
	let viewMoved = $state(false);
	/** The figure can be turned by dragging and has not been yet ('view:drag'). */
	let viewDrag = $state(false);

	// The Stage is re-created when the motion setting or the theme changes, so its scene starts from the
	// lesson's initial text again and has emitted nothing yet.
	const stageKey = $derived(`${settings.reducedMotion}|${settings.theme}`);
	$effect(() => {
		void stageKey;
		texts = {};
		readoutValues = {};
		viewMoved = false;
		viewDrag = false;
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
		highlight = null;
		viewMoved = false;
		viewDrag = false;
		progress.setStep(key, Math.min(n, S + Q - 1));
		if (n >= S - 1) progress.complete(key);
		// The complete page keeps the last page's figure.
		if (n < S + Q) applyStep(n);
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
		if (k === 'view:moved') {
			viewMoved = Number(v) === 1;
			return;
		}
		if (k === 'view:drag') {
			viewDrag = Number(v) === 1;
			return;
		}
		readoutValues = { ...readoutValues, [k]: v };
	}

	const complete = $derived(stepIndex >= S + Q);
	/** The page whose figure is on the stage: the open page, or the last one when the lesson is complete. */
	const pageIdx = $derived(Math.min(stepIndex, S + Q - 1));
	const question = $derived(pageIdx >= S ? lesson.questions![pageIdx - S] : null);
	const visibleControls = $derived(
		complete
			? []
			: question
				? questionControls(lesson, question)
				: lesson.steps[pageIdx].controls === undefined
					? lesson.controls
					: lesson.controls.filter((c) =>
							lesson.steps[pageIdx].controls!.includes('id' in c ? c.id : c.key)
						)
	);

	const current = $derived(question ?? lesson.steps[pageIdx]);
	const visibleLegend = $derived(complete ? [] : pageLegend(lesson, current));
	const visibleReadouts = $derived(complete ? [] : pageReadouts(lesson, current));
	/** Steps whose scene draws its own grid turn the paper grid off. Questions and the complete page use the last step's value. */
	const gridOn = $derived(lesson.steps[Math.min(pageIdx, S - 1)].grid !== false);

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

	// Parts of the stage covered by the floating chips, the control bar and the card (or its tab).
	const CARD_W = 380;
	const TAB_W = 44;
	const side = $derived((ui.cardFolded ? TAB_W : CARD_W) + 32);
	let legendH = $state(0);
	let readoutsH = $state(0);
	let barH = $state(0);
	const insets: Insets = $derived({
		top: Math.max(legendH, readoutsH) + 24,
		right: side,
		bottom: barH + 32,
		left: 24
	});
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
		const q = complete ? null : question;
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

<div class="area" style:--side="{side}px">
	<div class="paper" aria-hidden="true"></div>
	<div class="grid" class:off={!gridOn} class:still={settings.reducedMotion} aria-hidden="true"></div>

	<div class="stage-cell" id="lesson-stage" tabindex="-1" bind:this={stageCell}>
		{#key stageKey}
			<Stage
				bind:this={stage}
				createScene={lesson.createScene}
				{params}
				{highlight}
				{insets}
				onreadout={onreadout}
			/>
		{/key}
		<div class="tl" bind:clientHeight={legendH}>
			<Legend items={visibleLegend} {highlight} onhighlight={(t) => (highlight = t)} />
		</div>
		<div class="tr" bind:clientHeight={readoutsH}>
			<Readouts readouts={visibleReadouts} values={readoutValues} />
		</div>
		<div class="vc">
			<ViewChip
				moved={viewMoved && !complete}
				drag={viewDrag && !complete}
				onreset={() => stage?.command('view:reset')}
			/>
		</div>
		{#if visibleControls.length}
			<div class="bottom" bind:clientHeight={barH}>
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
		{highlight}
		folded={ui.cardFolded}
		onfold={(b) => ui.setCardFolded(b)}
		onsolve={(j) => progress.solve(key, j)}
		onstep={goTo}
		onhighlight={(t) => (highlight = t)}
	/>
</div>

<style>
	/* Desktop-only: one positioned area below the top bar. The stage fills it; everything else floats on it. */
	.area {
		position: relative;
		height: calc(100vh - var(--topbar-h));
		overflow: hidden;
	}
	.paper,
	.grid,
	.stage-cell {
		position: absolute;
		inset: 0;
	}
	.paper {
		background-color: var(--pg);
		background-image: var(--grain);
	}
	/* The paper grid; steps whose scene draws its own grid fade it out. */
	.grid {
		background-image:
			linear-gradient(var(--grid-major) 1px, transparent 1px),
			linear-gradient(90deg, var(--grid-major) 1px, transparent 1px),
			linear-gradient(var(--grid-minor) 1px, transparent 1px),
			linear-gradient(90deg, var(--grid-minor) 1px, transparent 1px);
		background-size:
			120px 120px,
			120px 120px,
			24px 24px,
			24px 24px;
		transition: opacity 0.4s;
	}
	.grid.off {
		opacity: 0;
	}
	.grid.still {
		transition: none;
	}
	.stage-cell {
		outline: none;
		overflow: hidden;
	}
	.tl,
	.tr,
	.vc,
	.bottom {
		position: absolute;
		z-index: 2;
		pointer-events: none;
	}
	.tl {
		top: 16px;
		left: 16px;
	}
	.tl :global(button) {
		pointer-events: auto;
	}
	/* Readouts sit left of the card (or its tab). */
	.tr {
		top: 16px;
		right: var(--side);
	}
	/* The camera chip and the drag hint share one spot, centred in the part of the stage the card leaves free. */
	.vc {
		top: 16px;
		left: 0;
		right: var(--side);
		display: flex;
		justify-content: center;
		align-items: flex-start;
	}
	.vc :global(.drag-hint) {
		margin-top: 6px;
	}
	.bottom {
		bottom: 16px;
		left: 24px;
		right: var(--side);
	}
</style>

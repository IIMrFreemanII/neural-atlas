<script lang="ts">
	import { untrack } from 'svelte';
	import type { Insets, Tone } from '#lib/gpu/types.ts';
	import lesson from '#lib/lessons/fundamentals/train-a-network/lesson.ts';
	import type { Control, Params } from '#lib/lessons/types.ts';
	import { settings } from '#lib/state/settings.svelte.ts';
	import ControlBar from '#lib/components/lesson/ControlBar.svelte';
	import Legend from '#lib/components/lesson/Legend.svelte';
	import Readouts from '#lib/components/lesson/Readouts.svelte';
	import Stage from '#lib/components/lesson/Stage.svelte';
	import { LossTrace } from '#lib/components/playground/loss-trace.ts';
	import SetupCard from '#lib/components/playground/SetupCard.svelte';

	let params = $state.raw<Params>({ ...lesson.defaults, scenario: 6, free: 1 });
	let highlight = $state<Tone | null>(null);
	let readoutValues = $state<Record<string, number | string>>({});
	let running = $state(false);
	let stage: { command(id: string): void } | undefined = $state();

	// The stage no longer draws the loss chart in free mode, so the legend drops its Loss item.
	const legend = (lesson.legend ?? []).filter((i) => i.label !== 'Loss');

	// The three action controls, in order. While training runs, the main one says Pause.
	const actions = $derived(
		(['run', 'step', 'reset'] as const).map((id) => {
			const c = lesson.controls.find((x) => x.kind === 'action' && x.id === id) as Extract<Control, { kind: 'action' }>;
			return id === 'run' && running ? { ...c, label: 'Pause' } : c;
		})
	);

	// The loss curve. LossTrace is plain data, so a version counter tells the card to redraw.
	const trace = new LossTrace();
	let traceVersion = $state(0);
	let latestLoss = NaN;
	let latestSteps = 0;
	function record() {
		if (!Number.isFinite(latestLoss)) return;
		trace.add(latestSteps, latestLoss);
		traceVersion++;
	}

	function onreadout(k: string, v: number | string) {
		if (k === 'running') {
			running = Number(v) === 1;
			return;
		}
		readoutValues = { ...readoutValues, [k]: v };
		if (k === 'loss') {
			latestLoss = Number(v);
			record();
		} else if (k === 'steps') {
			latestSteps = Number(v);
			record();
		}
	}

	// The Stage is re-created when the motion setting or the theme changes; its scene starts over.
	const stageKey = $derived(`${settings.reducedMotion}|${settings.theme}`);
	$effect(() => {
		void stageKey;
		untrack(() => {
			readoutValues = {};
			running = false;
			latestLoss = NaN;
			latestSteps = 0;
			trace.clear();
			traceVersion++;
		});
	});

	function setParam(k: string, v: number) {
		params = { ...params, [k]: v };
	}

	let legendH = $state(0);
	let readoutsH = $state(0);
	let barH = $state(0);
	const insets: Insets = $derived({
		top: Math.max(legendH, readoutsH) + 24,
		right: 380 + 32,
		bottom: barH + 32,
		left: 24
	});

	/** Space presses Train/Pause unless the focused element is one that space already activates or types into. */
	function onkeydown(e: KeyboardEvent) {
		if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey || e.key !== ' ') return;
		const t = e.target as HTMLElement | null;
		if (t?.closest('input:not([type="range"]), select, textarea, button, a, summary, [contenteditable]')) return;
		e.preventDefault();
		stage?.command('run');
	}
</script>

<svelte:head><title>Playground · Neural Atlas</title></svelte:head>
<svelte:window {onkeydown} />

<a class="skip-link" href="#playground-stage">Skip to the playground</a>

<div class="area">
	<div class="paper" aria-hidden="true"></div>
	<div class="grid" class:still={settings.reducedMotion} aria-hidden="true"></div>

	<div class="stage-cell" id="playground-stage" tabindex="-1">
		{#key stageKey}
			<Stage bind:this={stage} createScene={lesson.createScene} {params} {highlight} {insets} {onreadout} />
		{/key}
		<div class="tl" bind:clientHeight={legendH}>
			<Legend items={legend} {highlight} onhighlight={(t) => (highlight = t)} />
		</div>
		<div class="tr" bind:clientHeight={readoutsH}>
			<Readouts readouts={lesson.readouts ?? []} values={readoutValues} />
		</div>
		<div class="bottom" bind:clientHeight={barH}>
			<ControlBar
				controls={actions}
				{params}
				onparam={setParam}
				onaction={(id) => stage?.command(id)}
				spaceAction="run"
			/>
		</div>
	</div>

	<SetupCard {params} onparam={setParam} {trace} {traceVersion} />
</div>

<style>
	/* Desktop-only, like a lesson page: the stage fills the area below the top bar; everything else floats on it. */
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
	}
	.stage-cell {
		outline: none;
		overflow: hidden;
	}
	.tl,
	.tr,
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
	/* Readouts sit left of the card. */
	.tr {
		top: 16px;
		right: calc(380px + 32px);
	}
	.bottom {
		bottom: 16px;
		left: 24px;
		right: calc(380px + 32px);
	}
</style>

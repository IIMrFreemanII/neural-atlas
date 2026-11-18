<script lang="ts">
	import type { LossTrace } from './loss-trace.ts';

	interface Props {
		trace: LossTrace;
		/** Bumped by the owner after every change to the trace, so the curve redraws. */
		version?: number;
	}

	let { trace, version = 0 }: Props = $props();

	const W = 300;
	/** Nothing to draw until training has taken a step (the start reports step 0 for both loss and steps). */
	const empty = $derived((void version, trace.points.length < 2 || trace.points[trace.points.length - 1][0] === 0));
	const d = $derived.by(() => {
		void version;
		const pts = trace.points;
		if (pts.length < 2) return '';
		const lastSteps = pts[pts.length - 1][0];
		const max = trace.max > 0 ? trace.max : 1;
		return pts
			.map(([s, l], i) => {
				const x = lastSteps > 0 ? (s / lastSteps) * W : 0;
				const y = 66 - (l / max) * 60;
				return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
			})
			.join(' ');
	});
</script>

<div class="spark-wrap">
	<span class="sr-only">Loss curve</span>
	<svg class="spark" viewBox="0 0 300 70" preserveAspectRatio="none" aria-hidden="true">
		{#if !empty}
			<path {d} fill="none" stroke="var(--grad)" stroke-width="2" vector-effect="non-scaling-stroke" />
		{/if}
		<line x1="0" y1="69" x2="300" y2="69" stroke="var(--rule-2)" vector-effect="non-scaling-stroke" />
	</svg>
	{#if empty}<span class="hint">Press Train</span>{/if}
</div>

<style>
	.spark-wrap {
		position: relative;
	}
	.spark {
		display: block;
		width: 100%;
		height: 70px;
	}
	.hint {
		position: absolute;
		inset: 0;
		display: grid;
		place-items: center;
		font-size: 12px;
		color: var(--ink-3);
		pointer-events: none;
	}
	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		overflow: hidden;
		clip-path: inset(50%);
		white-space: nowrap;
	}
</style>

<script lang="ts">
	import { resolve } from '$app/paths';
	import { makeDataset, DATASET_NAMES, type DatasetId } from '../../ml/datasets.ts';
	import lesson from '../../lessons/fundamentals/train-a-network/lesson.ts';
	import type { Control, Params } from '../../lessons/types.ts';
	import Slider from '../lesson/Slider.svelte';
	import LossSpark from './LossSpark.svelte';
	import type { LossTrace } from './loss-trace.ts';

	type SelectControl = Extract<Control, { kind: 'select' }>;
	type SliderControl = Extract<Control, { kind: 'slider' }>;

	interface Props {
		params: Params;
		onparam: (key: string, value: number) => void;
		trace: LossTrace;
		/** Bumped after every change to `trace`. */
		traceVersion?: number;
	}

	let { params, onparam, trace, traceVersion = 0 }: Props = $props();

	/** The select options and slider ranges come from the lesson, so they cannot drift. */
	const controls = lesson.controls;
	const select = (key: string) => controls.find((c): c is SelectControl => c.kind === 'select' && c.key === key)!;
	const slider = (key: string) => controls.find((c): c is SliderControl => c.kind === 'slider' && c.key === key)!;
	const rows = [select('layers'), select('hidden'), select('act')];
	const lr = slider('lr');
	const noise = slider('noise');

	const IDS: DatasetId[] = [0, 1, 2, 3];
	const EXTENT = 1.4;
	/** The real data behind each tile, mapped from [-EXTENT, EXTENT]^2 into the 40x40 box (y up). */
	const thumbs = IDS.map((id) => {
		const d = makeDataset(id, 80, 0.05, 11);
		return Array.from({ length: d.n }, (_, k) => ({
			x: ((d.x[k * 2] + EXTENT) / (2 * EXTENT)) * 40,
			y: 40 - ((d.x[k * 2 + 1] + EXTENT) / (2 * EXTENT)) * 40,
			pos: d.y[k] >= 0.5
		}));
	});

	const lessonHref = resolve('/learn/[track]/[lesson]', { track: 'fundamentals', lesson: 'train-a-network' });
</script>

<aside class="card" aria-label="Playground setup">
	<div class="scroll">
		<h1>Playground</h1>
		<p class="lead">Pick a pattern and a network, then train it. Nothing here is graded.</p>

		<div class="cseg" role="group" aria-label="Data">
			<span class="lab">Data</span>
			<div class="tiles">
				{#each IDS as id (id)}
					<button
						type="button"
						class="tile"
						class:on={params.dataset === id}
						aria-pressed={params.dataset === id}
						onclick={() => onparam('dataset', id)}
					>
						<svg viewBox="0 0 40 40" aria-hidden="true">
							{#each thumbs[id] as p, i (i)}
								<circle cx={p.x} cy={p.y} r="1.6" fill={p.pos ? 'var(--data)' : 'var(--neg)'} />
							{/each}
						</svg>
						<span>{DATASET_NAMES[id]}</span>
					</button>
				{/each}
			</div>
		</div>

		{#each rows as c (c.key)}
			<div class="field-row" role="group" aria-label={c.label}>
				<span>{c.label}</span>
				<div class="seg">
					{#each c.options as o (o.value)}
						<button type="button" aria-pressed={params[c.key] === o.value} onclick={() => onparam(c.key, o.value)}
							>{o.label}</button
						>
					{/each}
				</div>
			</div>
		{/each}

		<div class="wide">
			<Slider control={lr} value={params.lr ?? lr.min} onchange={(v) => onparam('lr', v)} />
			<Slider control={noise} value={params.noise ?? noise.min} onchange={(v) => onparam('noise', v)} />
		</div>

		<div class="cseg">
			<span class="lab">Loss while training</span>
			<LossSpark {trace} version={traceVersion} />
		</div>
	</div>
	<footer>
		<span class="lab">Free play from Train a network</span>
		<a class="btn primary" href={lessonHref}>Open the lesson</a>
	</footer>
</aside>

<style>
	/* The shared slider is sized for the control bar; in the card it spans the full width. */
	.wide {
		display: grid;
		gap: 14px;
	}
	.wide :global(.slider) {
		width: 100%;
	}
	/* Same frame as the lesson narration card: floats 16px from the top, right and bottom. */
	.card {
		position: absolute;
		top: 16px;
		right: 16px;
		bottom: 16px;
		z-index: 3;
		display: grid;
		grid-template-rows: minmax(0, 1fr) auto;
		width: 380px;
		overflow: hidden;
		border: 1px solid var(--rule);
		border-radius: 6px;
		background-color: var(--sheet);
		background-image: var(--grain);
		box-shadow: var(--shadow);
	}
	.scroll {
		display: grid;
		align-content: start;
		gap: 18px;
		min-height: 0;
		overflow-y: auto;
		padding: 22px 24px 24px;
	}
	h1 {
		margin: 0;
		font-family: var(--font-display);
		font-weight: 500;
		font-size: 26px;
		line-height: 1.15;
		letter-spacing: -0.01em;
	}
	.lead {
		margin: -6px 0 0;
		font-size: 14.5px;
		color: var(--ink-2);
	}
	.lab {
		font-size: 12px;
		color: var(--ink-2);
		font-weight: 400;
	}
	.cseg {
		display: grid;
		gap: 5px;
	}
	.tiles {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 8px;
	}
	.tile {
		display: grid;
		gap: 4px;
		padding: 0;
		border: 0;
		background: transparent;
		font: inherit;
		font-size: 12px;
		text-align: center;
		color: var(--ink-2);
		cursor: pointer;
	}
	.tile svg {
		display: block;
		width: 100%;
		aspect-ratio: 1;
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		background: var(--pg);
	}
	.tile:hover {
		color: var(--ink);
	}
	.tile.on svg {
		border-color: var(--ink);
		box-shadow: 0 0 0 1.5px var(--ink);
	}
	.tile.on {
		color: var(--ink);
		font-weight: 500;
	}
	.field-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 12px;
		font-size: 13.5px;
	}
	.seg {
		display: inline-flex;
		width: max-content;
		overflow: hidden;
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		background: var(--sheet);
	}
	.seg button {
		padding: 4px 10px;
		border: 0;
		border-radius: 0;
		background: transparent;
		font-size: 12.5px;
		color: var(--ink-2);
		white-space: nowrap;
		cursor: pointer;
	}
	.seg button + button {
		border-left: 1px solid var(--rule);
	}
	.seg button:hover {
		color: var(--ink);
	}
	.seg button[aria-pressed='true'] {
		background: var(--ink);
		color: var(--pg);
	}
	footer {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		padding: 14px 20px;
		border-top: 1px solid var(--rule);
	}
</style>

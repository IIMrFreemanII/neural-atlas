<script lang="ts">
	import type { Tone } from '../../gpu/types.ts';
	import type { Question } from '../../lessons/types.ts';
	import Icon from '../shell/Icon.svelte';
	import { parseBody } from '../../lessons/markup.ts';
	import Prose from './Prose.svelte';

	interface Props {
		question: Question;
		solved: boolean;
		onsolve: () => void;
		onhighlight: (tone: Tone | null) => void;
		highlight?: Tone | null;
	}

	let { question, solved, onsolve, onhighlight, highlight = null }: Props = $props();

	// The card is re-created per question, so the starting values are read once on purpose.
	/** Indices of the wrong options the learner has tried. Component state only. */
	let tried = $state<number[]>([]);
	/** Index of the correct option once picked (or, for a card opened on a solved question, found). */
	let picked = $state<number | null>(
		(() => {
			if (!solved || question.kind !== 'choice') return null;
			const i = question.options.findIndex((o) => o.correct === true);
			return i === -1 ? null : i;
		})()
	);
	let hintOpen = $state(false);

	const done = $derived(question.kind === 'choice' ? picked !== null : solved);

	function pick(i: number) {
		if (question.kind !== 'choice' || done || tried.includes(i)) return;
		if (question.options[i].correct === true) {
			picked = i;
			onsolve();
		} else {
			tried = [...tried, i];
		}
	}
</script>

{#snippet inline(text: string)}
	{#each parseBody(text) as para, i (i)}{#if i > 0}{' '}{/if}{#each para as run, j (j)}{#if run.kind === 'bold'}<strong
					>{run.text}</strong
				>{:else if run.kind === 'term'}<span
					class="t"
					class:lit={highlight === run.tone}
					style:--tone="var(--c-{run.tone})">{run.text}</span
				>{:else}{run.text}{/if}{/each}{/each}
{/snippet}

<section class="question" data-question={question.kind} data-solved={solved}>
	<h2>{question.kind === 'choice' ? 'Pick an answer' : 'Do it in the visual'}</h2>
	<Prose text={question.prompt} {onhighlight} {highlight} />

	{#if question.kind === 'choice'}
		<ul class="options">
			{#each question.options as o, i (i)}
				{@const right = picked === i}
				{@const wrong = tried.includes(i)}
				<li class="opt" class:right class:wrong class:dim={done && !right && !wrong}>
					<button
						type="button"
						class="option"
						data-option={i}
						disabled={done || wrong}
						onclick={() => pick(i)}
					>
						<span class="dot" aria-hidden="true">
							{#if right}<Icon name="check" size={10} />{:else if wrong}<Icon name="close" size={10} />{/if}
						</span>
						<span class="text">{@render inline(o.text)}</span>
					</button>
					<div class="whys" aria-live="polite">
						{#if right || wrong}
							<div class="whybox">
								<span class="verdict" class:no={wrong}>{right ? 'Correct.' : 'Not quite.'}</span>
								<p class="why" role="status">{@render inline(o.why)}</p>
							</div>
						{/if}
					</div>
				</li>
			{/each}
		</ul>
	{:else}
		<div class="goal" class:done={solved}>
			<span class="status" role="status">
				<span class="ring" class:ok={solved} aria-hidden="true">
					{#if solved}<Icon name="check" size={12} />{/if}
				</span>
				<span class="state" class:ok={solved}>
					{solved ? 'Done' : 'Not done yet. Work in the visual and with its controls.'}
				</span>
			</span>
		</div>
		{#if solved}
			<div class="explain"><Prose text={question.solved} {onhighlight} {highlight} /></div>
		{:else if hintOpen}
			<div class="hint"><Prose text={question.hint} {onhighlight} {highlight} /></div>
		{:else}
			<button type="button" class="btn hint-btn" onclick={() => (hintOpen = true)}>Show a hint</button>
		{/if}
	{/if}
</section>

<style>
	h2 {
		margin: 0 0 12px;
		font-family: var(--font-display);
		font-weight: 500;
		font-size: 26px;
		line-height: 1.15;
		letter-spacing: -0.01em;
		text-wrap: balance;
	}
	.options {
		display: grid;
		gap: 6px;
		margin: 16px 0 0;
		padding: 0;
		list-style: none;
	}
	.opt {
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		font-size: 14px;
		line-height: 1.45;
		transition:
			border-color 0.15s,
			background-color 0.15s,
			opacity 0.15s;
	}
	.opt:has(.option:hover:not(:disabled)) {
		border-color: var(--ink-2);
		background: color-mix(in srgb, var(--ink) 4%, transparent);
	}
	.opt.right {
		border-color: var(--ok);
		background: color-mix(in srgb, var(--ok) 9%, transparent);
	}
	.opt.wrong {
		border-color: var(--neg);
		background: color-mix(in srgb, var(--neg) 8%, transparent);
	}
	.opt.dim {
		opacity: 0.5;
	}
	.option {
		display: grid;
		grid-template-columns: 14px 1fr;
		align-items: start;
		gap: 10px;
		width: 100%;
		padding: 9px 12px;
		border: 0;
		border-radius: 4px;
		background: transparent;
		font: inherit;
		color: var(--ink);
		text-align: left;
		cursor: pointer;
	}
	.option:disabled {
		cursor: default;
	}
	.dot {
		display: grid;
		place-items: center;
		width: 14px;
		height: 14px;
		margin-top: 2px;
		border: 1.5px solid var(--rule-2);
		border-radius: 50%;
		color: var(--sheet);
	}
	.right .dot {
		background: var(--ok);
		border-color: var(--ok);
	}
	.wrong .dot {
		background: var(--neg);
		border-color: var(--neg);
	}
	.text {
		min-width: 0;
	}
	.text strong,
	.why strong {
		font-weight: 600;
	}
	.t {
		color: var(--tone);
		font-weight: 500;
		border-radius: 2px;
	}
	.t.lit {
		background: color-mix(in srgb, var(--tone) 14%, transparent);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--tone) 14%, transparent);
	}
	.whybox {
		padding: 0 12px 9px 36px;
		font-size: 13px;
		line-height: 1.5;
		color: var(--ink-2);
	}
	.why {
		margin: 0;
	}
	.verdict {
		font-weight: 600;
		color: var(--ok);
	}
	.verdict.no {
		color: var(--neg);
	}
	.goal {
		display: flex;
		align-items: center;
		margin: 16px 0 12px;
		padding: 10px 12px;
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		transition:
			border-color 0.15s,
			background-color 0.15s;
	}
	.goal.done {
		border-color: var(--ok);
		background: color-mix(in srgb, var(--ok) 9%, transparent);
	}
	.status {
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.ring {
		display: inline-grid;
		place-items: center;
		flex: none;
		width: 20px;
		height: 20px;
		border: 1.5px solid var(--rule-2);
		border-radius: 50%;
		color: var(--sheet);
		transition:
			background-color 0.15s,
			border-color 0.15s;
	}
	.ring.ok {
		background: var(--ok);
		border-color: var(--ok);
	}
	.state {
		font-size: 12.5px;
		font-weight: 500;
		color: var(--grad);
	}
	.state.ok {
		color: var(--ok);
	}
	.hint-btn {
		height: 30px;
		padding: 0 12px;
		font-size: 13px;
	}
	.explain,
	.hint {
		padding-left: 12px;
		border-left: 2px solid var(--ink-3);
		font-size: 13.5px;
	}
	.explain {
		border-left-color: var(--ok);
	}
	.hint :global(p),
	.explain :global(p) {
		font-size: 13.5px;
		line-height: 1.55;
		color: var(--ink-2);
	}
	.hint :global(strong),
	.explain :global(strong) {
		font-weight: 500;
		color: var(--ink);
	}
	.explain :global(p:last-child),
	.hint :global(p:last-child) {
		margin-bottom: 0;
	}
</style>

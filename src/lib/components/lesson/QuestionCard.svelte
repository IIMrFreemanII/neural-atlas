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
	}

	let { question, solved, onsolve, onhighlight }: Props = $props();

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
				>{:else if run.kind === 'term'}<span style:color="var(--c-{run.tone})">{run.text}</span
				>{:else}{run.text}{/if}{/each}{/each}
{/snippet}

<section class="question" data-question={question.kind} data-solved={solved}>
	<h2>{question.kind === 'choice' ? 'Pick an answer' : 'Do it in the visual'}</h2>
	<Prose text={question.prompt} {onhighlight} />

	{#if question.kind === 'choice'}
		<ul class="options">
			{#each question.options as o, i (i)}
				{@const right = picked === i}
				{@const wrong = tried.includes(i)}
				<li>
					<button
						type="button"
						class="option"
						class:right
						class:wrong
						class:dim={done && !right && !wrong}
						data-option={i}
						disabled={done || wrong}
						onclick={() => pick(i)}
					>
						<span class="badge num" aria-hidden="true">
							{#if right}<Icon name="check" size={14} />{:else if wrong}<Icon
									name="close"
									size={14}
								/>{:else}{'ABCD'[i]}{/if}
						</span>
						<span class="text">{@render inline(o.text)}</span>
					</button>
					<div class="whys" aria-live="polite">
						{#if right || wrong}
							<p class="why" class:ok={right} role="status">{@render inline(o.why)}</p>
						{/if}
					</div>
				</li>
			{/each}
		</ul>
	{:else}
		<div class="status" role="status">
			<span class="ring" class:ok={solved} aria-hidden="true">
				{#if solved}<Icon name="check" size={14} />{/if}
			</span>
			<span class="state" class:ok={solved}>
				{solved ? 'Done' : 'Not done yet. Work in the visual and with its controls.'}
			</span>
		</div>
		{#if solved}
			<div class="explain"><Prose text={question.solved} {onhighlight} /></div>
		{:else if hintOpen}
			<div class="hint"><Prose text={question.hint} {onhighlight} /></div>
		{:else}
			<button type="button" class="btn hint-btn" onclick={() => (hintOpen = true)}>Show a hint</button>
		{/if}
	{/if}
</section>

<style>
	h2 {
		font-size: 17px;
		margin-bottom: 12px;
	}
	.options {
		display: flex;
		flex-direction: column;
		gap: 8px;
		margin: 16px 0 0;
		padding: 0;
		list-style: none;
	}
	.option {
		display: flex;
		align-items: flex-start;
		gap: 10px;
		width: 100%;
		padding: 10px 12px;
		border: 1px solid var(--line);
		border-radius: var(--r-md);
		background: var(--glass);
		text-align: left;
		cursor: pointer;
		transition:
			border-color 0.15s,
			background 0.15s,
			opacity 0.15s;
	}
	.option:hover:not(:disabled) {
		border-color: var(--line-strong);
		background: var(--glass-strong);
	}
	.option:disabled {
		cursor: default;
	}
	.option.right {
		border-color: var(--c-ok);
	}
	.option.wrong {
		border-color: var(--c-neg);
	}
	.option.dim {
		opacity: 0.5;
	}
	.badge {
		display: inline-grid;
		place-items: center;
		flex: none;
		width: 22px;
		height: 22px;
		border: 1px solid var(--line-strong);
		border-radius: var(--r-sm);
		font-size: 12px;
		color: var(--c-text-2);
	}
	.right .badge {
		border-color: var(--c-ok);
		color: var(--c-ok);
	}
	.wrong .badge {
		border-color: var(--c-neg);
		color: var(--c-neg);
	}
	.text {
		flex: 1;
		min-width: 0;
		padding-top: 2px;
	}
	.text {
		color: var(--c-text);
	}
	.text strong {
		font-weight: 600;
	}
	.why strong {
		font-weight: 600;
		color: var(--c-text);
	}
	.why {
		margin: 8px 0 4px;
		padding-left: 12px;
		border-left: 2px solid var(--c-neg);
		font-size: 14px;
		color: var(--c-text-2);
	}
	.why.ok {
		border-left-color: var(--c-ok);
	}
	.status {
		display: flex;
		align-items: center;
		gap: 10px;
		margin: 16px 0 12px;
	}
	.ring {
		display: inline-grid;
		place-items: center;
		flex: none;
		width: 22px;
		height: 22px;
		border: 1.5px solid var(--line-strong);
		border-radius: 50%;
		color: var(--bg-void);
		transition:
			background 0.15s,
			border-color 0.15s;
	}
	.ring.ok {
		background: var(--c-ok);
		border-color: var(--c-ok);
	}
	.state {
		font-size: 14px;
		color: var(--c-text-2);
	}
	.state.ok {
		color: var(--c-ok);
		font-weight: 500;
	}
	.hint-btn {
		height: 30px;
		padding: 0 12px;
		font-size: 13px;
	}
	.explain,
	.hint {
		padding-left: 12px;
		border-left: 2px solid var(--line-strong);
	}
	.explain {
		border-left-color: var(--c-ok);
	}
	.explain :global(p:last-child),
	.hint :global(p:last-child) {
		margin-bottom: 0;
	}
</style>

<script lang="ts">
	import { tick } from 'svelte';
	import { fade, fly } from 'svelte/transition';
	import { afterNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import { allLessons, lessonKey, tracks } from '../../content/curriculum.ts';
	import { lessonLink } from '../../link.ts';
	import { progress } from '../../state/progress.svelte.ts';
	import { settings } from '../../state/settings.svelte.ts';
	import { ui } from '../../state/ui.svelte.ts';
	import Icon from './Icon.svelte';
	import QuestionPips from './QuestionPips.svelte';

	const currentKey = $derived(
		page.route.id === '/learn/[track]/[lesson]'
			? lessonKey(page.params.track as never, page.params.lesson ?? '')
			: null
	);
	const ms = $derived(settings.reducedMotion ? 0 : 220);

	let open: Record<string, boolean> = $state({});
	let drawer: HTMLElement | undefined = $state();
	let wasOpen = false;

	/** The track to show open: the current lesson's, else the one with the next unfinished lesson. */
	function startTrack(): string {
		const cur = allLessons.find((l) => lessonKey(l.track, l.id) === currentKey);
		const next = allLessons.find((l) => l.status === 'ready' && !progress.isComplete(lessonKey(l.track, l.id)));
		return (cur ?? next)?.track ?? tracks[0].id;
	}

	$effect(() => {
		const isOpen = ui.contentsOpen;
		if (isOpen) {
			wasOpen = true;
			open = { [startTrack()]: true };
			void tick().then(() => {
				const target =
					drawer?.querySelector<HTMLElement>('a[aria-current="page"]') ??
					drawer?.querySelector<HTMLElement>('a[href]');
				target?.focus();
			});
		} else if (wasOpen) {
			wasOpen = false;
			// Focus goes back to the button, unless another panel took over.
			if (!ui.quickJumpOpen) document.querySelector<HTMLElement>('[data-contents-toggle]')?.focus();
		}
	});

	afterNavigate(() => ui.setContentsOpen(false));

	function onkeydown(e: KeyboardEvent): void {
		if (ui.contentsOpen && e.key === 'Escape') {
			e.preventDefault();
			ui.setContentsOpen(false);
		}
	}
</script>

<svelte:window {onkeydown} />

{#if ui.contentsOpen}
	<div class="scrim" aria-hidden="true" transition:fade={{ duration: ms }} onclick={() => ui.setContentsOpen(false)}></div>
	<aside
		id="contents-drawer"
		class="drawer"
		aria-label="Contents"
		bind:this={drawer}
		transition:fly={{ x: -360, duration: ms, opacity: 1 }}
	>
		<div class="head">
			<h2>Contents</h2>
			<button type="button" class="close" aria-label="Close contents" onclick={() => ui.setContentsOpen(false)}>
				<Icon name="close" size={14} />
			</button>
		</div>
		<div class="body">
			{#each tracks as t, i (t.id)}
				{@const isOpen = open[t.id] === true}
				{@const done = t.lessons.filter((l) => progress.isComplete(lessonKey(l.track, l.id))).length}
				<section class="track" style:--tone="var(--c-{t.tone})">
					<button
						type="button"
						class="track-head"
						aria-expanded={isOpen}
						aria-controls="contents-{t.id}"
						onclick={() => (open[t.id] = !isOpen)}
					>
						<span class="numl">{i + 1}</span>
						<h3>{t.title}</h3>
						<span class="count"
							>{done}/{t.lessons.length}<span class="chev" class:flip={isOpen}><Icon name="chevron-down" size={12} /></span></span
						>
					</button>
					{#if isOpen}
						<ul id="contents-{t.id}" class="lessons">
							{#each t.lessons as l (l.id)}
								{@const key = lessonKey(l.track, l.id)}
								{@const isDone = progress.isComplete(key)}
								<li>
									{#if l.status === 'ready'}
										<a
											class="lesson"
											class:cur={key === currentKey}
											href={lessonLink(l)}
											aria-current={key === currentKey ? 'page' : undefined}
										>
											<span class="mk" class:done={isDone} role="img" aria-label={isDone ? 'Completed' : 'Not completed'}></span>
											<span class="title">{l.title}</span>
											<span class="meta">
												<QuestionPips lessonKey={key} count={l.questions} />
												<span class="min">{l.minutes} min</span>
											</span>
										</a>
									{:else}
										<span class="lesson soon">
											<span class="mk"></span>
											<span class="title">{l.title}</span>
											<span class="meta"><span class="min">Soon</span></span>
										</span>
									{/if}
								</li>
							{/each}
						</ul>
					{/if}
				</section>
			{/each}
		</div>
	</aside>
{/if}

<style>
	.scrim {
		position: fixed;
		inset: var(--topbar-h) 0 0 0;
		z-index: 20;
		background: var(--scrim);
	}
	.drawer {
		position: fixed;
		top: var(--topbar-h);
		left: 0;
		bottom: 0;
		z-index: 21;
		width: 360px;
		display: grid;
		grid-template-rows: auto minmax(0, 1fr);
		overflow: hidden;
		background-color: var(--sheet);
		background-image: var(--grain);
		border-right: 1px solid var(--rule-2);
		box-shadow: var(--shadow);
	}
	.head {
		display: flex;
		align-items: start;
		justify-content: space-between;
		padding: 18px 22px 14px;
		border-bottom: 1px solid var(--rule);
	}
	h2 {
		margin: 0;
		font-family: var(--font-display);
		font-size: 24px;
		font-weight: 500;
		line-height: 1.1;
	}
	.close {
		display: grid;
		place-items: center;
		width: 28px;
		height: 28px;
		padding: 0;
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		background: var(--sheet);
		color: var(--ink-2);
		cursor: pointer;
	}
	.body {
		overflow-y: auto;
		padding: 8px 0;
	}
	.track {
		padding: 10px 22px;
		border-bottom: 1px solid var(--rule);
	}
	.track-head {
		display: grid;
		grid-template-columns: 26px 1fr auto;
		align-items: baseline;
		gap: 6px;
		width: 100%;
		padding: 0;
		border: 0;
		background: transparent;
		color: var(--ink);
		font: inherit;
		text-align: left;
		cursor: pointer;
	}
	.numl {
		font-family: var(--font-display);
		font-style: italic;
		font-size: 20px;
		color: var(--tone);
	}
	h3 {
		margin: 0;
		font-family: var(--font-display);
		font-size: 17px;
		font-weight: 500;
		color: var(--tone);
	}
	.count {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		font-family: var(--font-mono);
		font-size: 11px;
		color: var(--ink-3);
	}
	.chev {
		display: inline-flex;
		transition: transform 0.15s;
	}
	.chev.flip {
		transform: rotate(180deg);
	}
	.lessons {
		display: grid;
		margin: 6px 0 0;
		padding: 0;
		list-style: none;
	}
	.lesson {
		display: grid;
		grid-template-columns: 26px 1fr auto;
		align-items: center;
		gap: 6px;
		padding: 4px 8px 4px 0;
		border-radius: 3px;
		font-size: 13.5px;
		color: var(--ink-2);
	}
	a.lesson:hover {
		color: var(--ink);
	}
	.lesson.cur {
		margin: 0 -8px;
		padding-left: 8px;
		background: color-mix(in srgb, var(--tone) 12%, transparent);
		color: var(--ink);
		font-weight: 500;
	}
	.lesson.soon {
		color: var(--ink-3);
	}
	.mk {
		width: 9px;
		height: 9px;
		margin-left: 6px;
		border: 1.5px solid var(--tone);
		border-radius: 50%;
	}
	.soon .mk {
		border-color: var(--rule-2);
	}
	.mk.done {
		background: var(--tone);
	}
	.meta {
		display: inline-flex;
		align-items: center;
		gap: 10px;
	}
	.min {
		font-size: 12px;
		font-weight: 400;
		color: var(--ink-3);
	}
	:global([data-reduce-motion]) .chev {
		transition: none;
	}
</style>

<script lang="ts">
	import { tick } from 'svelte';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { allLessons, lessonKey, tracks } from '../../content/curriculum.ts';
	import { searchGlossary } from '../../content/glossary.ts';
	import { STEPS } from '../../content/steps-index.ts';
	import { highlight, rankLessons, rankSteps, type StepEntry } from '../../content/search.ts';
	import { lessonLink } from '../../link.ts';
	import { progress } from '../../state/progress.svelte.ts';
	import { ui } from '../../state/ui.svelte.ts';
	import Icon from './Icon.svelte';

	interface Item {
		id: string;
		/** Group heading it belongs to. */
		group: 'Lessons' | 'Steps' | 'Terms';
		title: string;
		sub: string;
		/** Right-hand label. */
		label: string;
		href: string;
		tone: string | null;
	}

	const GROUP_MAX = 6;
	const trackById = new Map(tracks.map((t) => [t.id, t]));
	const lessonByKey = new Map(allLessons.map((l) => [lessonKey(l.track, l.id), l]));
	const stepEntries: StepEntry[] = STEPS.flatMap((s) => {
		const meta = lessonByKey.get(lessonKey(s.track, s.lesson));
		return meta ? [{ track: s.track, lesson: s.lesson, lessonTitle: meta.title, stepIndex: s.step, title: s.title }] : [];
	});

	let query = $state('');
	let active = $state(0);
	let input: HTMLInputElement | undefined = $state();
	let opener: HTMLElement | null = null;

	function lessonItem(l: (typeof allLessons)[number]): Item {
		const t = trackById.get(l.track);
		const n = (t?.lessons.findIndex((x) => x.id === l.id) ?? 0) + 1;
		return {
			id: `lesson-${l.track}-${l.id}`,
			group: 'Lessons',
			title: l.title,
			sub: `${t?.title ?? ''}, lesson ${n}`,
			label: `${l.minutes} min`,
			href: lessonLink(l),
			tone: t ? `var(--c-${t.tone})` : null
		};
	}

	function firstSentence(s: string): string {
		const i = s.indexOf('. ');
		return i === -1 ? s : s.slice(0, i + 1);
	}

	const items = $derived.by((): Item[] => {
		const q = query.trim();
		if (!q) {
			// Nothing typed yet: the next unfinished lesson, then a few more.
			const ready = allLessons.filter((l) => l.status === 'ready');
			const next = ready.find((l) => !progress.isComplete(lessonKey(l.track, l.id))) ?? ready[0];
			const rest = ready.filter((l) => l !== next).slice(0, GROUP_MAX - 1);
			return [next, ...rest].filter(Boolean).map(lessonItem);
		}
		const lessons = rankLessons(allLessons, q, GROUP_MAX).map(lessonItem);
		const steps = rankSteps(stepEntries, q, GROUP_MAX).map((s): Item => {
			const meta = lessonByKey.get(lessonKey(s.track as never, s.lesson))!;
			const t = trackById.get(meta.track);
			return {
				id: `step-${s.track}-${s.lesson}-${s.stepIndex}`,
				group: 'Steps',
				title: s.title,
				sub: `${s.lessonTitle}, step ${s.stepIndex + 1}`,
				label: 'Step',
				href: `${lessonLink(meta)}?step=${s.stepIndex + 1}`,
				tone: t ? `var(--c-${t.tone})` : null
			};
		});
		const terms = searchGlossary(q)
			.slice(0, GROUP_MAX)
			.map(
				(e): Item => ({
					id: `term-${e.id}`,
					group: 'Terms',
					title: e.term,
					sub: firstSentence(e.definition),
					label: 'Term',
					href: `${resolve('/glossary')}#${e.id}`,
					tone: e.tone ? `var(--c-${e.tone})` : null
				})
			);
		return [...lessons, ...steps, ...terms];
	});

	const groups = $derived(
		(['Lessons', 'Steps', 'Terms'] as const)
			.map((g) => ({ name: g, rows: items.map((it, idx) => ({ it, idx })).filter((r) => r.it.group === g) }))
			.filter((g) => g.rows.length > 0)
	);

	$effect(() => {
		if (!ui.quickJumpOpen) return;
		opener = document.activeElement as HTMLElement | null;
		query = '';
		active = 0;
		void tick().then(() => input?.focus());
		return () => {
			// Closing by choosing a result navigates; otherwise hand focus back.
			if (!ui.contentsOpen) opener?.focus();
		};
	});

	$effect(() => {
		if (!ui.quickJumpOpen || items.length === 0) return;
		void tick().then(() => document.getElementById(`qj-opt-${active}`)?.scrollIntoView({ block: 'nearest' }));
	});

	function close(): void {
		ui.setQuickJumpOpen(false);
	}

	function choose(it: Item | undefined): void {
		if (!it) return;
		opener = null;
		close();
		void goto(it.href);
	}

	function onwindowkeydown(e: KeyboardEvent): void {
		if ((e.metaKey || e.ctrlKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'k') {
			e.preventDefault();
			ui.setQuickJumpOpen(!ui.quickJumpOpen);
		}
	}

	function onkeydown(e: KeyboardEvent): void {
		const n = items.length;
		if (e.key === 'Escape') {
			e.preventDefault();
			close();
		} else if (e.key === 'ArrowDown' && n) {
			e.preventDefault();
			active = (active + 1) % n;
		} else if (e.key === 'ArrowUp' && n) {
			e.preventDefault();
			active = (active - 1 + n) % n;
		} else if (e.key === 'Enter') {
			e.preventDefault();
			choose(items[active]);
		} else if (e.key === 'Tab') {
			// Arrow keys move through the results; keep focus in the field.
			e.preventDefault();
		}
	}
</script>

<svelte:window onkeydown={onwindowkeydown} />

{#if ui.quickJumpOpen}
	<div class="scrim" aria-hidden="true" onclick={close}></div>
	<div class="palette" role="dialog" aria-modal="true" aria-label="Quick jump" tabindex="-1" {onkeydown}>
		<div class="q">
			<Icon name="search" size={18} />
			<input
				bind:this={input}
				bind:value={query}
				type="text"
				role="combobox"
				aria-label="Search lessons, steps and terms"
				aria-expanded="true"
				aria-controls="qj-list"
				aria-autocomplete="list"
				aria-activedescendant={items.length ? `qj-opt-${active}` : undefined}
				placeholder="Search lessons, steps and terms"
				autocomplete="off"
				spellcheck="false"
				oninput={() => (active = 0)}
			/>
			<kbd>Esc</kbd>
		</div>
		<div id="qj-list" class="list" role="listbox" aria-label="Results">
			{#each groups as g (g.name)}
				<div class="grp" role="group" aria-labelledby="qj-g-{g.name}">
					<span class="lab" id="qj-g-{g.name}">{g.name}</span>
					{#each g.rows as { it, idx } (it.id)}
						<!-- svelte-ignore a11y_click_events_have_key_events -->
						<div
							id="qj-opt-{idx}"
							class="row"
							class:sel={idx === active}
							role="option"
							tabindex="-1"
							aria-selected={idx === active}
							style:--c={it.tone ?? 'var(--ink-2)'}
							onclick={() => choose(it)}
							onpointermove={() => (active = idx)}
						>
							<span class="ico">{#if it.group === 'Terms'}a{:else}<i></i>{/if}</span>
							<span class="text"
								>{#each highlight(it.title, query) as s, k (k)}{#if s.mark}<mark>{s.text}</mark>{:else}{s.text}{/if}{/each}<span class="s"> · {it.sub}</span></span
							>
							<span class="tag">{it.label}</span>
						</div>
					{/each}
				</div>
			{:else}
				<p class="empty">Nothing matches “{query.trim()}”.</p>
			{/each}
		</div>
		<footer>
			<span><kbd>↑</kbd><kbd>↓</kbd> move</span><span><kbd>↵</kbd> open</span><span><kbd>Esc</kbd> close</span>
		</footer>
	</div>
{/if}

<style>
	.scrim {
		position: fixed;
		inset: 0;
		z-index: 50;
		background: var(--scrim);
	}
	.palette {
		position: fixed;
		top: 104px;
		left: 50%;
		z-index: 51;
		transform: translateX(-50%);
		width: min(640px, calc(100vw - 32px));
		max-height: calc(100vh - 140px);
		display: flex;
		flex-direction: column;
		overflow: hidden;
		background-color: var(--sheet);
		background-image: var(--grain);
		border: 1px solid var(--rule-2);
		border-radius: 8px;
		box-shadow: var(--shadow);
		color: var(--ink);
	}
	.q {
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 16px 20px;
		border-bottom: 1px solid var(--rule);
		color: var(--ink-3);
	}
	input {
		flex: 1;
		min-width: 0;
		padding: 0;
		border: 0;
		outline: 0;
		background: transparent;
		color: var(--ink);
		font: inherit;
		font-size: 19px;
	}
	input::placeholder {
		color: var(--ink-3);
	}
	kbd {
		font-family: var(--font-mono);
		font-size: 10.5px;
		padding: 1px 5px;
		border: 1px solid var(--rule-2);
		border-radius: 3px;
		background: var(--sheet);
		color: var(--ink-2);
	}
	.list {
		overflow-y: auto;
	}
	.grp {
		padding: 8px;
	}
	.grp + .grp {
		border-top: 1px solid var(--rule);
	}
	.lab {
		display: block;
		padding: 4px 12px 6px;
		font-size: 12px;
		font-weight: 500;
		color: var(--ink-3);
	}
	.row {
		display: grid;
		grid-template-columns: 28px minmax(0, 1fr) auto;
		align-items: center;
		gap: 10px;
		padding: 8px 12px;
		border-radius: 4px;
		font-size: 14.5px;
		cursor: pointer;
	}
	.row.sel {
		background: color-mix(in srgb, var(--ink) 7%, transparent);
	}
	mark {
		padding: 0 1px;
		border-radius: 2px;
		background: color-mix(in srgb, var(--grad) 24%, transparent);
		color: inherit;
	}
	.ico {
		display: grid;
		place-items: center;
		width: 24px;
		height: 24px;
		border: 1px solid var(--rule);
		border-radius: 4px;
		font-family: var(--font-display);
		font-style: italic;
		font-size: 13px;
		color: var(--c);
	}
	.ico i {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--c);
	}
	.text {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.s {
		font-size: 12.5px;
		color: var(--ink-2);
	}
	.tag {
		font-size: 12px;
		color: var(--ink-3);
	}
	.empty {
		margin: 0;
		padding: 24px 20px;
		font-size: 14px;
		color: var(--ink-2);
	}
	footer {
		display: flex;
		gap: 18px;
		padding: 10px 20px;
		border-top: 1px solid var(--rule);
		font-size: 12px;
		color: var(--ink-2);
	}
	footer span {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
</style>

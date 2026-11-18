<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { page } from '$app/state';
	import { allLessons, tracks } from '#lib/content/curriculum.ts';
	import { lessonLink } from '#lib/link.ts';
	import { findEntry, glossary, searchGlossary } from '#lib/content/glossary.ts';
	import { settings } from '#lib/state/settings.svelte.ts';

	const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

	const lessonInfo = new Map(
		allLessons.map((l) => [
			`${l.track}/${l.id}`,
			{
				title: l.title,
				href: lessonLink(l),
				tone: tracks.find((t) => t.id === l.track)?.tone ?? 'data'
			}
		])
	);

	let query = $state('');
	let activeId: string | null = $state(null);
	let field: HTMLInputElement | undefined = $state();

	const searching = $derived(query.trim() !== '');
	const results = $derived(searchGlossary(query));
	const groups = $derived.by(() => {
		const map = new Map<string, typeof glossary>();
		for (const e of glossary) {
			const letter = e.term[0].toUpperCase();
			map.set(letter, [...(map.get(letter) ?? []), e]);
		}
		return map;
	});

	/** Show the entry a hash names: drop any search that hides it, then scroll it into view. */
	async function reveal(hash: string): Promise<void> {
		let id = hash.replace(/^#/, '');
		try {
			id = decodeURIComponent(id);
		} catch {
			// keep the raw fragment
		}
		if (!findEntry(id)) return;
		query = '';
		activeId = id;
		await tick();
		document
			.getElementById(id)
			?.scrollIntoView({ behavior: settings.reducedMotion ? 'auto' : 'smooth', block: 'start' });
	}

	onMount(() => {
		void reveal(location.hash);
	});

	// SvelteKit changes the URL with pushState for links inside the page, which fires no hashchange.
	let lastHash = '';
	$effect(() => {
		const hash = page.url.hash;
		if (hash === lastHash) return;
		lastHash = hash;
		void reveal(hash);
	});

	function onkeydown(e: KeyboardEvent): void {
		if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey) return;
		const el = e.target as HTMLElement | null;
		if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return;
		e.preventDefault();
		field?.focus();
	}

	function onfieldkey(e: KeyboardEvent): void {
		if (e.key === 'Escape') {
			e.preventDefault();
			query = '';
		}
	}
</script>

<svelte:head><title>Glossary · Neural Atlas</title></svelte:head>
<svelte:window {onkeydown} onhashchange={() => void reveal(location.hash)} />

<main>
	<div class="side">
		<h1>Glossary</h1>
		<p class="lede">
			Short definitions for the terms the lessons use, with links to where each one is shown.
		</p>
		<p class="count num">{glossary.length} terms</p>
		<input
			bind:this={field}
			bind:value={query}
			type="search"
			placeholder="Search terms"
			aria-label="Search the glossary"
			autocomplete="off"
			spellcheck="false"
			onkeydown={onfieldkey}
		/>
		{#if searching}
			<p class="found num" aria-live="polite">{results.length} of {glossary.length}</p>
		{:else}
			<nav class="letters" aria-label="Jump to letter">
				{#each LETTERS as l (l)}
					{#if groups.has(l)}
						<a href="#letter-{l.toLowerCase()}">{l}</a>
					{:else}
						<span class="dim" aria-hidden="true">{l}</span>
					{/if}
				{/each}
			</nav>
		{/if}
	</div>

	<div class="body">
		{#snippet article(e: (typeof glossary)[number])}
			<article id={e.id} class:active={activeId === e.id}>
				<h3>
					<span class="term" style:color={e.tone ? `var(--c-${e.tone})` : undefined}>{e.term}</span>
					{#if e.aliases?.length}
						<span class="aliases">also: {e.aliases.join(', ')}</span>
					{/if}
				</h3>
				<div class="text">
				<p class="def">{e.definition}</p>
				<div class="row">
					<span class="label">Shown in</span>
					{#each e.lessons as key (key)}
						{@const info = lessonInfo.get(key)}
						{#if info}
							<a class="pill" href={info.href}>
								<span class="dot" style:background="var(--c-{info.tone})"></span>{info.title}
							</a>
						{/if}
					{/each}
				</div>
				{#if e.related?.length}
					<p class="related">
						<span class="label">Related:</span>
						{#each e.related as id, i (id)}
							{@const r = findEntry(id)}
							{#if r}<a href="#{r.id}">{r.term}</a>{i < e.related.length - 1 ? ', ' : ''}{/if}
						{/each}
					</p>
				{/if}
				</div>
			</article>
		{/snippet}

		{#if !searching}
			{#each [...groups] as [letter, entries] (letter)}
				<section>
					<h2 id="letter-{letter.toLowerCase()}">{letter}</h2>
					{#each entries as e (e.id)}{@render article(e)}{/each}
				</section>
			{/each}
		{:else if results.length > 0}
			{#each results as e (e.id)}{@render article(e)}{/each}
		{:else}
			<div class="empty">
				<p>No term matches "{query.trim()}".</p>
				<button type="button" class="btn" onclick={() => (query = '')}>Clear search</button>
			</div>
		{/if}
	</div>
</main>

<style>
	main {
		display: grid;
		grid-template-columns: 300px minmax(0, 1fr);
		gap: 72px;
		align-items: start;
		max-width: 1280px;
		margin: 0 auto;
		padding: 56px 48px 96px;
	}
	.side {
		position: sticky;
		top: calc(var(--topbar-h) + 24px);
		display: grid;
		gap: 20px;
		align-content: start;
	}
	h1 {
		font-family: var(--font-display);
		font-weight: 500;
		font-size: 60px;
		line-height: 1;
		letter-spacing: -0.02em;
	}
	.lede {
		color: var(--ink-2);
		font-size: 15px;
	}
	.count,
	.found {
		font-size: 13px;
		color: var(--ink-3);
		margin-top: -10px;
	}
	input {
		width: 100%;
		height: 42px;
		padding: 0 14px;
		border-radius: 4px;
		border: 1px solid var(--rule-2);
		background-color: var(--sheet);
		color: var(--ink);
		font-family: var(--font-body);
		font-size: 14px;
	}
	input::placeholder {
		color: var(--ink-3);
	}
	input:focus-visible {
		outline: 2px solid var(--data);
		outline-offset: 1px;
	}
	.letters {
		display: grid;
		grid-template-columns: repeat(7, 1fr);
		gap: 4px;
		font-family: var(--font-display);
		font-size: 18px;
	}
	.letters a,
	.letters .dim {
		text-align: center;
		padding: 3px 0;
		border-radius: 3px;
	}
	.letters a {
		color: var(--ink);
	}
	.letters a:hover {
		background: var(--ink);
		color: var(--pg);
	}
	.letters .dim {
		color: var(--ink-3);
		opacity: 0.45;
	}

	.body {
		border-top: 2px solid var(--ink);
		min-width: 0;
	}
	h2 {
		margin: 0;
		padding: 22px 0 8px;
		font-family: var(--font-display);
		font-style: italic;
		font-weight: 400;
		font-size: 40px;
		line-height: 1;
		color: var(--ink-3);
		scroll-margin-top: calc(var(--topbar-h) + 16px);
	}
	article {
		display: grid;
		grid-template-columns: 210px minmax(0, 1fr);
		gap: 24px;
		padding: 14px 12px;
		margin: 0 -12px;
		border-bottom: 1px solid var(--rule);
		border-left: 2px solid transparent;
		scroll-margin-top: calc(var(--topbar-h) + 16px);
	}
	article:target,
	article.active {
		border-left-color: var(--data);
		background: color-mix(in srgb, var(--data) 9%, transparent);
	}
	h3 {
		display: grid;
		align-content: start;
		gap: 2px;
		margin: 0;
		font-family: var(--font-display);
		font-size: 22px;
		font-weight: 500;
		line-height: 1.2;
	}
	.aliases {
		font-family: var(--font-body);
		font-size: 12.5px;
		font-weight: 400;
		color: var(--ink-3);
	}
	.def {
		margin-bottom: 10px;
		font-size: 15px;
		line-height: 1.55;
		color: var(--ink);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
		font-size: 12px;
	}
	.label {
		color: var(--ink-3);
	}
	.pill {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 2px 10px;
		border: 1px solid var(--rule-2);
		border-radius: var(--r-pill);
		color: var(--ink-2);
	}
	.pill:hover {
		border-color: var(--ink-3);
		color: var(--ink);
	}
	.dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
	}
	.related {
		margin-top: 8px;
		font-size: 13px;
		color: var(--ink-2);
	}
	.related a {
		color: var(--data);
		text-decoration: underline;
		text-underline-offset: 2px;
	}
	.empty {
		display: grid;
		justify-items: start;
		gap: 12px;
		padding: 40px 0;
		color: var(--ink-2);
	}
</style>

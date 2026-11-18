<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { page } from '$app/state';
	import { allLessons, lessonHref, tracks } from '#lib/content/curriculum.ts';
	import { findEntry, glossary, searchGlossary } from '#lib/content/glossary.ts';
	import { settings } from '#lib/state/settings.svelte.ts';

	const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

	const lessonInfo = new Map(
		allLessons.map((l) => [
			`${l.track}/${l.id}`,
			{
				title: l.title,
				href: lessonHref(l),
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
	<header class="head">
		<h1>Glossary</h1>
		<p class="lede">
			Short definitions for the terms the lessons use, with links to where each one is shown.
		</p>
		<p class="count num">{glossary.length} terms</p>
	</header>

	<div class="bar">
		<div class="bar-in">
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
		--bar-h: 56px;
	}
	.head,
	.bar-in,
	.body {
		max-width: 860px;
		margin: 0 auto;
		padding-left: 24px;
		padding-right: 24px;
	}
	.head {
		padding-top: 48px;
		padding-bottom: 24px;
	}
	h1 {
		font-size: 32px;
		margin-bottom: 8px;
	}
	.lede {
		color: var(--c-text-2);
		margin-bottom: 8px;
	}
	.count {
		font-size: 13px;
		color: var(--c-muted);
	}

	.bar {
		position: sticky;
		top: var(--topbar-h);
		z-index: 20;
		height: var(--bar-h);
		background: var(--bg-void);
		border-bottom: 1px solid var(--line);
	}
	.bar-in {
		display: flex;
		align-items: center;
		gap: 20px;
		height: 100%;
	}
	input {
		flex: none;
		width: 320px;
		height: 32px;
		padding: 0 12px;
		border-radius: var(--r-pill);
		border: 1px solid var(--line-strong);
		background: var(--glass);
		color: var(--c-text);
		font-family: var(--font-mono);
		font-size: 13px;
	}
	input::placeholder {
		color: var(--c-muted);
	}
	input:focus-visible {
		outline: none;
		border-color: var(--c-data);
	}
	.found {
		font-size: 13px;
		color: var(--c-muted);
	}
	.letters {
		display: flex;
		flex-wrap: nowrap;
		gap: 2px;
		margin-left: auto;
		font-family: var(--font-mono);
		font-size: 12px;
	}
	.letters a,
	.letters .dim {
		display: grid;
		place-items: center;
		width: 17px;
		height: 22px;
		border-radius: 4px;
	}
	.letters a {
		color: var(--c-text-2);
	}
	.letters a:hover {
		background: var(--glass-strong);
		color: var(--c-text);
	}
	.letters .dim {
		color: var(--c-muted);
		opacity: 0.4;
	}

	.body {
		padding-top: 8px;
		padding-bottom: 96px;
	}
	h2 {
		margin: 28px 0 4px;
		font-family: var(--font-mono);
		font-size: 13px;
		font-weight: 500;
		color: var(--c-muted);
		scroll-margin-top: calc(var(--topbar-h) + var(--bar-h) + 16px);
	}
	article {
		margin: 0 -12px;
		padding: 14px 12px 14px 11px;
		border-left: 1px solid transparent;
		border-radius: var(--r-sm);
		scroll-margin-top: calc(var(--topbar-h) + var(--bar-h) + 16px);
	}
	article:target,
	article.active {
		border-left-color: var(--c-data);
		background: var(--glass);
	}
	h3 {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 10px;
		margin-bottom: 4px;
		font-family: var(--font-display);
		font-size: 17px;
		font-weight: 500;
	}
	.aliases {
		font-family: var(--font-body);
		font-size: 13px;
		font-weight: 400;
		color: var(--c-muted);
	}
	.def {
		margin-bottom: 10px;
		font-size: 15px;
		color: var(--c-text-2);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
		font-size: 12px;
	}
	.label {
		color: var(--c-muted);
	}
	.pill {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 2px 10px;
		border: 1px solid var(--line);
		border-radius: var(--r-pill);
		color: var(--c-text-2);
	}
	.pill:hover {
		border-color: var(--line-strong);
		background: var(--glass);
		color: var(--c-text);
	}
	.dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
	}
	.related {
		margin-top: 8px;
		font-size: 13px;
		color: var(--c-text-2);
	}
	.related a {
		color: var(--c-data);
	}
	.empty {
		display: grid;
		justify-items: start;
		gap: 12px;
		padding: 40px 0;
		color: var(--c-text-2);
	}
</style>

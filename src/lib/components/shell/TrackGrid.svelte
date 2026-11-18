<script lang="ts">
	import { tracks, lessonKey, lessonHref } from '../../content/curriculum.ts';
	import { progress } from '../../state/progress.svelte.ts';
</script>

<div class="grid">
	{#each tracks as track (track.id)}
		{@const done = track.lessons.filter((l) => progress.isComplete(lessonKey(l.track, l.id))).length}
		<section class="card glass" style:--tone="var(--c-{track.tone})">
			<h2>{track.title}</h2>
			<p class="blurb">{track.blurb}</p>
			<p class="count num">{done} of {track.lessons.length} complete</p>
			<ul>
				{#each track.lessons as l (l.id)}
					{@const complete = progress.isComplete(lessonKey(l.track, l.id))}
					<li>
						{#if l.status === 'ready'}
							<a class="row" href={lessonHref(l)}>
								<span class="dot" class:complete></span>
								<span class="title">{l.title}</span>
								<span class="min num">{l.minutes} min</span>
							</a>
						{:else}
							<div class="row soon">
								<span class="dot"></span>
								<span class="title">{l.title}</span>
								<span class="tag">Soon</span>
								<span class="min num">{l.minutes} min</span>
							</div>
						{/if}
					</li>
				{/each}
			</ul>
		</section>
	{/each}
</div>

<style>
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
		gap: 16px;
		max-width: 1120px;
		margin: 0 auto;
		padding: 0 24px;
	}
	.card {
		border-top: 2px solid var(--tone);
		padding: 20px 12px 12px;
	}
	h2 {
		font-size: 19px;
		padding: 0 8px;
	}
	.blurb {
		color: var(--c-text-2);
		font-size: 14px;
		margin-top: 6px;
		padding: 0 8px;
	}
	.count {
		font-size: 12px;
		color: var(--c-muted);
		margin: 10px 8px 12px;
	}
	ul {
		list-style: none;
		margin: 0;
		padding: 0;
	}
	.row {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 7px 8px;
		border-radius: var(--r-sm);
		font-size: 14px;
	}
	a.row:hover {
		background: var(--glass-strong);
	}
	.dot {
		flex: none;
		width: 6px;
		height: 6px;
		border-radius: 50%;
		border: 1px solid var(--c-muted);
	}
	.dot.complete {
		background: var(--c-ok);
		border-color: var(--c-ok);
	}
	a.row .dot:not(.complete) {
		background: var(--tone);
		border-color: var(--tone);
	}
	.title {
		flex: 1;
		min-width: 0;
	}
	.soon {
		color: var(--c-muted);
	}
	.tag {
		font-size: 11px;
		padding: 0 7px;
		border: 1px solid var(--line);
		border-radius: var(--r-pill);
	}
	.min {
		font-size: 12px;
		color: var(--c-muted);
	}
</style>

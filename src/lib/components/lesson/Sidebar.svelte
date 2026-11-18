<script lang="ts">
	import { tracks, lessonKey } from '../../content/curriculum.ts';
	import { lessonLink } from '../../link.ts';
	import { progress } from '../../state/progress.svelte.ts';
	import Icon from '../shell/Icon.svelte';

	interface Props {
		currentKey: string;
		onclose: () => void;
	}

	let { currentKey, onclose }: Props = $props();
</script>

<nav class="sidebar" aria-label="Lessons">
	<div class="head">
		<span class="title">Lessons</span>
		<button type="button" class="close" aria-label="Hide lesson list" onclick={onclose}>
			<Icon name="chevron-left" />
		</button>
	</div>
	<div class="scroll">
		{#each tracks as track (track.id)}
			<h3>{track.title}</h3>
			<ul>
				{#each track.lessons as l (l.id)}
					{@const key = lessonKey(l.track, l.id)}
					{@const current = key === currentKey}
					<li>
						{#if l.status === 'ready'}
							<a
								class="row"
								class:current
								href={lessonLink(l)}
								aria-current={current ? 'page' : undefined}
							>
								<span
									class="dot"
									class:complete={progress.isComplete(key)}
									class:cur={current}
									style:--tone="var(--c-{track.tone})"
								></span>
								<span class="t">{l.title}</span>
							</a>
						{:else}
							<div class="row soon">
								<span class="dot"></span>
								<span class="t">{l.title}</span>
								<span class="tag">Soon</span>
							</div>
						{/if}
					</li>
				{/each}
			</ul>
		{/each}
	</div>
</nav>

<style>
	.sidebar {
		display: flex;
		flex-direction: column;
		width: 100%;
		height: 100%;
		min-height: 0;
		background: var(--bg-panel);
		border-right: 1px solid var(--line);
	}
	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		height: 44px;
		padding: 0 8px 0 16px;
		flex: none;
	}
	.title {
		font-size: 12px;
		color: var(--c-muted);
	}
	.close {
		display: grid;
		place-items: center;
		width: 28px;
		height: 28px;
		padding: 0;
		background: transparent;
		border: 0;
		border-radius: var(--r-sm);
		color: var(--c-text-2);
		cursor: pointer;
	}
	.close:hover {
		background: var(--glass-strong);
		color: var(--c-text);
	}
	.scroll {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		padding: 0 8px 16px;
	}
	h3 {
		font-family: var(--font-body);
		font-size: 12px;
		font-weight: 500;
		color: var(--c-muted);
		padding: 16px 8px 6px;
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
		padding: 6px 8px;
		border-radius: var(--r-sm);
		font-size: 14px;
		color: var(--c-text-2);
	}
	a.row:hover {
		color: var(--c-text);
		background: var(--glass);
	}
	.row.current {
		background: var(--glass-strong);
		color: var(--c-text);
	}
	.row.soon {
		color: var(--c-muted);
	}
	.t {
		flex: 1;
		min-width: 0;
	}
	.dot {
		flex: none;
		width: 6px;
		height: 6px;
		border-radius: 50%;
		border: 1px solid var(--c-muted);
	}
	.dot.cur {
		background: var(--tone);
		border-color: var(--tone);
	}
	.dot.complete {
		background: var(--c-ok);
		border-color: var(--c-ok);
	}
	.tag {
		font-size: 11px;
		padding: 0 7px;
		border: 1px solid var(--line);
		border-radius: var(--r-pill);
	}
</style>

<script lang="ts">
	import Stage from '../lesson/Stage.svelte';
	import { createAtlasScene, regions, type AtlasAnchor } from '../../lessons/home/atlas-scene.ts';
	import { settings } from '../../state/settings.svelte.ts';

	interface Props {
		region: string | null;
		lesson: number | null;
		reducedMotion: boolean;
		completed?: number;
		onregion: (id: string) => void;
		onlesson: (index: number) => void;
	}

	let { region, lesson, reducedMotion, completed = 0, onregion, onlesson }: Props = $props();
	let stage: ReturnType<typeof Stage> | undefined = $state();
	let anchors: AtlasAnchor[] = $state.raw([]);
	let activePointer: number | null = null;
	const params = $derived({
		region: regions.findIndex((topic) => topic.id === region) + 1,
		lesson: lesson ?? -1,
		completed
	});
	const visibleAnchors = $derived(
		anchors.filter((anchor) =>
			region === 'fundamentals' ? anchor.kind === 'lesson' :
			anchor.kind === 'region' && (region === null || anchor.id === region)
		)
	);

	function makeScene() {
		return createAtlasScene((next) => { anchors = next; }, reducedMotion);
	}

	export function resetView(): void {
		stage?.command('view:reset');
	}

	function capturePointer(event: PointerEvent) {
		const target = event.target;
		if (!(target instanceof HTMLCanvasElement)) return;
		if (!event.isPrimary || event.button !== 0) {
			event.stopPropagation();
			return;
		}
		activePointer = event.pointerId;
		target.setPointerCapture(event.pointerId);
	}

	function endPointer(event: PointerEvent) {
		if (activePointer !== event.pointerId) return;
		activePointer = null;
		stage?.command('view:cancel-drag');
	}
</script>

<div
	class="atlas"
	role="region"
	aria-label="Interactive learning atlas"
	data-view={region ?? 'overview'}
	data-lesson={lesson ?? 'none'}
	onpointerdowncapture={capturePointer}
	onpointerup={endPointer}
	onpointercancel={endPointer}
	onlostpointercapture={endPointer}
>
	{#key settings.theme + String(reducedMotion)}
		<Stage createScene={makeScene} {params} bind:this={stage} />
	{/key}
	<div class="anchors">
		{#each visibleAnchors as anchor (anchor.kind + anchor.id)}
			{@const selected = anchor.kind === 'lesson' && Number(anchor.id) === lesson}
			<button
				type="button"
				class:lesson={anchor.kind === 'lesson'}
				aria-label={anchor.kind === 'region' ? `Explore ${anchor.label}` : `Preview ${anchor.label}`}
				aria-pressed={anchor.kind === 'lesson' ? selected : anchor.id === region}
				title={anchor.label}
				style:left={anchor.x + 'px'}
				style:top={anchor.y + 'px'}
				style:--accent={'var(--c-' + anchor.tone + ')'}
				onclick={() => anchor.kind === 'region' ? onregion(anchor.id) : onlesson(Number(anchor.id))}
			>
				{#if anchor.kind === 'lesson'}
					<span class="ordinal" aria-hidden="true">{String(Number(anchor.id) + 1).padStart(2, '0')}</span>
				{:else}
					<span class="dot" aria-hidden="true"></span>
				{/if}
				<span>{anchor.label}</span>
			</button>
		{/each}
	</div>
	{#if visibleAnchors.length}
		<span class="drag-hint">
			<svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
				<path d="M3 6 1 8l2 2m10-4 2 2-2 2M1 8h14M6 3l2-2 2 2m-4 10 2 2 2-2M8 1v14" />
			</svg>
			Drag to look around
		</span>
	{/if}
</div>

<style>
	.atlas {
		position: relative;
		min-width: 0;
		height: 100%;
		min-height: 400px;
		isolation: isolate;
		touch-action: pan-y;
		user-select: none;
	}

	.atlas :global(canvas) { cursor: grab; }
	.atlas :global(canvas:active) { cursor: grabbing; }

	.anchors {
		position: absolute;
		inset: 0;
		pointer-events: none;
		container-type: inline-size;
	}

	button {
		--accent: var(--c-data);
		position: absolute;
		transform: translate(-50%, -50%);
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 0.42rem;
		width: max-content;
		max-width: 190px;
		min-height: 38px;
		padding: 0.31rem 0.66rem;
		border: 1px solid color-mix(in srgb, var(--accent), transparent 67%);
		border-radius: var(--r-pill);
		background: color-mix(in srgb, var(--sheet), transparent 4%);
		color: var(--ink);
		box-shadow: 0 4px 14px -8px var(--accent);
		font-family: var(--font-body);
		font-size: 0.76rem;
		font-weight: 650;
		line-height: 1.15;
		text-align: left;
		pointer-events: auto;
		cursor: pointer;
	}

	button.lesson {
		max-width: 148px;
		padding-inline: 0.53rem 0.67rem;
	}

	button:hover, button[aria-pressed='true'] {
		border-color: var(--accent);
		background: color-mix(in srgb, var(--sheet), var(--accent) 12%);
	}

	button:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 3px;
		z-index: 1;
	}

	.dot {
		width: 0.48rem;
		height: 0.48rem;
		flex: 0 0 auto;
		border-radius: 50%;
		background: var(--accent);
	}

	.ordinal {
		font-family: var(--font-mono);
		font-size: 0.64rem;
		color: var(--accent);
	}

	.drag-hint {
		position: absolute;
		right: 1rem;
		bottom: 0.65rem;
		display: inline-flex;
		align-items: center;
		gap: 0.38rem;
		color: var(--ink-3);
		font-family: var(--font-mono);
		font-size: 0.64rem;
		pointer-events: none;
	}

	.drag-hint svg {
		width: 0.74rem;
		height: 0.74rem;
		stroke: currentColor;
		stroke-width: 1.1;
		stroke-linejoin: round;
		stroke-linecap: round;
	}

	:global([data-theme='night']) button {
		box-shadow: 0 0 16px -10px var(--accent);
	}

	@container (width < 630px) {
		button {
			max-width: 125px;
			font-size: 0.67rem;
			padding-inline: 0.45rem;
		}
		button.lesson {
			max-width: 124px;
			min-height: 39px;
			padding-inline: 0.48rem;
		}
	}
</style>

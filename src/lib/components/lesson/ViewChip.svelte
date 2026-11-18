<script lang="ts">
	interface Props {
		/** The learner has turned the figure away from the step's own view. */
		moved: boolean;
		/** The figure can be turned by dragging. */
		drag: boolean;
		onreset: () => void;
	}

	let { moved, drag, onreset }: Props = $props();
</script>

{#if moved}
	<button type="button" class="view-chip" onclick={onreset}>
		<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"
			><ellipse cx="8" cy="8.5" rx="6.6" ry="3.1" stroke="currentColor" stroke-width="1.3" /><path
				d="M11.6 3.9 14.3 5.3 12.4 7.6"
				stroke="currentColor"
				stroke-width="1.3"
				stroke-linecap="round"
				stroke-linejoin="round"
			/><circle cx="8" cy="8.5" r="1.6" fill="currentColor" /></svg
		>
		Back to this step's view
	</button>
{:else if drag}
	<p class="drag-hint">
		<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"
			><ellipse cx="8" cy="8.5" rx="6.6" ry="3.1" stroke="currentColor" stroke-width="1.3" /><path
				d="M11.6 3.9 14.3 5.3 12.4 7.6"
				stroke="currentColor"
				stroke-width="1.3"
				stroke-linecap="round"
				stroke-linejoin="round"
			/><circle cx="8" cy="8.5" r="1.6" fill="currentColor" /></svg
		>
		Drag to turn the figure
	</p>
{/if}

<style>
	.view-chip,
	.drag-hint {
		display: inline-flex;
		align-items: center;
		gap: 7px;
		margin: 0;
		white-space: nowrap;
	}
	svg {
		flex: none;
	}
	.view-chip {
		height: 30px;
		padding: 0 12px;
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		background-color: var(--sheet);
		background-image: var(--grain);
		color: var(--ink);
		font: inherit;
		font-size: 12.5px;
		font-weight: 500;
		line-height: 1;
		box-shadow: var(--shadow);
		cursor: pointer;
		pointer-events: auto;
	}
	.view-chip svg {
		width: 13px;
		height: 13px;
	}
	.view-chip:hover {
		background-color: color-mix(in srgb, var(--ink) 7%, var(--sheet));
	}
	.drag-hint {
		font-size: 12px;
		color: var(--ink-3);
		pointer-events: none;
	}
	.drag-hint svg {
		width: 15px;
		height: 15px;
	}
</style>

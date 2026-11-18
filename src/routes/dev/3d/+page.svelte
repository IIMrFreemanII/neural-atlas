<script lang="ts">
	import Stage from '#lib/components/lesson/Stage.svelte';
	import { createToolkit3dDemoScene } from '#lib/gpu/demo-scene-3d.ts';

	let stage: ReturnType<typeof Stage> | undefined = $state();
	let moved = $state(false);
</script>

<div class="host">
	<Stage
		bind:this={stage}
		createScene={createToolkit3dDemoScene}
		onreadout={(key, value) => {
			if (key === 'view:moved') moved = value === 1;
		}}
	/>
	<div class="bar">
		<button onclick={() => stage?.command('pose')}>Toggle pose</button>
		{#if moved}
			<button onclick={() => stage?.command('view:reset')}>Back to this step's view</button>
		{/if}
		<span>Drag to turn. Click the floor to move the ball.</span>
	</div>
</div>

<style>
	.host {
		position: relative;
		width: 100%;
		height: 100dvh;
		background: var(--pg);
	}
	.bar {
		position: absolute;
		left: 16px;
		bottom: 16px;
		display: flex;
		gap: 8px;
		align-items: center;
		font-size: 12px;
		color: var(--c-muted, #687080);
	}
</style>

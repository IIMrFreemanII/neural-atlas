<script lang="ts">
	import Stage from '#lib/components/lesson/Stage.svelte';
	import { createPrimitivesDemoScene } from '#lib/gpu/demo-scene-2.ts';
</script>

<div class="host">
	<Stage createScene={createPrimitivesDemoScene} />
</div>

<style>
	.host {
		width: 100%;
		height: 100dvh;
		background: #07080d;
	}
</style>

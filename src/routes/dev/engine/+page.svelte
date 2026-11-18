<script lang="ts">
	import Stage from '#lib/components/lesson/Stage.svelte';
	import { createDemoScene } from '#lib/gpu/demo-scene.ts';
</script>

<div class="host">
	<Stage createScene={createDemoScene} />
</div>

<style>
	.host {
		width: 100%;
		height: 100dvh;
		background: #07080d;
	}
</style>

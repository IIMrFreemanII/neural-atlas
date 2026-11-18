<script lang="ts">
	import Stage from '#lib/components/lesson/Stage.svelte';
	import { createNetworkDemoScene } from '#lib/lessons/shared/network-demo-scene.ts';
</script>

<div class="host">
	<Stage createScene={createNetworkDemoScene} />
</div>

<style>
	.host {
		width: 100%;
		height: 100dvh;
		background: #07080d;
	}
</style>

<script lang="ts">
	import Stage from '#lib/components/lesson/Stage.svelte';
	import TrackGrid from '#lib/components/shell/TrackGrid.svelte';
	import { createHeroScene } from '#lib/lessons/home/hero-scene.ts';
</script>

<svelte:head><title>Neural Atlas</title></svelte:head>

<section class="hero">
	<div class="bg"><Stage createScene={createHeroScene} /></div>
	<div class="content">
		<p class="eyebrow">Neural Atlas</p>
		<h1>See how machines learn</h1>
		<p class="lede">
			Interactive visuals that take neural networks, classic ML and language models apart, one idea at
			a time.
		</p>
		<div class="actions">
			<a class="btn primary" href="/learn/fundamentals/neuron">Start with the neuron</a>
			<a class="btn" href="/learn">Browse all lessons</a>
		</div>
	</div>
</section>

<div class="grid"><TrackGrid /></div>

<style>
	.hero {
		position: relative;
		min-height: min(72vh, 640px);
		display: flex;
		align-items: center;
		overflow: hidden;
	}
	.bg {
		position: absolute;
		inset: 0;
		pointer-events: none;
	}
	.content {
		position: relative;
		width: 100%;
		max-width: 1120px;
		margin: 0 auto;
		padding: 48px 24px;
	}
	.eyebrow {
		font-family: var(--font-mono);
		font-size: 12px;
		color: var(--c-data);
		margin-bottom: 16px;
	}
	h1 {
		font-size: clamp(36px, 6vw, 64px);
		line-height: 1.05;
		margin-bottom: 20px;
	}
	.lede {
		max-width: 52ch;
		color: var(--c-text-2);
		font-size: 17px;
		margin-bottom: 28px;
	}
	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: 12px;
	}
	.actions .btn {
		height: 42px;
		padding: 0 22px;
		font-size: 15px;
	}
	.grid {
		padding: 24px 0 80px;
	}
</style>

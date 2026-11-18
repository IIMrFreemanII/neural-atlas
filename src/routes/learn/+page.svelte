<script lang="ts">
	import TrackGrid from '#lib/components/shell/TrackGrid.svelte';
</script>

<svelte:head><title>Lessons · Neural Atlas</title></svelte:head>

<div class="top"><header>
	<h1>Lessons</h1>
	<p>
		Four tracks, each a short run of interactive lessons. Start anywhere; inside a track, each lesson
		builds on the one before it.
	</p>
</header></div>
<div class="grid"><TrackGrid /></div>

<style>
	.top {
		max-width: 1280px;
		margin: 0 auto 40px;
		padding: 56px 48px 0;
		box-sizing: border-box;
	}
	header {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 40px;
		align-items: end;
		padding-bottom: 22px;
		border-bottom: 2px solid var(--ink);
	}
	h1 {
		font-family: var(--font-display);
		font-weight: 500;
		font-size: 60px;
		line-height: 1;
		letter-spacing: -0.02em;
	}
	p {
		color: var(--ink-2);
		font-size: 16px;
		max-width: 60ch;
	}
	.grid {
		padding-bottom: 80px;
	}
</style>

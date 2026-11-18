<script lang="ts">
	import TrackGrid from '#lib/components/shell/TrackGrid.svelte';
</script>

<svelte:head><title>Lessons · Neural Atlas</title></svelte:head>

<header>
	<h1>Lessons</h1>
	<p>
		Four tracks, each a short sequence of interactive lessons. Start anywhere; within a track, each
		lesson builds on the one before it.
	</p>
</header>
<div class="grid"><TrackGrid /></div>

<style>
	header {
		max-width: 1120px;
		margin: 0 auto;
		padding: 48px 24px 28px;
	}
	h1 {
		font-size: clamp(30px, 4vw, 40px);
		margin-bottom: 8px;
	}
	p {
		color: var(--c-text-2);
		max-width: 60ch;
	}
	.grid {
		padding-bottom: 80px;
	}
</style>

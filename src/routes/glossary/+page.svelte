<svelte:head><title>Glossary · Neural Atlas</title></svelte:head>

<main>
	<div class="card glass">
		<h1>Glossary</h1>
		<p>Short definitions for every term used in the lessons will live here.</p>
	</div>
</main>

<style>
	main {
		display: grid;
		place-items: center;
		min-height: calc(100dvh - var(--topbar-h));
		padding: 24px;
	}
	.card {
		max-width: 480px;
		padding: 32px;
		text-align: center;
	}
	h1 {
		font-size: 28px;
		margin-bottom: 10px;
	}
	p {
		color: var(--c-text-2);
	}
</style>

<svelte:head><title>Playground · Neural Atlas</title></svelte:head>

<main>
	<div class="card glass">
		<h1>Playground</h1>
		<p>A free-form sandbox for training small networks is coming after the first lessons.</p>
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

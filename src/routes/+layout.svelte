<script lang="ts">
	import '@fontsource-variable/inter';
	import '@fontsource-variable/space-grotesk';
	import '@fontsource-variable/jetbrains-mono';
	import '../app.css';
	import favicon from '#lib/assets/favicon.svg';
	import TopBar from '#lib/components/shell/TopBar.svelte';
	import { settings } from '#lib/state/settings.svelte.ts';

	let { children } = $props();

	$effect(() => {
		document.documentElement.toggleAttribute('data-reduce-motion', settings.reducedMotion);
	});
</script>

<svelte:head>
	<link rel="icon" href={favicon} type="image/svg+xml" />
</svelte:head>

<TopBar />
{@render children()}

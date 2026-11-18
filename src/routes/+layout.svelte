<script lang="ts">
	import '@fontsource-variable/inter';
	import '@fontsource-variable/newsreader/opsz.css';
	import '@fontsource-variable/newsreader/opsz-italic.css';
	import '@fontsource-variable/jetbrains-mono';
	import '../app.css';
	import favicon from '#lib/assets/favicon.svg';
	import ContentsDrawer from '#lib/components/shell/ContentsDrawer.svelte';
	import QuickJump from '#lib/components/shell/QuickJump.svelte';
	import TopBar from '#lib/components/shell/TopBar.svelte';
	import { settings } from '#lib/state/settings.svelte.ts';

	let { children } = $props();

	$effect(() => {
		document.documentElement.toggleAttribute('data-reduce-motion', settings.reducedMotion);
	});

	$effect(() => {
		document.documentElement.dataset.theme = settings.theme;
		document
			.querySelector('meta[name="theme-color"]')
			?.setAttribute('content', settings.theme === 'night' ? '#15181d' : '#eef0ea');
	});
</script>

<svelte:head>
	<link rel="icon" href={favicon} type="image/svg+xml" />
</svelte:head>

<TopBar />
{@render children()}
<ContentsDrawer />
<QuickJump />

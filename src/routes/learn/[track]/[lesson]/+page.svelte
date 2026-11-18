<script lang="ts">
	import LessonLayout from '#lib/components/lesson/LessonLayout.svelte';

	let { data } = $props();
</script>

<svelte:head><title>{data.lesson.title} · Neural Atlas</title></svelte:head>

{#key data.meta.id}
	<LessonLayout lesson={data.lesson} meta={data.meta} track={data.track} next={data.next} />
{/key}

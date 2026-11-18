<script lang="ts">
	export type IconName =
		| 'play'
		| 'step'
		| 'reset'
		| 'shuffle'
		| 'chevron-left'
		| 'chevron-right'
		| 'menu'
		| 'close'
		| 'check';

	interface Props {
		name: IconName;
		size?: number;
	}

	let { name, size = 16 }: Props = $props();

	const paths: Record<IconName, string> = {
		play: 'M4.5 2.8v10.4L13 8z',
		step: 'M3 2.8v10.4L10 8zM12.5 3v10',
		reset: 'M3 8a5 5 0 1 0 1.6-3.7M3 2.5v3h3',
		shuffle: 'M2 4.5h2.5c3 0 4 7 7 7H14M2 11.5h2.5c1 0 1.8-.8 2.4-1.8M9.1 6.3c.6-.9 1.4-1.8 2.4-1.8H14M12 2.8l2 1.7-2 1.7M12 9.8l2 1.7-2 1.7',
		'chevron-left': 'M10 3L5 8l5 5',
		'chevron-right': 'M6 3l5 5-5 5',
		menu: 'M2.5 4.5h11M2.5 8h11M2.5 11.5h11',
		close: 'M3.5 3.5l9 9M12.5 3.5l-9 9',
		check: 'M3 8.5l3.2 3.2L13 4.8'
	};
</script>

<svg
	width={size}
	height={size}
	viewBox="0 0 16 16"
	fill="none"
	stroke="currentColor"
	stroke-width="1.5"
	stroke-linecap="round"
	stroke-linejoin="round"
	aria-hidden="true"
>
	<path d={paths[name]} />
</svg>

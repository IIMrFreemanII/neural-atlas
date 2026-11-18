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
		| 'check'
		| 'search'
		| 'gear'
		| 'sun'
		| 'moon'
		| 'chevron-down';

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
		check: 'M3 8.5l3.2 3.2L13 4.8',
		search: 'M7 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10zM10.8 10.8L14 14',
		gear: 'M8 10.2a2.2 2.2 0 1 0 0-4.4 2.2 2.2 0 0 0 0 4.4zM8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4',
		sun: 'M8 10.6a2.6 2.6 0 1 0 0-5.2 2.6 2.6 0 0 0 0 5.2zM8 1.5v1.3M8 13.2v1.3M1.5 8h1.3M13.2 8h1.3M3.4 3.4l.9.9M11.7 11.7l.9.9M3.4 12.6l.9-.9M11.7 4.3l.9-.9',
		moon: 'M13.5 9.6A5.6 5.6 0 1 1 6.4 2.5a4.6 4.6 0 0 0 7.1 7.1z',
		'chevron-down': 'M3 6l5 5 5-5'
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

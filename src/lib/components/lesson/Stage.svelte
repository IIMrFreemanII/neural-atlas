<script lang="ts">
	import { onMount } from 'svelte';
	import { TONE_HEX } from '../../gpu/colors.ts';
	import { getGpu } from '../../gpu/device.ts';
	import { createFrameLoop } from '../../gpu/frame.ts';
	import { Renderer } from '../../gpu/renderer.ts';
	import type { Label, Scene, SceneFactory, ScenePointer, Tone } from '../../gpu/types.ts';
	import { settings } from '../../state/settings.svelte.ts';
	import { ViewUniforms } from '../../gpu/view.ts';

	interface Props {
		/**
		 * Creates the scene. Read once on mount; remount the component (e.g. with {#key}) to change scenes.
		 * The motion setting is also read once: wrap the Stage in {#key settings.reducedMotion} to follow it.
		 */
		createScene: SceneFactory;
		/** Complete current param set. scene.setParams is called on every change. */
		params?: Readonly<Record<string, number>>;
		/** Forwarded to scene.highlight(). */
		highlight?: Tone | null;
		/** When true, frames still render but dt is 0 and time does not advance. */
		paused?: boolean;
		/** Called when the scene emits a readout value. */
		onreadout?: (key: string, value: number | string) => void;
	}

	let { createScene, params = {}, highlight = null, paused = false, onreadout }: Props = $props();

	let container: HTMLDivElement;
	let canvas = $state<HTMLCanvasElement>();
	let status = $state<'loading' | 'ready' | 'nogpu' | 'error'>('loading');
	let labels = $state.raw<Label[]>([]);

	let scene: Scene | null = null;
	let ready = false;
	let loop: ReturnType<typeof createFrameLoop> | null = null;
	let onScreen = true;

	/** Forward a lesson action to scene.command(id). */
	export function command(id: string): void {
		scene?.command?.(id);
	}

	$effect(() => {
		const p = params;
		if (ready) scene?.setParams(p);
	});
	$effect(() => {
		const h = highlight;
		if (ready) scene?.highlight?.(h);
	});
	$effect(() => {
		const p = paused;
		if (loop) loop.paused = p;
	});

	function syncRunning() {
		if (!loop) return;
		if (onScreen && !document.hidden) loop.start();
		else loop.stop();
	}

	onMount(() => {
		let destroyed = false;
		let renderer: Renderer | null = null;
		let view: ViewUniforms | null = null;
		let resizeObs: ResizeObserver | null = null;
		let intersectObs: IntersectionObserver | null = null;

		const teardown = () => {
			loop?.stop();
			loop = null;
			ready = false;
			resizeObs?.disconnect();
			intersectObs?.disconnect();
			document.removeEventListener('visibilitychange', syncRunning);
			try {
				scene?.dispose();
			} catch (err) {
				console.error(err);
			}
			scene = null;
			view?.destroy();
			renderer?.destroy();
			view = null;
			renderer = null;
		};

		const applySize = (w: number, h: number) => {
			const dpr = Math.min(window.devicePixelRatio || 1, 2);
			renderer?.resize(w, h, dpr);
			view?.setSize(w, h, dpr);
		};

		void (async () => {
			const gpu = await getGpu();
			if (destroyed) return;
			if (!gpu) {
				status = 'nogpu';
				return;
			}
			try {
				if (!canvas) throw new Error('Stage canvas missing');
				renderer = new Renderer(gpu, canvas);
				view = new ViewUniforms(gpu);
				const r = renderer;
				const v = view;
				const rect = container.getBoundingClientRect();
				applySize(rect.width, rect.height);

				const s = createScene();
				scene = s;
				await s.init({
					gpu,
					target: r.target,
					view: v,
					labels: {
						set(list) {
							labels = list;
						}
					},
					emit: (key, value) => onreadout?.(key, value),
					reducedMotion: settings.reducedMotion
				});
				if (destroyed) return;
				s.setParams(params);
				s.highlight?.(highlight);
				ready = true;

				resizeObs = new ResizeObserver((entries) => {
					const box = entries[entries.length - 1].contentRect;
					applySize(box.width, box.height);
				});
				resizeObs.observe(container);

				loop = createFrameLoop((frame) => {
					try {
						v.setTime(frame.time);
						s.update(frame);
						v.flush();
						r.render((pass) => s.draw(pass));
					} catch (err) {
						console.error('[gpu] frame failed', err);
						loop?.stop();
					}
				});
				loop.paused = paused;

				intersectObs = new IntersectionObserver((entries) => {
					onScreen = entries[entries.length - 1].isIntersecting;
					syncRunning();
				});
				intersectObs.observe(container);
				document.addEventListener('visibilitychange', syncRunning);
				status = 'ready';
				syncRunning();
			} catch (err) {
				console.error(err);
				teardown();
				if (!destroyed) status = 'error';
			}
		})();

		return () => {
			destroyed = true;
			teardown();
		};
	});

	function sendPointer(type: ScenePointer['type'], e: PointerEvent) {
		if (!ready) return;
		const rect = container.getBoundingClientRect();
		scene?.pointer?.({ type, x: e.clientX - rect.left, y: e.clientY - rect.top });
	}

	const ALIGN = { center: '-50%, -50%', left: '0, -50%', right: '-100%, -50%' } as const;

	function labelStyle(l: Label): string {
		const t = l.tone ?? 'text';
		const font = l.mono
			? "var(--font-mono, ui-monospace, monospace)"
			: "var(--font-body, system-ui, sans-serif)";
		return (
			`transform: translate(${l.x}px, ${l.y}px) translate(${ALIGN[l.align ?? 'center']});` +
			`opacity: ${l.opacity ?? 1};` +
			`color: var(--c-${t}, ${TONE_HEX[t]});` +
			`font-family: ${font};` +
			(l.size ? `font-size: ${l.size}px;` : '')
		);
	}
</script>

<div
	class="stage"
	bind:this={container}
	role="presentation"
	onpointermove={(e) => sendPointer('move', e)}
	onpointerdown={(e) => sendPointer('down', e)}
	onpointerup={(e) => sendPointer('up', e)}
	onpointerleave={(e) => sendPointer('leave', e)}
>
	{#if status === 'nogpu' || status === 'error'}
		<div class="notice">
			{#if status === 'nogpu'}
				<h3>This visual needs WebGPU</h3>
				<p>Open this page in a current version of Chrome, Edge or Safari to see it.</p>
			{:else}
				<h3>This visual couldn't start.</h3>
			{/if}
		</div>
	{:else}
		<canvas bind:this={canvas} aria-hidden="true"></canvas>
		<div class="labels">
			{#each labels as l (l.id)}
				<span style={labelStyle(l)}>{l.text}</span>
			{/each}
		</div>
	{/if}
</div>

<style>
	.stage {
		position: relative;
		width: 100%;
		height: 100%;
		overflow: hidden;
	}
	canvas {
		display: block;
		width: 100%;
		height: 100%;
	}
	.labels {
		position: absolute;
		inset: 0;
		overflow: hidden;
		pointer-events: none;
	}
	.labels span {
		position: absolute;
		left: 0;
		top: 0;
		font-size: 12px;
		white-space: nowrap;
		will-change: transform;
		/* A dark halo keeps labels readable where they cross bright lines. */
		text-shadow:
			0 0 3px var(--bg-void, #07080d),
			0 0 6px var(--bg-void, #07080d),
			0 0 1px var(--bg-void, #07080d);
	}
	.notice {
		position: absolute;
		inset: 0;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 6px;
		padding: 16px;
		text-align: center;
		color: var(--c-text, #e8eaf2);
	}
	.notice h3 {
		margin: 0;
		font-size: 16px;
		font-weight: 600;
	}
	.notice p {
		margin: 0;
		font-size: 13px;
		color: var(--c-muted, #5c647a);
	}
</style>

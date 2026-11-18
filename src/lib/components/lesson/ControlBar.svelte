<script lang="ts">
	import { flushSync, tick } from 'svelte';
	import type { Control, Params } from '../../lessons/types.ts';
	import Icon from '../shell/Icon.svelte';
	import Slider from './Slider.svelte';

	interface Props {
		controls: Control[];
		params: Params;
		/** Current values of 'text' controls by id. A missing id shows the control's initial text. */
		texts?: Record<string, string>;
		onparam: (key: string, value: number) => void;
		onaction: (id: string) => void;
		ontext?: (id: string, value: string) => void;
		/** Id of the action the space bar presses, if any. Only used to announce the shortcut. */
		spaceAction?: string;
	}

	let { controls, params, texts = {}, onparam, onaction, ontext, spaceAction }: Props = $props();

	const firstAction = $derived(controls.findIndex((c) => c.kind === 'action'));
	/** Sliders form the lower row of a busy bar; everything else the upper row. */
	const others = $derived(controls.filter((c) => c.kind !== 'slider'));
	const sliders = $derived(controls.filter((c) => c.kind === 'slider'));

	let wrap: HTMLDivElement | undefined = $state();
	let bar: HTMLDivElement | undefined = $state();
	/** The single row would not fit the room it has, so the bar is split in two rows. */
	let rows = $state(false);

	/** Try one row; keep it only if it fits. Runs synchronously, so no frame is painted with the trial layout. */
	function measure() {
		if (!wrap || !bar) return;
		if (rows) {
			rows = false;
			flushSync();
		}
		const fits = bar.offsetWidth <= wrap.clientWidth;
		if (!fits) {
			rows = true;
			flushSync();
		}
	}

	$effect(() => {
		if (!wrap) return;
		const ro = new ResizeObserver(() => measure());
		ro.observe(wrap);
		return () => ro.disconnect();
	});
	// A different set of controls (another page) can need a different layout.
	$effect(() => {
		void controls;
		void tick().then(measure);
	});
	$effect(() => {
		void document.fonts?.ready.then(measure);
	});

	function keyOf(c: Control): string {
		return 'id' in c ? 'a:' + c.id : c.key;
	}
</script>

{#snippet item(c: Control, i: number)}
	{#if c.kind === 'slider'}
		<Slider control={c} value={params[c.key] ?? c.min} onchange={(v) => onparam(c.key, v)} />
	{:else if c.kind === 'toggle'}
		{@const on = (params[c.key] ?? 0) >= 0.5}
		<div class="toggle">
			<span class="lbl">{c.label}</span>
			<button
				type="button"
				role="switch"
				aria-checked={on}
				aria-label={c.label}
				class="switch"
				class:on
				onclick={() => onparam(c.key, on ? 0 : 1)}
			><span class="knob"></span></button>
		</div>
	{:else if c.kind === 'select'}
		<div class="select" role="group" aria-label={c.label}>
			<span class="lbl">{c.label}</span>
			<div class="seg">
				{#each c.options as o (o.value)}
					<button
						type="button"
						aria-pressed={params[c.key] === o.value}
						onclick={() => onparam(c.key, o.value)}>{o.label}</button
					>
				{/each}
			</div>
		</div>
	{:else if c.kind === 'text'}
		<label class="text">
			<span class="top">
				<span class="lbl">{c.label}</span>
				<span class="lbl num">{(texts[c.id] ?? c.initial).length}/{c.maxLength ?? 60}</span>
			</span>
			<input
				type="text"
				value={texts[c.id] ?? c.initial}
				maxlength={c.maxLength ?? 60}
				spellcheck="false"
				autocomplete="off"
				oninput={(e) => ontext?.(c.id, e.currentTarget.value)}
			/>
		</label>
	{:else}
		<button
			type="button"
			class="act"
			class:primary={i === firstAction}
			title={c.id === spaceAction ? 'Shortcut: space bar' : undefined}
			aria-keyshortcuts={c.id === spaceAction ? 'Space' : undefined}
			onclick={() => onaction(c.id)}
		>
			{#if c.icon}<Icon name={c.icon} size={13} />{/if}
			{c.label}
		</button>
	{/if}
{/snippet}

<div class="wrap" bind:this={wrap}>
	<div class="bar" class:rows role="group" aria-label="Controls" bind:this={bar}>
		<div class="row">
			{#each others as c (keyOf(c))}
				{@render item(c, controls.indexOf(c))}
			{/each}
		</div>
		{#if others.length && sliders.length}<span class="sep" aria-hidden="true"></span>{/if}
		<div class="row">
			{#each sliders as c (keyOf(c))}
				{@render item(c, controls.indexOf(c))}
			{/each}
		</div>
	</div>
</div>

<style>
	.wrap {
		display: flex;
		justify-content: center;
		width: 100%;
	}
	/* One row: the two groups sit side by side. Busy: they stack, each centred. */
	.bar {
		display: flex;
		flex: none;
		align-items: center;
		gap: 22px;
		padding: 12px 20px;
		border: 1px solid var(--rule-2);
		border-radius: 6px;
		background-color: var(--sheet);
		background-image: var(--grain);
		box-shadow: var(--shadow);
		pointer-events: auto;
	}
	.bar.rows {
		flex-direction: column;
		gap: 12px;
	}
	.row {
		display: flex;
		align-items: center;
		gap: 22px;
	}
	.row:empty {
		display: none;
	}
	.sep {
		align-self: stretch;
		width: 1px;
		background: var(--rule);
	}
	.rows .sep {
		display: none;
	}
	.lbl {
		font-size: 12px;
		color: var(--ink-2);
		white-space: nowrap;
	}
	.text {
		display: grid;
		gap: 5px;
		width: 290px;
	}
	.text .top {
		display: flex;
		justify-content: space-between;
	}
	.text input {
		height: 32px;
		min-width: 0;
		padding: 0 10px;
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		background: var(--sheet);
		color: var(--ink);
		font-family: var(--font-body);
		font-size: 13.5px;
	}
	.text input:focus-visible {
		outline: none;
		border-color: var(--ink);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--data) 25%, transparent);
	}
	.toggle {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		white-space: nowrap;
	}
	.toggle .lbl {
		font-size: 13px;
		color: var(--ink);
	}
	.switch {
		position: relative;
		flex: none;
		width: 30px;
		height: 18px;
		padding: 0;
		border: 0;
		border-radius: 9px;
		background: var(--rule-2);
		cursor: pointer;
		transition: background-color 0.15s;
	}
	.switch.on {
		background: var(--ink);
	}
	.knob {
		position: absolute;
		top: 2px;
		left: 2px;
		width: 14px;
		height: 14px;
		border-radius: 50%;
		background: var(--sheet);
		box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
		transition: transform 0.15s;
	}
	.switch.on .knob {
		transform: translateX(12px);
	}
	.select {
		display: grid;
		gap: 5px;
	}
	.seg {
		display: inline-flex;
		width: max-content;
		overflow: hidden;
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		background: var(--sheet);
	}
	.seg button {
		padding: 4px 10px;
		border: 0;
		border-radius: 0;
		background: transparent;
		font-size: 12.5px;
		color: var(--ink-2);
		white-space: nowrap;
		cursor: pointer;
	}
	.seg button + button {
		border-left: 1px solid var(--rule);
	}
	.seg button:hover {
		color: var(--ink);
	}
	.seg button[aria-pressed='true'] {
		background: var(--ink);
		color: var(--pg);
	}
	.act {
		display: inline-flex;
		align-items: center;
		gap: 7px;
		height: 32px;
		padding: 0 12px;
		border: 1px solid var(--rule-2);
		border-radius: 4px;
		background: var(--sheet);
		color: var(--ink);
		font-size: 13px;
		font-weight: 500;
		white-space: nowrap;
		cursor: pointer;
	}
	.act:hover {
		background: color-mix(in srgb, var(--ink) 7%, var(--sheet));
	}
	.act.primary {
		background: var(--btn-ink);
		border-color: var(--btn-ink);
		color: var(--btn-on-ink);
	}
	.act.primary:hover {
		opacity: 0.88;
	}
</style>

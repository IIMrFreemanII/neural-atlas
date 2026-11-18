<script lang="ts">
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
</script>

<div class="bar glass" role="group" aria-label="Controls">
	{#each controls as c, i ('id' in c ? 'a:' + c.id : c.key)}
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
				<span class="lbl">{c.label}</span>
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
				class="btn act"
				class:primary={i === firstAction}
				title={c.id === spaceAction ? 'Shortcut: space bar' : undefined}
				aria-keyshortcuts={c.id === spaceAction ? 'Space' : undefined}
				onclick={() => onaction(c.id)}
			>
				{#if c.icon}<Icon name={c.icon} />{/if}
				{c.label}
			</button>
		{/if}
	{/each}
</div>

<style>
	.bar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: center;
		gap: 8px 14px;
		padding: 8px 14px;
		max-width: calc(100% - 32px);
		border-radius: 22px;
	}
	.lbl {
		font-size: 12px;
		color: var(--c-text-2);
	}
	.text {
		display: flex;
		align-items: center;
		gap: 10px;
		width: 440px;
		max-width: 100%;
	}
	.text input {
		flex: 1;
		min-width: 0;
		height: 28px;
		padding: 0 12px;
		border-radius: var(--r-pill);
		border: 1px solid var(--line-strong);
		background: var(--glass);
		color: var(--c-text);
		font-family: var(--font-mono, ui-monospace, monospace);
		font-size: 13px;
	}
	.text input:focus-visible {
		outline: none;
		border-color: var(--c-data);
	}
	.toggle,
	.select {
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.switch {
		position: relative;
		width: 32px;
		height: 18px;
		padding: 0;
		border-radius: var(--r-pill);
		border: 1px solid var(--line-strong);
		background: var(--glass-strong);
		cursor: pointer;
		transition: background 0.15s;
	}
	.switch.on {
		background: var(--c-data);
		border-color: var(--c-data);
	}
	.knob {
		position: absolute;
		top: 2px;
		left: 2px;
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--c-text);
		transition: transform 0.15s;
	}
	.switch.on .knob {
		transform: translateX(14px);
		background: var(--bg-void);
	}
	.seg {
		display: flex;
		padding: 2px;
		gap: 2px;
		border-radius: var(--r-pill);
		border: 1px solid var(--line);
		background: var(--glass);
	}
	.seg button {
		height: 24px;
		padding: 0 12px;
		border: 0;
		border-radius: var(--r-pill);
		background: transparent;
		color: var(--c-text-2);
		font-size: 12px;
		cursor: pointer;
	}
	.seg button:hover {
		color: var(--c-text);
	}
	.seg button[aria-pressed='true'] {
		background: var(--glass-strong);
		color: var(--c-text);
		box-shadow: inset 0 0 0 1px var(--line-strong);
	}
	.act {
		height: 32px;
		padding: 0 14px;
		font-size: 13px;
	}
</style>

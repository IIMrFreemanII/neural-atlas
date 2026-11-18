<script lang="ts">
	import type { Control } from '../../lessons/types.ts';

	type SliderControl = Extract<Control, { kind: 'slider' }>;

	interface Props {
		control: SliderControl;
		value: number;
		onchange: (value: number) => void;
	}

	let { control, value, onchange }: Props = $props();

	const span = $derived(control.max - control.min);
	const at = (v: number) => (span === 0 ? 0 : ((v - control.min) / span) * 100);
	const pct = $derived(Math.min(100, Math.max(0, at(value))));
	/** A range that crosses zero fills from zero and shows a tick there; any other range fills from its start. */
	const hasZero = $derived(control.min < 0 && control.max > 0);
	const origin = $derived(hasZero ? at(0) : 0);
	const inputId = $props.id();
</script>

<div
	class="slider"
	class:zero={hasZero}
	style:--tone="var(--c-{control.tone ?? 'param'})"
	style:--from="{Math.min(origin, pct)}%"
	style:--to="{Math.max(origin, pct)}%"
	style:--zero="{origin}%"
>
	<div class="top">
		<label for={inputId}>{control.label}</label>
		<output class="num" for={inputId}>{value.toFixed(control.digits ?? 2)}</output>
	</div>
	<input
		id={inputId}
		type="range"
		min={control.min}
		max={control.max}
		step={control.step}
		{value}
		oninput={(e) => onchange(e.currentTarget.valueAsNumber)}
	/>
</div>

<style>
	.slider {
		position: relative;
		display: grid;
		gap: 2px;
		width: 128px;
		flex: none;
	}
	.top {
		display: flex;
		justify-content: space-between;
		gap: 8px;
		font-size: 12px;
		color: var(--ink-2);
		white-space: nowrap;
	}
	label {
		color: var(--ink-2);
	}
	output {
		font-weight: 500;
		color: var(--ink);
	}
	input {
		-webkit-appearance: none;
		appearance: none;
		width: 100%;
		height: 18px;
		margin: 0;
		background: transparent;
		cursor: grab;
	}
	input:active {
		cursor: grabbing;
	}
	/* The tick where the range crosses zero. */
	.zero::after {
		content: '';
		position: absolute;
		left: var(--zero);
		bottom: 7px;
		width: 1px;
		height: 9px;
		background: var(--ink-3);
		pointer-events: none;
	}
	input::-webkit-slider-runnable-track {
		height: 3px;
		border-radius: 2px;
		background: linear-gradient(
			to right,
			var(--rule-2) 0,
			var(--rule-2) var(--from),
			var(--tone) var(--from),
			var(--tone) var(--to),
			var(--rule-2) var(--to),
			var(--rule-2) 100%
		);
	}
	input::-moz-range-track {
		height: 3px;
		border-radius: 2px;
		background: var(--rule-2);
	}
	input::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 13px;
		height: 13px;
		margin-top: -5px;
		box-sizing: border-box;
		border-radius: 50%;
		border: 2px solid var(--sheet);
		background: var(--tone);
		box-shadow: 0 0 0 1px var(--tone);
	}
	input::-moz-range-thumb {
		width: 13px;
		height: 13px;
		box-sizing: border-box;
		border-radius: 50%;
		border: 2px solid var(--sheet);
		background: var(--tone);
		box-shadow: 0 0 0 1px var(--tone);
	}
	input:focus-visible {
		outline: none;
	}
	input:focus-visible::-webkit-slider-thumb {
		box-shadow:
			0 0 0 1px var(--tone),
			0 0 0 6px color-mix(in srgb, var(--tone) 24%, transparent);
	}
	input:focus-visible::-moz-range-thumb {
		box-shadow:
			0 0 0 1px var(--tone),
			0 0 0 6px color-mix(in srgb, var(--tone) 24%, transparent);
	}
</style>

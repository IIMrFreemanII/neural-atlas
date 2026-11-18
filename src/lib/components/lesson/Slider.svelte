<script lang="ts">
	import type { Control } from '../../lessons/types.ts';

	type SliderControl = Extract<Control, { kind: 'slider' }>;

	interface Props {
		control: SliderControl;
		value: number;
		onchange: (value: number) => void;
	}

	let { control, value, onchange }: Props = $props();

	const pct = $derived(
		control.max === control.min ? 0 : ((value - control.min) / (control.max - control.min)) * 100
	);
	const inputId = $props.id();
</script>

<div class="slider" style:--tone="var(--c-{control.tone ?? 'param'})" style:--pct="{pct}%">
	<label for={inputId}>{control.label}</label>
	<input
		id={inputId}
		type="range"
		min={control.min}
		max={control.max}
		step={control.step}
		{value}
		oninput={(e) => onchange(e.currentTarget.valueAsNumber)}
	/>
	<output class="num" for={inputId}>{value.toFixed(control.digits ?? 2)}</output>
</div>

<style>
	.slider {
		display: grid;
		grid-template-columns: auto 120px auto;
		align-items: center;
		gap: 10px;
		padding: 2px 6px;
	}
	label {
		font-size: 12px;
		color: var(--c-text-2);
		white-space: nowrap;
	}
	output {
		font-size: 12px;
		min-width: 3.4ch;
		text-align: right;
		color: var(--c-text);
	}
	input {
		-webkit-appearance: none;
		appearance: none;
		width: 100%;
		height: 20px;
		margin: 0;
		background: transparent;
		cursor: pointer;
	}
	input::-webkit-slider-runnable-track {
		height: 3px;
		border-radius: 2px;
		background: linear-gradient(
			to right,
			var(--tone) 0,
			var(--tone) var(--pct),
			var(--line-strong) var(--pct),
			var(--line-strong) 100%
		);
	}
	input::-moz-range-track {
		height: 3px;
		border-radius: 2px;
		background: var(--line-strong);
	}
	input::-moz-range-progress {
		height: 3px;
		border-radius: 2px;
		background: var(--tone);
	}
	input::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 14px;
		height: 14px;
		margin-top: -5.5px;
		border-radius: 50%;
		border: 0;
		background: var(--c-text);
	}
	input::-moz-range-thumb {
		width: 14px;
		height: 14px;
		border-radius: 50%;
		border: 0;
		background: var(--c-text);
	}
</style>

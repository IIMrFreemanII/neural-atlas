<script lang="ts">
	import { tick } from 'svelte';
	import { settings } from '../../state/settings.svelte.ts';

	interface Props {
		open: boolean;
		/** The gear button: clicks on it are not "outside", and focus goes back to it on Escape. */
		anchor: HTMLElement | undefined;
		onclose: () => void;
	}

	let { open, anchor, onclose }: Props = $props();

	let panel: HTMLDivElement | undefined = $state();
	let mod = $state('⌘');

	$effect(() => {
		mod = /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘' : 'Ctrl';
	});

	$effect(() => {
		if (!open) return;
		void tick().then(() => panel?.querySelector<HTMLElement>('input, button')?.focus());
	});

	function onkeydown(e: KeyboardEvent): void {
		if (!open || e.key !== 'Escape') return;
		e.preventDefault();
		onclose();
		anchor?.focus();
	}

	function onpointerdown(e: PointerEvent): void {
		if (!open) return;
		const t = e.target as Node;
		if (panel?.contains(t) || anchor?.contains(t)) return;
		onclose();
	}

	function toggleFollow(e: Event): void {
		const on = (e.currentTarget as HTMLInputElement).checked;
		settings.setTheme(on ? null : settings.theme);
	}
</script>

<svelte:window {onkeydown} {onpointerdown} />

{#if open}
	<div class="menu" bind:this={panel} role="dialog" aria-label="Settings">
		<section>
			<h4>Theme</h4>
			<label class="check">
				<input type="checkbox" checked={settings.themeChoice === null} onchange={toggleFollow} />
				<span>Follow system theme</span>
			</label>
			<p class="note">Picking Paper or Night in the bar turns this off.</p>
		</section>
		<section>
			<h4>Motion</h4>
			<div class="row">
				<span id="reduce-motion-label">Reduce motion</span>
				<button
					type="button"
					role="switch"
					class="switch"
					class:on={settings.reducedMotion}
					aria-checked={settings.reducedMotion}
					aria-labelledby="reduce-motion-label"
					onclick={() => settings.setReducedMotion(!settings.reducedMotion)}
				><span class="knob"></span></button>
			</div>
			<p class="note">Stops decorative animation and makes slides and transitions instant.</p>
		</section>
		<section>
			<h4>Keyboard</h4>
			<div class="keys">
				<span><kbd>←</kbd><kbd>→</kbd></span><span>Previous or next step</span>
				<span><kbd>Space</kbd></span><span>The main action, such as Train or Fit</span>
				<span><kbd>{mod}</kbd><kbd>K</kbd></span><span>Search lessons, steps and terms</span>
				<span><kbd>Esc</kbd></span><span>Close a menu or the deep dive</span>
			</div>
		</section>
	</div>
{/if}

<style>
	.menu {
		position: fixed;
		top: calc(var(--topbar-h) + 4px);
		right: 16px;
		z-index: 40;
		width: 330px;
		padding: 4px 0;
		background-color: var(--sheet);
		background-image: var(--grain);
		border: 1px solid var(--rule-2);
		border-radius: 6px;
		box-shadow: var(--shadow);
		font-size: 13px;
		color: var(--ink);
	}
	section {
		display: grid;
		gap: 8px;
		padding: 12px 16px;
	}
	section + section {
		border-top: 1px solid var(--rule);
	}
	h4 {
		margin: 0;
		font-size: 13px;
		font-weight: 500;
	}
	.note {
		margin: 0;
		font-size: 12px;
		line-height: 1.45;
		color: var(--ink-2);
	}
	.check {
		display: flex;
		align-items: center;
		gap: 8px;
		cursor: pointer;
	}
	.check input {
		margin: 0;
		accent-color: var(--ink);
	}
	.row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
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
		transition: left 0.15s;
	}
	.switch.on {
		background: var(--ink);
	}
	.switch.on .knob {
		left: 14px;
	}
	:global([data-reduce-motion]) .knob {
		transition: none;
	}
	.keys {
		display: grid;
		grid-template-columns: auto 1fr;
		align-items: center;
		gap: 6px 12px;
		color: var(--ink-2);
	}
	.keys > span:nth-child(odd) {
		display: flex;
		gap: 4px;
	}
	kbd {
		font-family: var(--font-mono);
		font-size: 10.5px;
		padding: 1px 5px;
		border: 1px solid var(--rule-2);
		border-radius: 3px;
		color: var(--ink-2);
		background: var(--sheet);
	}
</style>

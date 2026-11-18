import type { Gpu } from './types.ts';

let cached: Promise<Gpu | null> | null = null;

export function isWebGpuSupported(): boolean {
	return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

async function createGpu(): Promise<Gpu | null> {
	if (!isWebGpuSupported()) return null;
	try {
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) return null;
		const device = await adapter.requestDevice({ label: 'neural-atlas device' });
		device.lost.then((info) => {
			console.error('[gpu] device lost', info.reason, info.message);
			cached = null;
		});
		device.addEventListener('uncapturederror', (e) => {
			console.error('[gpu]', (e as GPUUncapturedErrorEvent).error.message);
		});
		return { device, format: navigator.gpu.getPreferredCanvasFormat() };
	} catch (err) {
		console.error('[gpu] initialisation failed', err);
		return null;
	}
}

/** App-wide singleton. Resolves to null when WebGPU is unavailable. */
export function getGpu(): Promise<Gpu | null> {
	if (!cached) {
		const p = createGpu();
		cached = p;
		void p.then((g) => {
			if (!g && cached === p) cached = null;
		});
	}
	return cached;
}

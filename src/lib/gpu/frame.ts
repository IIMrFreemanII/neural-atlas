import type { FrameInfo } from './types.ts';

export function createFrameLoop(cb: (frame: FrameInfo) => void) {
	let raf = 0;
	let running = false;
	let last = -1;
	let time = 0;

	const tick = (now: number) => {
		if (!running) return;
		raf = requestAnimationFrame(tick);
		const dt = last < 0 ? 0 : Math.min(0.1, Math.max(0, (now - last) / 1000));
		last = now;
		if (api.paused) {
			cb({ time, dt: 0 });
		} else {
			time += dt;
			cb({ time, dt });
		}
	};

	const api = {
		start() {
			if (running) return;
			running = true;
			last = -1;
			raf = requestAnimationFrame(tick);
		},
		stop() {
			running = false;
			cancelAnimationFrame(raf);
		},
		paused: false
	};
	return api;
}

/** Loss samples for the chart: one per step, thinned (every second sample kept) once past MAX_SAMPLES. */
export const MAX_SAMPLES = 400;

export class LossHistory {
	values: number[] = [];
	/** A sample is kept every `interval` steps. */
	interval = 1;
	/** Largest loss seen since the last reset. */
	max = 0;
	private count = 0;

	reset(initial: number): void {
		this.values = [];
		this.interval = 1;
		this.count = 0;
		this.max = Number.isFinite(initial) ? initial : 0;
	}

	push(loss: number): void {
		if (!Number.isFinite(loss)) return;
		this.count++;
		if (loss > this.max) this.max = loss;
		if (this.count % this.interval !== 0) return;
		this.values.push(loss);
		if (this.values.length > MAX_SAMPLES) {
			this.values = this.values.filter((_, i) => i % 2 === 1);
			this.interval *= 2;
		}
	}
}

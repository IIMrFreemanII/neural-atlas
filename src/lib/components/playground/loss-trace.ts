export const MAX_POINTS = 300;

/** The loss samples the playground card draws, in step order. Pure data, no rendering. */
export class LossTrace {
	/** [steps, loss] samples in step order. */
	points: Array<[number, number]> = [];
	/** Largest loss seen since the last clear (0 when empty). */
	max = 0;

	/**
	 * Record the scene's latest values. A steps value lower than the last one (Reset, new data, remount)
	 * clears the trace first. A non-finite loss is ignored.
	 */
	add(steps: number, loss: number): void {
		if (!Number.isFinite(loss) || !Number.isFinite(steps)) return;
		const last = this.points[this.points.length - 1];
		if (last && steps < last[0]) this.clear();
		this.points.push([steps, loss]);
		if (loss > this.max) this.max = loss;
		if (this.points.length > MAX_POINTS) {
			// Keep every second point, but always the last one.
			const end = this.points.length - 1;
			this.points = this.points.filter((_, i) => i % 2 === 0 || i === end);
		}
	}

	clear(): void {
		this.points = [];
		this.max = 0;
	}
}

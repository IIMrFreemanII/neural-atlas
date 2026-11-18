const STORAGE_KEY = 'neural-atlas:progress:v1';

interface Stored {
	done: Record<string, true>;
	step: Record<string, number>;
}

function load(): Stored {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (raw) {
			const p = JSON.parse(raw) as Partial<Stored>;
			return { done: p.done ?? {}, step: p.step ?? {} };
		}
	} catch {
		/* storage unavailable or corrupt */
	}
	return { done: {}, step: {} };
}

const data: Stored = $state(load());

function save() {
	try {
		localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
	} catch {
		/* storage unavailable */
	}
}

export const progress = {
	isComplete(key: string): boolean {
		return data.done[key] === true;
	},
	complete(key: string): void {
		if (data.done[key]) return;
		data.done[key] = true;
		save();
	},
	getStep(key: string): number {
		return data.step[key] ?? 0;
	},
	setStep(key: string, index: number): void {
		if (data.step[key] === index) return;
		data.step[key] = index;
		save();
	},
	get completedCount(): number {
		return Object.keys(data.done).length;
	}
};

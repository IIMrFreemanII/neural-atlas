const STORAGE_KEY = 'neural-atlas:progress:v1';

interface Stored {
	done: Record<string, true>;
	step: Record<string, number>;
	/** Lesson key to the 0-based indices of its solved questions. */
	solved: Record<string, number[]>;
}

/** Keeps only entries that are arrays of non-negative integers; anything else is dropped. */
function cleanSolved(raw: unknown): Record<string, number[]> {
	const out: Record<string, number[]> = {};
	if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return out;
	for (const [k, v] of Object.entries(raw)) {
		if (!Array.isArray(v)) continue;
		out[k] = v.filter((n): n is number => Number.isInteger(n) && n >= 0).sort((a, b) => a - b);
	}
	return out;
}

function load(): Stored {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (raw) {
			const p = JSON.parse(raw) as Partial<Stored>;
			return { done: p.done ?? {}, step: p.step ?? {}, solved: cleanSolved(p.solved) };
		}
	} catch {
		/* storage unavailable or corrupt */
	}
	return { done: {}, step: {}, solved: {} };
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
	isSolved(key: string, index: number): boolean {
		return data.solved[key]?.includes(index) === true;
	},
	solve(key: string, index: number): void {
		const list = data.solved[key] ?? [];
		if (list.includes(index)) return;
		data.solved[key] = [...list, index].sort((a, b) => a - b);
		save();
	},
	get completedCount(): number {
		return Object.keys(data.done).length;
	}
};

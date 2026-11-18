const STORAGE_KEY = 'neural-atlas:settings:v1';

interface Stored {
	/** The learner's own choice. null = follow the operating system. */
	reducedMotion: boolean | null;
}

function load(): Stored {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (raw) {
			const p = JSON.parse(raw) as Partial<Stored>;
			return { reducedMotion: typeof p.reducedMotion === 'boolean' ? p.reducedMotion : null };
		}
	} catch {
		/* storage unavailable or corrupt */
	}
	return { reducedMotion: null };
}

const data: Stored = $state(load());

const query =
	typeof window !== 'undefined' && typeof window.matchMedia === 'function'
		? window.matchMedia('(prefers-reduced-motion: reduce)')
		: null;
let system = $state(query?.matches ?? false);
query?.addEventListener('change', (e) => (system = e.matches));

function save() {
	try {
		localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
	} catch {
		/* storage unavailable */
	}
}

export const settings = {
	/**
	 * True when motion should be reduced: the learner's choice in the top bar, or the operating
	 * system's "reduce motion" preference when they have not chosen. Reactive.
	 */
	get reducedMotion(): boolean {
		return data.reducedMotion ?? system;
	},
	setReducedMotion(value: boolean): void {
		data.reducedMotion = value;
		save();
	}
};

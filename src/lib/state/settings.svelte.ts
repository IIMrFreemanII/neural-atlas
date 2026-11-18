export const STORAGE_KEY = 'neural-atlas:settings:v1';

export type Theme = 'paper' | 'night';

interface Stored {
	/** The learner's own choice. null = follow the operating system. */
	reducedMotion: boolean | null;
	/** The learner's own choice. null = follow the operating system. */
	theme: Theme | null;
}

function load(): Stored {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (raw) {
			const p = JSON.parse(raw) as Partial<Stored>;
			return {
				reducedMotion: typeof p.reducedMotion === 'boolean' ? p.reducedMotion : null,
				theme: p.theme === 'paper' || p.theme === 'night' ? p.theme : null
			};
		}
	} catch {
		/* storage unavailable or corrupt */
	}
	return { reducedMotion: null, theme: null };
}

const data: Stored = $state(load());

function media(q: string): MediaQueryList | null {
	return typeof window !== 'undefined' && typeof window.matchMedia === 'function'
		? window.matchMedia(q)
		: null;
}

const motionQuery = media('(prefers-reduced-motion: reduce)');
let systemReduced = $state(motionQuery?.matches ?? false);
motionQuery?.addEventListener('change', (e) => (systemReduced = e.matches));

const darkQuery = media('(prefers-color-scheme: dark)');
let systemDark = $state(darkQuery?.matches ?? false);
darkQuery?.addEventListener('change', (e) => (systemDark = e.matches));

function save() {
	try {
		localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
	} catch {
		/* storage unavailable */
	}
}

export const settings = {
	/**
	 * True when motion should be reduced: the learner's choice, or the operating
	 * system's "reduce motion" preference when they have not chosen. Reactive.
	 */
	get reducedMotion(): boolean {
		return data.reducedMotion ?? systemReduced;
	},
	setReducedMotion(value: boolean): void {
		data.reducedMotion = value;
		save();
	},
	/** The theme in effect: the learner's choice, else the system's light/dark preference. Reactive. */
	get theme(): Theme {
		return data.theme ?? (systemDark ? 'night' : 'paper');
	},
	/** The stored choice; null means the theme follows the system. Reactive. */
	get themeChoice(): Theme | null {
		return data.theme;
	},
	/** Pick a theme, or null to follow the system again. */
	setTheme(value: Theme | null): void {
		data.theme = value;
		save();
	}
};

import type { Tone } from './types.ts';

export type PaletteName = 'paper' | 'night';

export interface Palette {
	/** Stage background ("paper"), sRGB hex. */
	pg: string;
	/** Raised surface behind cards and sheets, sRGB hex. */
	sheet: string;
	/** Ink colour per semantic tone, sRGB hex. */
	tones: Record<Tone, string>;
	/** Colour of cast shadows, sRGB hex. */
	cast: string;
	/** Opacity of cast shadows. */
	castAlpha: number;
}

export const PALETTES: Record<PaletteName, Palette> = {
	paper: {
		pg: '#eef0ea',
		sheet: '#fbfbf8',
		tones: {
			text: '#1b2130',
			muted: '#687080',
			data: '#0b7a8a',
			param: '#4a3dbf',
			grad: '#b9590c',
			neg: '#c0244a',
			ok: '#2c8650'
		},
		cast: '#1b2130',
		castAlpha: 0.14
	},
	night: {
		pg: '#15181d',
		sheet: '#1e2228',
		tones: {
			text: '#e9ecf1',
			muted: '#8a93a1',
			data: '#7fdcef',
			param: '#b7a8ff',
			grad: '#f4b968',
			neg: '#ff8fae',
			ok: '#8fe3b4'
		},
		cast: '#000000',
		castAlpha: 0.42
	}
};

let active: PaletteName = 'paper';

/** Select the palette every colour helper reads from. Call before a scene is created. */
export function setPalette(name: PaletteName): void {
	active = name;
}

export function palette(): Palette {
	return PALETTES[active];
}

export function paletteName(): PaletteName {
	return active;
}

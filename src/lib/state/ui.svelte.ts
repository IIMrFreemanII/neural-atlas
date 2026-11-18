/**
 * Shell UI state shared by the top bar, the Contents drawer, quick jump and the lesson page,
 * so those components do not import each other.
 */

const CARD_KEY = 'neural-atlas:card-folded:v1';

function loadFolded(): boolean {
	try {
		return localStorage.getItem(CARD_KEY) === '1';
	} catch {
		return false;
	}
}

const state = $state({
	contentsOpen: false,
	quickJumpOpen: false,
	cardFolded: loadFolded()
});

export const ui = {
	/** The Contents drawer (lesson list sliding over the page). Reactive. */
	get contentsOpen(): boolean {
		return state.contentsOpen;
	},
	setContentsOpen(open: boolean): void {
		state.contentsOpen = open;
		if (open) state.quickJumpOpen = false;
	},
	toggleContents(): void {
		ui.setContentsOpen(!state.contentsOpen);
	},

	/** The ⌘K quick jump dialog. Reactive. */
	get quickJumpOpen(): boolean {
		return state.quickJumpOpen;
	},
	setQuickJumpOpen(open: boolean): void {
		state.quickJumpOpen = open;
		if (open) state.contentsOpen = false;
	},

	/** The lesson narration card is folded into a tab. Remembered in this browser. Reactive. */
	get cardFolded(): boolean {
		return state.cardFolded;
	},
	setCardFolded(folded: boolean): void {
		state.cardFolded = folded;
		try {
			localStorage.setItem(CARD_KEY, folded ? '1' : '0');
		} catch {
			/* storage unavailable */
		}
	}
};

import type { Tone } from '../gpu/types.ts';

export type Inline =
	| { kind: 'text'; text: string }
	| { kind: 'bold'; text: string }
	| { kind: 'term'; tone: Tone; text: string };

const TONES: readonly string[] = ['text', 'muted', 'data', 'param', 'grad', 'neg', 'ok'];

function parseParagraph(src: string): Inline[] {
	const out: Inline[] = [];
	let buf = '';
	const flush = () => {
		if (buf) out.push({ kind: 'text', text: buf });
		buf = '';
	};
	let i = 0;
	while (i < src.length) {
		if (src.startsWith('**', i)) {
			const end = src.indexOf('**', i + 2);
			if (end > i + 2) {
				flush();
				out.push({ kind: 'bold', text: src.slice(i + 2, end) });
				i = end + 2;
				continue;
			}
		} else if (src.startsWith('[[', i)) {
			const end = src.indexOf(']]', i + 2);
			if (end !== -1) {
				const inner = src.slice(i + 2, end);
				const colon = inner.indexOf(':');
				const tone = colon > 0 ? inner.slice(0, colon) : '';
				const text = colon > 0 ? inner.slice(colon + 1) : '';
				if (text && TONES.includes(tone)) {
					flush();
					out.push({ kind: 'term', tone: tone as Tone, text });
					i = end + 2;
					continue;
				}
			}
		}
		buf += src[i];
		i++;
	}
	flush();
	return out;
}

/** Parse narration markup into one array of inline runs per paragraph. */
export function parseBody(body: string): Inline[][] {
	return body
		.split(/\n\s*\n/)
		.map((p) => p.trim())
		.filter((p) => p.length > 0)
		.map(parseParagraph);
}

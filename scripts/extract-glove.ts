/**
 * Keeps the GloVe vectors of the words the "Embedding space" lesson shows and writes them to
 * src/lib/lessons/llm/embeddings/vectors.ts.
 * Run from the project root:
 *   node --experimental-strip-types scripts/extract-glove.ts <path to glove-wiki-gigaword-100.gz>
 * The input is the gzip-compressed text file: a first line "400000 100", then one line per word (the word, then
 * 100 numbers separated by spaces). It is streamed, never loaded whole. The numbers are copied as published, with
 * no rounding, so running the script again gives the same file.
 * The word list comes from the lesson's words.ts.
 */
import { createReadStream, writeFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { createGunzip } from 'node:zlib';
import { WORD_LIST } from '../src/lib/lessons/llm/embeddings/words.ts';

const DIM = 100;
const OUT = new URL('../src/lib/lessons/llm/embeddings/vectors.ts', import.meta.url);

const input = process.argv[2];
if (!input) {
	console.error('Usage: node --experimental-strip-types scripts/extract-glove.ts <path to glove-wiki-gigaword-100.gz>');
	process.exit(1);
}

const wanted = new Set(WORD_LIST);
const found = new Map<string, string[]>();
const lines = createInterface({ input: createReadStream(input).pipe(createGunzip()), crlfDelay: Infinity });
let first = true;
let rows = 0;
for await (const line of lines) {
	if (first) {
		first = false;
		if (line.trim() !== `400000 ${DIM}`) {
			console.error(`Unexpected header "${line.trim()}": expected "400000 ${DIM}". Is this glove-wiki-gigaword-100?`);
			process.exit(1);
		}
		continue;
	}
	rows++;
	const space = line.indexOf(' ');
	const word = line.slice(0, space);
	if (!wanted.has(word) || found.has(word)) continue;
	const nums = line.slice(space + 1).trim().split(' ');
	if (nums.length !== DIM || nums.some((x) => !Number.isFinite(Number(x)))) {
		console.error(`The line for "${word}" does not hold ${DIM} numbers.`);
		process.exit(1);
	}
	found.set(word, nums);
}

const missing = WORD_LIST.filter((w) => !found.has(w));
if (missing.length > 0) {
	console.error(`Not in the file: ${missing.join(', ')}. Nothing was written.`);
	process.exit(1);
}

const body = WORD_LIST.map((w) => `\t${w}: [${found.get(w)!.join(', ')}]`).join(',\n');
const text = `// Written by scripts/extract-glove.ts. Do not edit by hand.
// GloVe vectors (Pennington, Socher and Manning, 2014), "glove-wiki-gigaword-100": trained on Wikipedia 2014
// and Gigaword 5, 100 numbers per word. Released under the Public Domain Dedication and License.
// Only the words this lesson shows are kept, with the published numbers unchanged.
export const GLOVE_DIM = ${DIM};
export const GLOVE: Record<string, number[]> = {
${body}
};
`;
writeFileSync(OUT, text);
console.log(`Read ${rows} words, kept ${WORD_LIST.length}, wrote ${OUT.pathname}`);

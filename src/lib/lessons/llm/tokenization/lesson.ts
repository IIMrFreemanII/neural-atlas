import type { LessonDef } from '../../types.ts';
import { DEFAULT_TEXT, MAX_MERGES, TRAIN_MAX } from './data.ts';
import { questions } from './questions.ts';
import { createTokenizationScene } from './scene.ts';

const lesson: LessonDef = {
	id: 'tokenization',
	track: 'llm',
	title: 'Tokenization',
	summary: 'See how text is cut into tokens, and watch byte pair encoding learn its pieces one merge at a time.',
	defaults: { view: 0, mode: 2, merges: MAX_MERGES, tmerges: 0, ids: 1, chart: 0, scenario: 1 },
	discrete: ['view', 'tmerges', 'ids', 'chart', 'scenario'],
	controls: [
		{ kind: 'text', id: 'text', label: 'Your text', initial: DEFAULT_TEXT, maxLength: 48 },
		{
			kind: 'select',
			key: 'mode',
			label: 'Split by',
			options: [
				{ value: 0, label: 'Characters' },
				{ value: 1, label: 'Words' },
				{ value: 2, label: 'Subwords' }
			]
		},
		{ kind: 'slider', key: 'merges', label: 'Merges learned', min: 0, max: MAX_MERGES, step: 1, tone: 'param', digits: 0 },
		{ kind: 'action', id: 'merge', label: 'Merge once', icon: 'step' },
		{ kind: 'action', id: 'run', label: 'Keep merging', icon: 'play' },
		{ kind: 'action', id: 'reset', label: 'Start over', icon: 'reset' },
		{ kind: 'slider', key: 'tmerges', label: 'Merges', min: 0, max: TRAIN_MAX, step: 1, tone: 'param', digits: 0 }
	],
	readouts: [
		{ key: 'tokens', label: 'Tokens', tone: 'data', digits: 0 },
		{ key: 'vocab', label: 'Vocabulary size', tone: 'param', digits: 0 }
	],
	legend: [
		{ tone: 'data', label: 'Single characters' },
		{ tone: 'param', label: 'Learned pieces' },
		{ tone: 'grad', label: 'Most common pair' },
		{ tone: 'neg', label: 'Unknown word' }
	],
	steps: [
		{
			title: 'Models read numbers, not letters',
			body: 'A language model cannot read letters. First the text is cut into pieces called tokens, and each token gets an ID number. The model only ever sees those IDs.\n\nType in the box and watch the pieces change. This tokenizer learned from a few paragraphs, so its pieces differ from a real model\'s.',
			params: { view: 0, mode: 2, ids: 1 },
			controls: ['text']
		},
		{
			title: 'One token per character',
			body: 'The simplest cut makes every character a token. The vocabulary is tiny, only the letters and signs that appear.\n\nBut the sequences get long, and a single letter carries almost no meaning on its own. Type a longer sentence and count the [[data:single characters]].',
			params: { view: 0, mode: 0, ids: 1 },
			controls: ['text']
		},
		{
			title: 'One token per word',
			body: 'Now every word is one token. Sequences are short, but the vocabulary has to hold every word there is.\n\nTry a word this tokenizer has not seen, like your own name. An [[neg:unknown word, shown in rose]] cannot be represented at all.',
			params: { view: 0, mode: 1, ids: 1 },
			controls: ['text']
		},
		{
			title: 'Subwords: start from characters and merge',
			body: 'Here is a better idea. Start with single characters, count every adjacent pair, and fuse the [[grad:most common pair]] into a new token. Then repeat.\n\nThis is a tiny text. Press **Merge once** and watch the amber pair close up.',
			params: { view: 1, tmerges: 0, scenario: 4 },
			controls: ['merge', 'reset'],
			deepDive: {
				note: 'Byte pair encoding in a few lines. Each pass picks the most frequent adjacent pair and replaces it everywhere with one new token.',
				code: {
					lang: 'ts',
					source: `let words = splitWords(text).map((w) => [...w]);
const merges = [];
while (merges.length < maxMerges) {
	const counts = countPairs(words);
	if (counts.length === 0 || counts[0].count < 2) break;
	merges.push(counts[0].pair);
	words = applyMerge(words, counts[0].pair);
}`
				}
			}
		},
		{
			title: 'Keep merging',
			body: 'Common words soon become a single token. Every merge adds one piece to the vocabulary, and the token count falls.\n\nPress **Keep merging**, or drag **Merges** to move back and forth. The [[param:learned pieces]] collect on the right.',
			params: { view: 1, tmerges: 3, scenario: 5 },
			controls: ['run', 'merge', 'reset', 'tmerges']
		},
		{
			title: 'The same merges split new text',
			body: 'The merges learned from the paragraphs now cut any text. Drag **Merges learned** and watch your text go from letters to whole words.\n\nRare words stay in several pieces, but they are never unknown, because the single characters are always there.',
			params: { view: 0, mode: 2, chart: 1, merges: 40 },
			controls: ['text', 'merges'],
			deepDive: {
				note: 'More merges mean a bigger vocabulary but shorter sequences. Fewer merges mean a small vocabulary but long sequences. GPT-2 small has a vocabulary of 50,257 tokens, and current large models use from tens of thousands to about 200,000. In English a token is about three quarters of a word on average.'
			}
		},
		{
			title: 'What the model receives',
			body: 'This is all the model gets: a list of ID numbers. The letters are gone.\n\nSwitch **Split by** to see how the same text becomes a different list. In the next lesson each ID turns into a vector.',
			params: { view: 0, mode: 2, merges: MAX_MERGES, ids: 1 },
			controls: ['text', 'mode']
		}
	],
	questions,
	createScene: createTokenizationScene
};

export default lesson;

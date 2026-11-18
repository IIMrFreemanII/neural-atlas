import { encode } from '../../../ml/bpe.ts';
import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';
import { DEFAULT_TEXT, MODEL } from './data.ts';

/** Single characters in the vocabulary: what the vocabulary size is with no merges learned. */
const BASE_CHARS = MODEL.vocab.length - MODEL.merges.length;

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'The text is split into subwords, with no merges learned yet, so every character is its own token. You drag **Merges learned** up to 100. What happens to the number of tokens and to the vocabulary size?\n\nDecide first, then check with the slider.',
		params: { view: 0, mode: 2, merges: 0, ids: 1, chart: 0 },
		controls: ['merges'],
		legend: ['Single characters', 'Learned pieces'],
		options: [
			{
				text: 'Both numbers fall, because merging joins pieces together.',
				why: 'Merging does shorten the text, but every merge also adds a new piece to the vocabulary. The vocabulary only grows.'
			},
			{
				text: 'The text gets fewer tokens, and the vocabulary gets bigger.',
				correct: true,
				why: 'A merge fuses a common pair into one new token, so each use of the pair costs one token instead of two, and the new token joins the vocabulary. Here the text drops from 35 tokens to 13, and the vocabulary grows from 43 to 143.'
			},
			{
				text: 'The text gets fewer tokens, and the vocabulary stays the same.',
				why: 'The single characters stay, but each merge adds one more piece. After 100 merges the vocabulary is 43 characters plus 100 learned pieces, 143 in all.'
			},
			{
				text: 'The text gets more tokens, because the new pieces are longer.',
				why: 'A longer piece covers more characters, so the same text needs fewer of them. Only splitting by characters gives the longest sequence.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'Split the text into subwords. Using only **Merges learned**, cut it to **20 tokens or fewer** while keeping the [[param:vocabulary size]] **under 80**.',
		params: { view: 0, mode: 2, merges: 0, ids: 1, chart: 0 },
		controls: ['merges'],
		legend: ['Single characters', 'Learned pieces'],
		goal: ({ params: p }) =>
			p.mode === 2 &&
			encode(DEFAULT_TEXT, MODEL, Math.round(p.merges)).tokens.length <= 20 &&
			BASE_CHARS + Math.round(p.merges) < 80,
		hint: 'Every merge shortens the text and adds one piece to the vocabulary. Watch both numbers at once, and look for the range where both are good enough.',
		solved:
			'The vocabulary is 43 single characters plus one piece per merge, so it stays under 80 up to 36 merges. The text first reaches 20 tokens at 27 merges. More merges would shorten the text further, but they make the vocabulary bigger.',
		example: { params: { merges: 30 } }
	},
	{
		kind: 'choice',
		prompt:
			'In **Words** mode the word Tokenizers is shown in rose as unknown, because the tokenizer never saw it. Why can **Subwords** mode cut this word, and any other new word, without an unknown piece?',
		params: { view: 0, mode: 1, merges: 300, ids: 1, chart: 0 },
		controls: ['text', 'mode'],
		legend: ['Single characters', 'Learned pieces', 'Unknown word'],
		options: [
			{
				text: 'Its vocabulary holds every word, so no word is ever new.',
				why: 'The vocabulary is a fixed list, 343 pieces at most in this lesson, and Tokenizers is not a single piece in it. That is why Words mode fails on it.'
			},
			{
				text: 'It adds each new word to the vocabulary as soon as you type it.',
				why: 'The vocabulary is fixed by the merges learned. Type anything you like and the vocabulary size does not change.'
			},
			{
				text: 'It cuts a new word into smaller pieces, and every single letter has a token.',
				correct: true,
				why: 'A new word breaks into learned pieces, and into single letters where no piece fits. Tokenizers becomes T, oken, iz and ers. Every letter of the alphabet is a token, so the cutting always ends.'
			},
			{
				text: 'It finds the known word that means the closest thing.',
				why: 'A tokenizer does not know what words mean. It only cuts the text into pieces from its list, by the merges it learned.'
			}
		]
	}
];

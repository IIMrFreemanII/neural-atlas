import { trainBpe } from '../../../ml/bpe.ts';

/** The tiny text the live merge trainer works on. */
export const TRAIN_TEXT = 'the cat sat on the mat the fat cat ate the hat then the cat sat on the hat';
export const TRAIN_MODEL = trainBpe(TRAIN_TEXT, 16);
export const TRAIN_MAX = TRAIN_MODEL.merges.length;

/** A few original paragraphs of plain English. The tokenizer in the lesson learns its merges from this. */
export const CORPUS = [
	'Every story has a believer and a doubter. The believer says, I believe you, and the doubter says, I cannot believe it. We believe what we see, and we believed what we heard. Believing is easy for a believer, and believing is hard for a doubter. When the believers believed the story, the doubter stayed unbelieving. Disbelief is a quiet feeling, and the doubters let the disbelief show. We tried to disbelieve the rumor, but the rumor was believable, and a believable rumor is hard to disbelieve.',
	'A believer believes, a doubter doubts, and a believer who believes is slow to judge. Believers and doubters both believe in something. I believe it, you believe it, and they believed it. Believe me, a believable story is a good story. A map should be readable. A readable map is clear and a readable page is calm. A page that is unreadable makes the reader tired, and an unreadable map makes the traveler lost.',
	'The old atlas was readable, the new atlas was unreadable, and the clerk made a readable copy by hand. A readable sign is a kind sign. A sign that is unreadable is a sign that is waiting for paint. A glass is breakable, and a plate is breakable too. A steel pot is not breakable. We bought a strong cup for the picnic, and the cup survived the hill. A breakable thing needs care, and a strong thing needs only a wash.',
	'The chair was comfortable, the bed was comfortable, and the old sofa was the most comfortable seat in the house. A comfortable day is a quiet day. Some chairs are reliable, some trains are reliable, and a reliable friend is worth the wait. Organizers plan, and realizers do. Two organizers planned the festival, and two realizers built it. Organizers, organizers, and realizers, realizers. The organizers booked the hall, and the organizers printed the signs.',
	'The festival organizers like to organize things, and they organize the stalls by color. When the stalls were organized, the realizers began to realize the plan. Two realizers can realize a plan faster than one, and the realizers were happy to realize that the plan was working. It is easy to realize a small plan, and it is hard to organize a large one. They tried to recognize every face, and they tried to memorize every name. A token is a small piece of text. A token can be a letter, a word, or a part of a word.',
	'Tokens are what a machine reads. The machine does not read letters, it reads tokens. Each token has a number, and the tokens in a sentence become a list of numbers. When we tokenize a sentence, we cut the sentence into tokens. We tokenized the sentence, and the tokens were short. The more we tokenize, the more tokens we have. A tokenizing step is the first step.',
	'Tokens are cheap to store, and tokens are easy to count. A token that is common is a short token, and a token that is rare is a long token in many pieces. Some words begin with un. We were unhappy, and the unhappy crowd left early. An unhappy child is unable to sleep, and an unable worker is unhappy too. An unread letter is an unread letter until someone opens it. An unreadable note is unreadable, and an unreadable note is hard to answer.',
	'Tokens come first. Token by token, the machine reads the sentence. Tokens are numbers, and Token lists are long. Tokens are small, and Tokens are everywhere. Token counts matter, and token lists are long. Each token has a number. Tokens are made of letters, and a token can hold a whole word.',
	'Tokens that are rare are cut into smaller tokens. The tokens of a sentence are a list of numbers, and Token numbers are all the machine sees. Tokens, tokens, and more tokens. Look at the endings. A word can end in able, as in readable, breakable, comfortable, reliable, and believable. A word can end in ing, as in walking, talking, believing, and organizing. A word can end in ed, as in walked, talked, believed, and organized.',
	'A word can end in ers, as in organizers, realizers, believers, and doubters. A word can end in ize, as in organize, realize, recognize, and memorize. Able, able, and able. The able reader is able to read the readable page. The rain fell on the town. The rain was cold, but the kitchen was warm.',
	'My sister was making soup, and the soup was warm. When we arrived, the town was quiet and the streets were quiet. We walked and talked, we jumped and played, and we were happy. Learning takes time, and learning feels slow at first. A student who is learning will make mistakes, and a student who is learning will believe in small steps. An unhappy start does not decide the end.',
	'I believe you are able to learn, and I believe you are able to read. A machine reads numbers, not letters. So a sentence has to be turned into numbers before a machine can read the sentence. One way to do that is to split the text into pieces. A piece can be a letter, a word, or something in between. Whole words are quick to read, but a machine that has never seen a word cannot read the word.',
	'Common words can be one piece, and rare words are built from pieces that appear in many words. We split the text into pieces, and we split the pieces again. The words are easy to split, and the words are easy to read. The pieces of walked, talked, jumped, and played share the same ending. To disbelieve is not to doubt. We disbelieve, we disbelieved, and we are disbelieving. Disbelief grows, and disbelief fades.',
	'The disbelieving crowd shared its disbelief. A disbelieving look is a look of disbelief. Organizers, organizers, and more organizers. The organizers, the realizers, and the believers. Tokens are small, tokens are neat, and tokens are fast. The unbelieving crowd stood still, and the unbelieving guard laughed. An unbelieving child is a quiet child.',
	'Unbelieving eyes, unbelieving ears, and unbelieving hearts. The path was broken, and the glass was broken. The spoken word is a spoken word, and the spoken line was clear. Nobody had woken, and the woken child was broken hearted. A broken toy, a spoken thanks, and a woken town. Nonbelievers are not believers.'
].join(' ');

export const MAX_MERGES = 300;
export const MODEL = trainBpe(CORPUS, MAX_MERGES);

/** Distinct words of the corpus, letters only, lowercased. */
export const WORDS: Set<string> = new Set(CORPUS.toLowerCase().match(/[a-z]+/g) ?? []);
/** Sorted words, for the id of a word in the "one token per word" view. */
export const WORD_LIST: string[] = [...WORDS].sort();

export const DEFAULT_TEXT = 'Tokenizers split unbelievable words';

/** Letters of a word, lowercased, punctuation and spaces stripped. */
export function wordKey(word: string): string {
	return word.toLowerCase().replace(/[^a-z]/g, '');
}

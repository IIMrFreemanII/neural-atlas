/** Words by category. Words joined by '/' are related and only kept next to each other in the list. */
export const SPEC: string[][] = [
	[
		'man/woman/boy/girl',
		'king/queen/prince/princess',
		'father/mother',
		'uncle/aunt',
		'brother/sister',
		'actor/actress'
	],
	['cat/kitten', 'dog/puppy', 'lion', 'tiger', 'horse', 'cow', 'wolf', 'mouse', 'bird', 'fish'],
	['apple', 'bread', 'cheese', 'rice', 'pizza', 'soup', 'cake', 'banana', 'coffee', 'tea'],
	['france/paris', 'italy/rome', 'japan/tokyo', 'spain/madrid', 'egypt/cairo', 'germany/berlin'],
	['walk/walked', 'run/ran', 'jump/jumped', 'eat/ate', 'sing/sang', 'swim/swam'],
	['big/bigger', 'small/smaller', 'fast/faster', 'slow/slower', 'cold/colder', 'hot/hotter']
];

/** All 74 words, in list order. scripts/extract-glove.ts reads this to know which vectors to keep,
 * which is why this file must not import vectors.ts. */
export const WORD_LIST: string[] = SPEC.flatMap((groups) => groups.flatMap((g) => g.split('/')));

import { buildNgram, tokenizeWords } from '../../../ml/ngram.ts';

/** A tiny original text in a small world. The toy language model only counts which words follow which. */
export const CORPUS = `
the cat sat on the mat . the cat sat on the mat . the cat sat on the mat . the cat sat on the mat . the cat sat on the sofa .
the cat sat on the floor . the cat sat on the bed . the dog sat on the mat . the dog sat on the mat .
the girl sat on the mat . the girl sat on the sofa . the girl sat on the floor . the girl sat on the step .
the cat slept on the mat . the cat slept on the sofa . the cat slept on the bed . the dog slept on the floor .
the dog slept on the mat . the girl slept on the mat . the cat slept on the mat . the girl slept on the sofa . the bird sat on the roof . the bird sat on the wall .
the girl looked at the cat on the chair . the dog sat on the chair . the cat sat on the wall .
in the morning the sun came up . in the morning the sun was warm . in the morning the sun came up over the garden .
in the morning the cat ate its fish . in the morning the cat went to the kitchen . in the morning the dog ran in the garden .
in the morning the dog wanted to run . in the morning the girl drank her milk . in the morning the girl went to the garden .
in the morning the rain stopped . in the morning the garden was wet . in the morning the kitchen was quiet .
the dog wanted to run . the dog wanted to run . the dog wanted to run . the dog wanted to run . the dog wanted to run . the dog wanted to run .
the dog wanted to run in the garden . the dog wanted to play with the ball . the dog wanted to eat .
the dog wanted to run after the cat . the dog wanted to run in the rain .
the cat wanted to eat . the cat wanted to sleep . the cat wanted to watch the bird . the girl wanted to play in the garden .
the girl wanted to run . the girl wanted to pick a red flower .
the sun was warm . the sun was bright . the sun went down behind the old tree . the rain was cold .
the rain came down on the garden . the rain came down on the roof . the rain stopped and the sun came up .
the cat ran in the garden . the dog ran after the cat . the cat ran to the kitchen . the girl ran to the door .
the girl ate her bread in the kitchen . the cat ate its fish in the kitchen . the dog found a bone in the garden .
the cat looked at the dog . the dog looked at the cat . the girl looked at the sky . the girl looked at the moon .
the cat jumped on the table . the cat jumped over the puddle . the dog jumped over the little wall .
the girl opened the window and the cat heard a bird . the dog smelled the sweet flower in the tall grass .
a yellow flower grew near the old wall . a soft blanket lay on the bed . a little star shone in the sky .
is the cat in the kitchen ? the cat is in the kitchen . where is the dog ? the dog is in the garden .
the girl is happy . the cat is happy . the dog is happy . the girl is small . the dog is big .
at night the cat slept on the mat . at night the dog slept on the floor . at night the girl slept in her bed .
at night the rain came down . at night the garden was quiet . at night a cloud hid the moon .
`;

export const MODEL = buildNgram(tokenizeWords(CORPUS));

export const PROMPTS = ['the cat sat on the', 'in the morning the', 'the dog wanted to'];

/** Token ids of a prompt. */
export function promptIds(index: number): number[] {
	const i = Math.min(PROMPTS.length - 1, Math.max(0, Math.round(index)));
	return tokenizeWords(PROMPTS[i]).map((t) => MODEL.ids.get(t)!);
}

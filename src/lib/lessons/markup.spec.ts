import { describe, expect, it } from 'vitest';
import { parseBody } from './markup.ts';

describe('parseBody', () => {
	it('parses plain text', () => {
		expect(parseBody('hello')).toEqual([[{ kind: 'text', text: 'hello' }]]);
	});

	it('parses bold', () => {
		expect(parseBody('a **b** c')).toEqual([
			[
				{ kind: 'text', text: 'a ' },
				{ kind: 'bold', text: 'b' },
				{ kind: 'text', text: ' c' }
			]
		]);
	});

	it('parses terms with a tone', () => {
		expect(parseBody('the [[param:weight]] here')).toEqual([
			[
				{ kind: 'text', text: 'the ' },
				{ kind: 'term', tone: 'param', text: 'weight' },
				{ kind: 'text', text: ' here' }
			]
		]);
	});

	it('leaves unknown tones as plain text', () => {
		expect(parseBody('[[purple:x]]')).toEqual([[{ kind: 'text', text: '[[purple:x]]' }]]);
	});

	it('splits paragraphs on blank lines', () => {
		const r = parseBody('one\n\ntwo\n  \nthree');
		expect(r).toHaveLength(3);
		expect(r[1]).toEqual([{ kind: 'text', text: 'two' }]);
	});

	it('leaves unterminated marks as plain text', () => {
		expect(parseBody('a **b')).toEqual([[{ kind: 'text', text: 'a **b' }]]);
		expect(parseBody('a [[data:b')).toEqual([[{ kind: 'text', text: 'a [[data:b' }]]);
		expect(parseBody('[[data]]')).toEqual([[{ kind: 'text', text: '[[data]]' }]]);
	});
});

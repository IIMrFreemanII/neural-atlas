import { describe, expect, it } from 'vitest';
import { nextLogits, paramCount } from '../../../ml/gpt.ts';
import { encode } from '../../../ml/wordtok.ts';
import { lmFromBuffer } from './lm.ts';
import { LM_CONFIG, LM_VOCAB } from './lm-meta.ts';

// The project has no Node type definitions, so node:fs is loaded through a variable and typed by hand.
const fsName = 'node:fs';
const fs = (await import(/* @vite-ignore */ fsName)) as { readFileSync(url: URL): Uint8Array };

function readWeights(): ArrayBuffer {
	const bytes = fs.readFileSync(new URL('./tinystories.bin', import.meta.url));
	return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

describe('language model files', () => {
	it('has the byte length the config says', () => {
		expect(readWeights().byteLength).toBe(paramCount(LM_CONFIG) * 4);
	});
	it('has a vocabulary that matches the config', () => {
		expect(LM_VOCAB.length).toBe(LM_CONFIG.vocab);
		expect(LM_VOCAB[0]).toBe('<unk>');
		expect(LM_VOCAB[1]).toBe('<end>');
		expect(new Set(LM_VOCAB).size).toBe(LM_VOCAB.length);
	});
	it('gives finite logits for a prompt', () => {
		const lm = lmFromBuffer(readWeights());
		const logits = nextLogits(lm.model, encode(lm.vocab, 'once upon a time'));
		expect(logits.length).toBe(LM_CONFIG.vocab);
		for (const x of logits) expect(Number.isFinite(x)).toBe(true);
	});
	it('rejects a buffer of the wrong length', () => {
		expect(() => lmFromBuffer(new ArrayBuffer(12))).toThrow();
	});
});

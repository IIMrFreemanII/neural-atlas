import weightsUrl from './tinystories.bin?url';
import { paramCount, type Gpt } from '../../../ml/gpt.ts';
import { vocabFromWords, type Vocab } from '../../../ml/wordtok.ts';
import { LM_CONFIG, LM_VOCAB } from './lm-meta.ts';

export interface Lm {
	model: Gpt;
	vocab: Vocab;
}

/** Builds the model from the bytes of tinystories.bin. Throws if the length is not paramCount * 4. */
export function lmFromBuffer(buffer: ArrayBuffer): Lm {
	const expected = paramCount(LM_CONFIG) * 4;
	if (buffer.byteLength !== expected) {
		throw new Error(`tinystories.bin has ${buffer.byteLength} bytes, expected ${expected}`);
	}
	// The file is little-endian float32; read it explicitly so the result does not depend on the machine.
	const view = new DataView(buffer);
	const params = new Float32Array(expected / 4);
	for (let i = 0; i < params.length; i++) params[i] = view.getFloat32(i * 4, true);
	return { model: { config: LM_CONFIG, params }, vocab: vocabFromWords(LM_VOCAB) };
}

let loading: Promise<Lm> | null = null;

/** Fetches the weights once (later calls return the same promise) and builds the model. Rejects with a clear Error if the fetch fails. */
export function loadLm(): Promise<Lm> {
	if (loading === null) {
		loading = (async () => {
			let response: Response;
			try {
				response = await fetch(weightsUrl);
			} catch (e) {
				throw new Error(`Could not load the language model weights: ${e instanceof Error ? e.message : String(e)}`);
			}
			if (!response.ok) throw new Error(`Could not load the language model weights: HTTP ${response.status}`);
			return lmFromBuffer(await response.arrayBuffer());
		})();
		// Let a later call try again after a failure.
		loading.catch(() => {
			loading = null;
		});
	}
	return loading;
}

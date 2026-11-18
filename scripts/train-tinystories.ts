/**
 * Trains the small GPT of the language-model lessons on the TinyStories text and writes its weights.
 * Run from the project root:
 *   node --experimental-strip-types scripts/train-tinystories.ts <path to TinyStoriesV2-GPT4-valid.txt> [--steps N] [--out-dir DIR] [--checkpoint-dir DIR]
 * Fixed seeds and a fixed worker count: running it again on the same machine reproduces the same files.
 * Writes tinystories.bin (float32 parameters) and lm-meta.ts (config, vocabulary, training facts).
 * With --checkpoint-dir it also saves the weights at a few steps along the way (step-000000.bin before any
 * training, then step 25, 50, 100, ... doubling, and the last step) and curve.json with the loss of every
 * step. scripts/build-training-log.ts turns those into the data of the "Watching a model learn" lesson.
 * Saving checkpoints does not change the training run or its result.
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads';
import { createWorkspace, loss, lossAndGrad, nextLogits, paramCount, initGpt, type GptConfig } from '../src/lib/ml/gpt.ts';
import { mulberry32 } from '../src/lib/ml/rng.ts';
import { buildVocab, decode, encode, END, STORY_SEPARATOR, tokenizeWords, UNK } from '../src/lib/ml/wordtok.ts';

const CONFIG: GptConfig = { vocab: 2048, d: 64, layers: 4, heads: 4, ctx: 32 };
const INIT_SEED = 7;
const DATA_SEED = 1001;
const WORKERS = 8; // fixed, so the result does not depend on the machine's core count
const BATCH = 64; // sequences per step, each ctx + 1 tokens
const STEPS = 5000; // --steps overrides
const WARMUP = 200; // steps of linear warm-up from 0
const LR = 0.003; // peak; cosine decay to LR_END at the last step
const LR_END = 0.0003;
const BETA1 = 0.9;
const BETA2 = 0.999;
const EPS = 1e-8;
const CLIP = 1; // clip the global gradient norm to this
const VALID_SHARE = 0.02; // the last 2% of stories are held out
const EVAL_EVERY = 500;
const EVAL_SEQUENCES = 400; // fixed validation windows, chosen once with seed DATA_SEED + 1
const PROMPT = 'once upon a time';
const SAMPLE_TOKENS = 40;
const CHECKPOINT_FIRST = 25; // first checkpoint after step 0; later ones double

interface WorkerInit {
	config: GptConfig;
	params: SharedArrayBuffer;
	stream: SharedArrayBuffer;
	grads: SharedArrayBuffer;
}
interface WorkJob {
	starts: number[];
}
interface WorkResult {
	loss: number;
}

// ------------------------------------------------------------------------------------------ worker thread

function runWorker(): void {
	const init = workerData as WorkerInit;
	const model = { config: init.config, params: new Float32Array(init.params) };
	const stream = new Uint16Array(init.stream);
	const grads = new Float64Array(init.grads);
	const ws = createWorkspace(init.config);
	const window = new Int32Array(init.config.ctx + 1);
	const port = parentPort;
	if (port === null) throw new Error('worker started without a parent');
	port.on('message', (job: WorkJob) => {
		grads.fill(0);
		let sum = 0;
		for (const start of job.starts) {
			for (let i = 0; i <= init.config.ctx; i++) window[i] = stream[start + i];
			sum += lossAndGrad(model, ws, window, init.config.ctx, grads);
		}
		const result: WorkResult = { loss: sum };
		port.postMessage(result);
	});
}

// ------------------------------------------------------------------------------------------ main thread

function dirUrl(dir: string): URL {
	return new URL(`file://${dir.startsWith('/') ? dir : process.cwd() + '/' + dir}/`);
}

function parseArgs(argv: string[]): { input: string; steps: number; outDir: URL; checkpointDir: URL | null } {
	let input = '';
	let steps = STEPS;
	let outDir = new URL('../src/lib/lessons/llm/shared/', import.meta.url);
	let checkpointDir: URL | null = null;
	for (let i = 0; i < argv.length; i++) {
		if (argv[i] === '--steps') steps = Number(argv[++i]);
		else if (argv[i] === '--out-dir') outDir = dirUrl(argv[++i]);
		else if (argv[i] === '--checkpoint-dir') checkpointDir = dirUrl(argv[++i]);
		else input = argv[i];
	}
	if (input === '' || !Number.isInteger(steps) || steps < 1) {
		console.error(
			'usage: node --experimental-strip-types scripts/train-tinystories.ts <TinyStories text file> [--steps N] [--out-dir DIR] [--checkpoint-dir DIR]'
		);
		process.exit(1);
	}
	return { input, steps, outDir, checkpointDir };
}

/** Steps at which a checkpoint is saved: 0, then 25 doubling while below the last step, then the last step. */
function checkpointSteps(steps: number): number[] {
	const list = [0];
	for (let s = CHECKPOINT_FIRST; s < steps; s *= 2) list.push(s);
	list.push(steps);
	return list;
}

function quote(s: string): string {
	return `'${s.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
}

function round(x: number, digits: number): number {
	const f = 10 ** digits;
	return Math.round(x * f) / f;
}

function callWorker(worker: Worker, job: WorkJob): Promise<WorkResult> {
	return new Promise((resolve, reject) => {
		const onMessage = (r: WorkResult): void => {
			worker.off('error', onError);
			resolve(r);
		};
		const onError = (e: Error): void => {
			worker.off('message', onMessage);
			reject(e);
		};
		worker.once('message', onMessage);
		worker.once('error', onError);
		worker.postMessage(job);
	});
}

async function main(): Promise<void> {
	const { input, steps, outDir, checkpointDir } = parseArgs(process.argv.slice(2));
	const t0 = Date.now();

	// Text -> tokens.
	const raw = readFileSync(input, 'utf8');
	const stories = raw
		.split(STORY_SEPARATOR)
		.map((s) => tokenizeWords(s))
		.filter((s) => s.length > 0);
	const vocab = buildVocab(stories, CONFIG.vocab);
	const encoded = stories.map((s) => {
		const ids = s.map((w) => vocab.ids.get(w) ?? UNK);
		ids.push(END);
		return ids;
	});
	const validStories = Math.max(1, Math.round(encoded.length * VALID_SHARE));
	const trainParts = encoded.slice(0, encoded.length - validStories);
	const validParts = encoded.slice(encoded.length - validStories);
	const count = (parts: number[][]): number => parts.reduce((n, s) => n + s.length, 0);
	const trainTokens = count(trainParts);
	const validTokens = count(validParts);
	const trainStream = new Uint16Array(new SharedArrayBuffer(trainTokens * 2));
	const validStream = new Uint16Array(validTokens);
	let at = 0;
	for (const s of trainParts) for (const id of s) trainStream[at++] = id;
	at = 0;
	for (const s of validParts) for (const id of s) validStream[at++] = id;
	let unk = 0;
	for (let i = 0; i < trainStream.length; i++) if (trainStream[i] === UNK) unk++;
	for (let i = 0; i < validStream.length; i++) if (validStream[i] === UNK) unk++;
	const unkShare = unk / (trainTokens + validTokens);
	const nParams = paramCount(CONFIG);
	console.log(`stories ${stories.length} (train ${trainParts.length}, held out ${validParts.length})`);
	console.log(`tokens: train ${trainTokens}, valid ${validTokens}; <unk> share ${(unkShare * 100).toFixed(2)}%`);
	console.log(`parameters ${nParams}`);

	// Model and workers.
	const paramBuffer = new SharedArrayBuffer(nParams * 4);
	const model = initGpt(CONFIG, INIT_SEED);
	const params = new Float32Array(paramBuffer);
	params.set(model.params);
	const shared = { config: CONFIG, params };
	const gradArrays: Float64Array[] = [];
	const workers: Worker[] = [];
	for (let w = 0; w < WORKERS; w++) {
		const grads = new Float64Array(new SharedArrayBuffer(nParams * 8));
		gradArrays.push(grads);
		const init: WorkerInit = { config: CONFIG, params: paramBuffer, stream: trainStream.buffer as SharedArrayBuffer, grads: grads.buffer as SharedArrayBuffer };
		workers.push(new Worker(new URL(import.meta.url), { workerData: init }));
	}

	// Fixed validation windows.
	const evalRand = mulberry32(DATA_SEED + 1);
	const evalStarts: number[] = [];
	for (let i = 0; i < EVAL_SEQUENCES; i++) evalStarts.push(Math.floor(evalRand() * (validStream.length - CONFIG.ctx)));
	const evalWs = createWorkspace(CONFIG);
	const evalWindow = new Int32Array(CONFIG.ctx + 1);
	const validLoss = (): number => {
		let sum = 0;
		for (const s of evalStarts) {
			for (let i = 0; i <= CONFIG.ctx; i++) evalWindow[i] = validStream[s + i];
			sum += loss(shared, evalWs, evalWindow, CONFIG.ctx);
		}
		return sum / evalStarts.length;
	};

	const promptIds = encode(vocab, PROMPT);
	const sample = (): string => {
		const ids = promptIds.slice();
		for (let i = 0; i < SAMPLE_TOKENS; i++) {
			const lg = nextLogits(shared, ids);
			let best = -1;
			for (let t = 0; t < lg.length; t++) if (t !== UNK && (best < 0 || lg[t] > lg[best])) best = t;
			ids.push(best);
		}
		return decode(vocab, ids);
	};

	const paramBytes = (): Uint8Array => {
		const bytes = new Uint8Array(nParams * 4);
		const view = new DataView(bytes.buffer);
		for (let i = 0; i < nParams; i++) view.setFloat32(i * 4, params[i], true);
		return bytes;
	};

	// Checkpoints (only with --checkpoint-dir).
	const checkpointAt = new Set(checkpointDir ? checkpointSteps(steps) : []);
	const checkpoints: { step: number; validLoss: number }[] = [];
	const stepLosses: number[] = [];
	const writeCheckpoint = (step: number, vLoss: number): void => {
		if (checkpointDir === null) return;
		mkdirSync(checkpointDir, { recursive: true });
		writeFileSync(new URL(`step-${String(step).padStart(6, '0')}.bin`, checkpointDir), paramBytes());
		checkpoints.push({ step, validLoss: round(vLoss, 4) });
		const curve = {
			config: CONFIG,
			steps,
			batch: BATCH,
			evalSequences: EVAL_SEQUENCES,
			// trainLoss[i] is the mean loss of the batch of step i + 1, measured before that step's update.
			trainLoss: stepLosses,
			checkpoints
		};
		writeFileSync(new URL('curve.json', checkpointDir), JSON.stringify(curve));
	};

	const writeFiles = (step: number, vLoss: number): void => {
		mkdirSync(outDir, { recursive: true });
		writeFileSync(new URL('tinystories.bin', outDir), paramBytes());
		const lines: string[] = [];
		for (let i = 0; i < vocab.words.length; i += 16) lines.push('\t' + vocab.words.slice(i, i + 16).map(quote).join(', '));
		const meta = `// Written by scripts/train-tinystories.ts. Do not edit by hand.
import type { GptConfig } from '../../../ml/gpt.ts';

export const LM_CONFIG: GptConfig = { vocab: ${CONFIG.vocab}, d: ${CONFIG.d}, layers: ${CONFIG.layers}, heads: ${CONFIG.heads}, ctx: ${CONFIG.ctx} };
/** Token texts by id. 0 is '<unk>', 1 is '<end>'. */
export const LM_VOCAB: string[] = [
${lines.join(',\n')}
];
export const LM_TRAINING = {
	source: 'TinyStoriesV2-GPT4-valid.txt (roneneldan/TinyStories)',
	stories: ${stories.length},
	trainTokens: ${trainTokens},
	validTokens: ${validTokens},
	unkShare: ${round(unkShare, 4)},
	steps: ${step},
	batch: ${BATCH},
	tokensSeen: ${step * BATCH * CONFIG.ctx},
	learningRate: [${LR}, ${LR_END}],
	validLoss: ${round(vLoss, 4)},
	validPerplexity: ${round(Math.exp(vLoss), 2)},
	parameters: ${nParams},
	seeds: { init: ${INIT_SEED}, data: ${DATA_SEED} }
};
`;
		writeFileSync(new URL('lm-meta.ts', outDir), meta);
	};

	// Training loop.
	const rand = mulberry32(DATA_SEED);
	const total = new Float64Array(nParams);
	const adamM = new Float64Array(nParams);
	const adamV = new Float64Array(nParams);
	const maxStart = trainStream.length - (CONFIG.ctx + 1);
	let sinceLoss = 0;
	let sinceSteps = 0;
	let sinceTime = Date.now();
	if (checkpointAt.has(0)) writeCheckpoint(0, validLoss());
	for (let step = 1; step <= steps; step++) {
		const perWorker: number[][] = Array.from({ length: WORKERS }, () => []);
		for (let i = 0; i < BATCH; i++) perWorker[i % WORKERS].push(Math.floor(rand() * (maxStart + 1)));
		const results = await Promise.all(workers.map((w, i) => callWorker(w, { starts: perWorker[i] })));
		let stepLoss = 0;
		for (const r of results) stepLoss += r.loss / BATCH;
		sinceLoss += stepLoss;
		sinceSteps++;
		if (checkpointDir !== null) stepLosses.push(round(stepLoss, 4));

		total.fill(0);
		for (let w = 0; w < WORKERS; w++) {
			const g = gradArrays[w];
			for (let i = 0; i < nParams; i++) total[i] += g[i];
		}
		let sq = 0;
		for (let i = 0; i < nParams; i++) {
			total[i] /= BATCH;
			sq += total[i] * total[i];
		}
		const norm = Math.sqrt(sq);
		const clip = norm > CLIP ? CLIP / norm : 1;
		let lr: number;
		if (step <= WARMUP) lr = (LR * step) / WARMUP;
		else {
			const progress = steps > WARMUP ? (step - WARMUP) / (steps - WARMUP) : 1;
			lr = LR_END + 0.5 * (LR - LR_END) * (1 + Math.cos(Math.PI * progress));
		}
		const c1 = 1 - BETA1 ** step;
		const c2 = 1 - BETA2 ** step;
		for (let i = 0; i < nParams; i++) {
			const g = total[i] * clip;
			adamM[i] = BETA1 * adamM[i] + (1 - BETA1) * g;
			adamV[i] = BETA2 * adamV[i] + (1 - BETA2) * g * g;
			params[i] -= (lr * (adamM[i] / c1)) / (Math.sqrt(adamV[i] / c2) + EPS);
		}

		const isEval = step % EVAL_EVERY === 0 || step === steps;
		const vLoss = isEval || checkpointAt.has(step) ? validLoss() : 0;
		if (checkpointAt.has(step)) writeCheckpoint(step, vLoss);
		if (isEval) {
			const now = Date.now();
			const tokensPerSecond = (sinceSteps * BATCH * CONFIG.ctx) / ((now - sinceTime) / 1000);
			console.log(
				`step ${step}  train loss ${(sinceLoss / sinceSteps).toFixed(4)}  valid loss ${vLoss.toFixed(4)}  perplexity ${Math.exp(vLoss).toFixed(2)}  ` +
					`${Math.round(tokensPerSecond)} tokens/s  elapsed ${((now - t0) / 1000).toFixed(1)}s`
			);
			console.log(`  sample: ${sample()}`);
			writeFiles(step, vLoss);
			sinceLoss = 0;
			sinceSteps = 0;
			sinceTime = Date.now();
		}
	}
	await Promise.all(workers.map((w) => w.terminate()));
	console.log(`wrote ${fileURLToPath(new URL('tinystories.bin', outDir))} and lm-meta.ts`);
}

if (isMainThread) {
	main().catch((e) => {
		console.error(e);
		process.exit(1);
	});
} else {
	runWorker();
}

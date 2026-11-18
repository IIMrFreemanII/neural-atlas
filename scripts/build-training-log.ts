/**
 * Turns the checkpoints saved by scripts/train-tinystories.ts into the data file of the "Watching a model learn"
 * lesson, src/lib/lessons/llm/training/log.ts.
 * Run from the project root:
 *   node --experimental-strip-types scripts/build-training-log.ts <checkpoint folder> [--partial] [--weights FILE] [--out FILE]
 * The folder holds step-NNNNNN.bin (raw little-endian float32 parameters) and curve.json.
 * A finished run must end with the same bytes as the weights the other language lessons run (--weights, default
 * shared/tinystories.bin). An unfinished run is only accepted with --partial.
 * Everything is deterministic: the same folder gives the same file, byte for byte.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { nextLogits, paramCount } from '../src/lib/ml/gpt.ts';
import type { Gpt, GptConfig } from '../src/lib/ml/gpt.ts';
import { mulberry32 } from '../src/lib/ml/rng.ts';
import { sampleIndex } from '../src/lib/ml/sampling.ts';
import { softmax } from '../src/lib/ml/vec.ts';
import { END, UNK, encode, vocabFromWords } from '../src/lib/ml/wordtok.ts';
import { LM_CONFIG, LM_VOCAB } from '../src/lib/lessons/llm/shared/lm-meta.ts';
import { PROMPTS } from '../src/lib/lessons/llm/sampling/data.ts';
import type { Checkpoint, TrainLog } from '../src/lib/lessons/llm/training/log-types.ts';

const PROBE = 'tom and his dog went to the park . they saw a big red ball .';
const STORY_TOKENS = 28; // tokens written after each prompt
const SAMPLE_SEED = 2024; // sampled[p] uses mulberry32(SAMPLE_SEED + p), restarted at every checkpoint
const CURVE_POINTS = 160; // about this many points on the loss curve

const USAGE =
	'Usage: node --experimental-strip-types scripts/build-training-log.ts <checkpoint folder> [--partial] [--weights FILE] [--out FILE]';

function fail(message: string): never {
	console.error(message);
	process.exit(1);
}

interface CurveFile {
	config: GptConfig;
	steps: number;
	batch: number;
	evalSequences: number;
	trainLoss: number[];
	checkpoints: { step: number; validLoss: number }[];
}

// ---- arguments ----
let folder: string | null = null;
let partial = false;
let weightsPath = new URL('../src/lib/lessons/llm/shared/tinystories.bin', import.meta.url).pathname;
let outPath = new URL('../src/lib/lessons/llm/training/log.ts', import.meta.url).pathname;
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i++) {
	const a = args[i];
	if (a === '--partial') partial = true;
	else if (a === '--weights' || a === '--out') {
		const v = args[++i];
		if (v === undefined) fail(`${a} needs a file name.\n${USAGE}`);
		if (a === '--weights') weightsPath = v;
		else outPath = v;
	} else if (a.startsWith('--')) fail(`Unknown option ${a}.\n${USAGE}`);
	else if (folder === null) folder = a;
	else fail(`Unexpected argument ${a}.\n${USAGE}`);
}
if (folder === null) fail(USAGE);

// ---- input ----
const curvePath = `${folder}/curve.json`;
if (!existsSync(curvePath)) fail(`No curve.json in ${folder}.`);
const curve = JSON.parse(readFileSync(curvePath, 'utf8')) as CurveFile;
if (!Array.isArray(curve.checkpoints) || curve.checkpoints.length < 2) fail('curve.json lists fewer than 2 checkpoints.');
if (!Array.isArray(curve.trainLoss) || curve.trainLoss.length === 0) fail('curve.json has no trainLoss.');
const config = curve.config;
const sameConfig = (Object.keys(LM_CONFIG) as (keyof GptConfig)[]).every((k) => config[k] === LM_CONFIG[k]);
const last = curve.checkpoints[curve.checkpoints.length - 1];
const complete = last.step === curve.steps;

if (complete) {
	if (!sameConfig) fail(`curve.json config ${JSON.stringify(config)} differs from LM_CONFIG ${JSON.stringify(LM_CONFIG)}.`);
	if (!existsSync(weightsPath)) fail(`Weights file ${weightsPath} not found.`);
	const finished = readFileSync(weightsPath);
	if (!existsSync(`${folder}/${fileName(last.step)}`)) fail(`Checkpoint file ${folder}/${fileName(last.step)} is missing.`);
	const lastBytes = readFileSync(`${folder}/${fileName(last.step)}`);
	if (!lastBytes.equals(finished)) {
		fail(`The last checkpoint (step ${last.step}) is not byte for byte the same as ${weightsPath}. The log would describe a different model than the other lessons run. Nothing written.`);
	}
} else if (!partial) {
	fail(`The run is not finished: the last checkpoint is step ${last.step} of ${curve.steps}. Pass --partial to build a log from what exists.`);
} else {
	console.warn(`WARNING: partial log. The run has reached step ${last.step} of ${curve.steps}; the byte comparison with the finished weights is skipped.`);
	if (!sameConfig) fail(`curve.json config ${JSON.stringify(config)} differs from LM_CONFIG ${JSON.stringify(LM_CONFIG)}.`);
}

function fileName(step: number): string {
	return `step-${String(step).padStart(6, '0')}.bin`;
}

const vocab = vocabFromWords(LM_VOCAB);
const expectedBytes = paramCount(LM_CONFIG) * 4;

function loadModel(step: number): Gpt {
	const path = `${folder}/${fileName(step)}`;
	if (!existsSync(path)) fail(`Checkpoint file ${path} is missing.`);
	const bytes = readFileSync(path);
	if (bytes.byteLength !== expectedBytes) fail(`${path} has ${bytes.byteLength} bytes, expected ${expectedBytes}.`);
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const params = new Float32Array(expectedBytes / 4);
	for (let i = 0; i < params.length; i++) params[i] = view.getFloat32(i * 4, true);
	return { config: LM_CONFIG, params };
}

// ---- helpers ----
const round = (x: number, digits: number): number => Number(x.toFixed(digits));
const text = (id: number): string => vocab.words[id];

function mean(values: number[], from: number, to: number): number {
	let sum = 0;
	for (let i = from; i <= to; i++) sum += values[i];
	return sum / (to - from + 1);
}

function maskedScores(model: Gpt, ids: number[]): Float32Array {
	const scores = nextLogits(model, ids);
	scores[UNK] = -Infinity;
	return scores;
}

function story(model: Gpt, prompt: number[], rand: (() => number) | null): string[] {
	const ids = prompt.slice();
	const out: string[] = [];
	for (let n = 0; n < STORY_TOKENS; n++) {
		const scores = maskedScores(model, ids);
		let pick = 0;
		if (rand === null) {
			for (let i = 1; i < scores.length; i++) if (scores[i] > scores[pick]) pick = i;
		} else {
			pick = sampleIndex(softmax(scores), rand());
		}
		ids.push(pick);
		out.push(text(pick));
		if (pick === END) break;
	}
	return out;
}

// ---- probe ----
const probeIds = encode(vocab, PROBE);
if (probeIds.some((id) => id === UNK)) fail('The probe sentence has a word that is not in the vocabulary.');
const probeTokens = probeIds.map(text);
const promptIds = PROMPTS.map((p) => {
	const ids = encode(vocab, p);
	if (ids.some((id) => id === UNK)) fail(`The prompt "${p}" has a word that is not in the vocabulary.`);
	return ids;
});

// ---- loss curve ----
const steps = curve.trainLoss.length;
const curveStep: number[] = [];
const curveLoss: number[] = [];
{
	const span = Math.log2(1 + steps / 25);
	const seen = new Set<number>();
	for (let k = 0; k < CURVE_POINTS; k++) {
		const s = k === CURVE_POINTS - 1 ? steps : Math.max(1, Math.round(25 * (2 ** ((span * k) / (CURVE_POINTS - 1)) - 1)));
		if (seen.has(s)) continue;
		seen.add(s);
		curveStep.push(s);
	}
	if (curveStep[0] !== 1) curveStep.unshift(1);
	for (const c of curveStep) {
		const half = Math.floor(c * 0.05);
		const from = Math.max(1, c - half);
		const to = Math.min(steps, c + half);
		curveLoss.push(round(mean(curve.trainLoss, from - 1, to - 1), 3));
	}
}

// ---- checkpoints ----
const checkpoints: Checkpoint[] = [];
const summary: string[] = [];
for (const entry of curve.checkpoints) {
	const model = loadModel(entry.step);
	let trainLoss: number;
	if (entry.step === 0) trainLoss = curve.trainLoss[0];
	else trainLoss = mean(curve.trainLoss, Math.max(0, entry.step - 5), Math.min(curve.trainLoss.length - 1, entry.step + 4));

	const greedy = promptIds.map((p) => story(model, p, null));
	const sampled = promptIds.map((p, i) => story(model, p, mulberry32(SAMPLE_SEED + i)));

	const probe: number[] = [];
	const guess: string[] = [];
	const guessP: number[] = [];
	for (let i = 0; i < probeIds.length - 1; i++) {
		const probs = softmax(nextLogits(model, probeIds.slice(0, i + 1)));
		probe.push(probs[probeIds[i + 1]]);
		let best = -1;
		for (let j = 0; j < probs.length; j++) {
			if (j === UNK) continue;
			if (best < 0 || probs[j] > probs[best]) best = j;
		}
		guess.push(text(best));
		guessP.push(probs[best]);
	}
	const probeLoss = probe.reduce((a, p) => a - Math.log(p), 0) / probe.length;

	checkpoints.push({
		step: entry.step,
		trainLoss: round(trainLoss, 4),
		validLoss: round(entry.validLoss, 4),
		greedy,
		sampled,
		probe: probe.map((p) => round(p, 4)),
		guess,
		guessP: guessP.map((p) => round(p, 4)),
		probeLoss: round(probeLoss, 4)
	});
	const head = (t: string[]): string => t.slice(0, 12).join(' ');
	summary.push(
		`step ${String(entry.step).padStart(6)}  train ${trainLoss.toFixed(4)}  valid ${entry.validLoss.toFixed(4)}  probe ${probeLoss.toFixed(4)}\n` +
			`    greedy:  ${head(greedy[0])}\n    sampled: ${head(sampled[0])}`
	);
}

const log: TrainLog = {
	prompts: PROMPTS.slice(),
	probe: probeTokens,
	vocab: curve.config.vocab,
	uniformLoss: round(Math.log(curve.config.vocab), 4),
	batch: curve.batch,
	ctx: curve.config.ctx,
	steps: curve.steps,
	curveStep,
	curveLoss,
	checkpoints
};

// ---- output ----
const str = (s: string): string => `'${s.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
const strs = (a: string[]): string => `[${a.map(str).join(', ')}]`;
const nums = (a: number[]): string => `[${a.join(', ')}]`;

const lines: string[] = [];
lines.push('// Written by scripts/build-training-log.ts. Do not edit by hand.');
lines.push(`// The model of the language lessons at ${checkpoints.length} moments of its training run (scripts/train-tinystories.ts).`);
lines.push("import type { TrainLog } from './log-types.ts';");
lines.push('');
lines.push('export const TRAIN_LOG: TrainLog = {');
lines.push(`\tprompts: ${strs(log.prompts)},`);
lines.push(`\tprobe: ${strs(log.probe)},`);
lines.push(`\tvocab: ${log.vocab},`);
lines.push(`\tuniformLoss: ${log.uniformLoss},`);
lines.push(`\tbatch: ${log.batch},`);
lines.push(`\tctx: ${log.ctx},`);
lines.push(`\tsteps: ${log.steps},`);
lines.push(`\tcurveStep: ${nums(log.curveStep)},`);
lines.push(`\tcurveLoss: ${nums(log.curveLoss)},`);
lines.push('\tcheckpoints: [');
checkpoints.forEach((c, i) => {
	lines.push('\t\t{');
	lines.push(`\t\t\tstep: ${c.step},`);
	lines.push(`\t\t\ttrainLoss: ${c.trainLoss},`);
	lines.push(`\t\t\tvalidLoss: ${c.validLoss},`);
	lines.push('\t\t\tgreedy: [');
	c.greedy.forEach((g, j) => lines.push(`\t\t\t\t${strs(g)}${j < c.greedy.length - 1 ? ',' : ''}`));
	lines.push('\t\t\t],');
	lines.push('\t\t\tsampled: [');
	c.sampled.forEach((g, j) => lines.push(`\t\t\t\t${strs(g)}${j < c.sampled.length - 1 ? ',' : ''}`));
	lines.push('\t\t\t],');
	lines.push(`\t\t\tprobe: ${nums(c.probe)},`);
	lines.push(`\t\t\tguess: ${strs(c.guess)},`);
	lines.push(`\t\t\tguessP: ${nums(c.guessP)},`);
	lines.push(`\t\t\tprobeLoss: ${c.probeLoss}`);
	lines.push(`\t\t}${i < checkpoints.length - 1 ? ',' : ''}`);
});
lines.push('\t]');
lines.push('};');
writeFileSync(outPath, lines.join('\n') + '\n');

console.log(summary.join('\n'));
console.log(`Wrote ${outPath} (${checkpoints.length} checkpoints, ${curveStep.length} curve points${complete ? '' : ', partial'}).`);

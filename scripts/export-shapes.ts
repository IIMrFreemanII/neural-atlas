/**
 * Writes the "Latent space" lesson's generated shape drawings (centeredShape) to a binary file, so that the
 * Python training script learns from exactly the drawings the lesson shows. scripts/train-autoencoder.py runs
 * it; it is not meant to be run by hand.
 *
 *   node --experimental-strip-types scripts/export-shapes.ts <out file> <count> <seed>
 *
 * The file holds `count` drawings of 16 x 16 pixels, each pixel one byte (0..255, row-major), followed by
 * `count` labels of one byte each (0 circle, 1 square, 2 triangle, 3 cross).
 */
import { writeFileSync } from 'node:fs';
import { mulberry32 } from '../src/lib/ml/rng.ts';
import { centeredShape } from '../src/lib/lessons/vision/autoencoder/data.ts';

const [out, countArg, seedArg] = process.argv.slice(2);
const count = Number(countArg);
const seed = Number(seedArg);
if (!out || !Number.isInteger(count) || count <= 0 || !Number.isInteger(seed)) {
	console.error('usage: export-shapes.ts <out file> <count> <seed>');
	process.exit(1);
}

const PIXELS = 256;
const bytes = new Uint8Array(count * (PIXELS + 1));
const rand = mulberry32(seed);
for (let i = 0; i < count; i++) {
	const { image, label } = centeredShape(rand);
	for (let p = 0; p < PIXELS; p++) bytes[i * PIXELS + p] = Math.round(image[p] * 255);
	bytes[count * PIXELS + i] = label;
}
writeFileSync(out, bytes);

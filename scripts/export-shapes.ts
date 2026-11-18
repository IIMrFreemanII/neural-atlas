/**
 * Writes the lessons' generated shape drawings to a binary file, so that the Python training scripts learn from
 * exactly the drawings the lessons show. scripts/train-autoencoder.py and scripts/train-vit.py run it; it is not
 * meant to be run by hand.
 *
 *   node --experimental-strip-types scripts/export-shapes.ts <out file> <count> <seed> [centered|random|samples]
 *
 * The generator is `centered` (the "Latent space" lesson's centeredShape, the default), `random` (randomShape of
 * the "Features" lesson: any position, size and rotation) or `samples` (the four clean SAMPLES of that lesson,
 * in that order, labels 0..3; count and seed are ignored but must still be valid numbers).
 *
 * The file holds `count` drawings of 16 x 16 pixels, each pixel one byte (0..255, row-major), followed by
 * `count` labels of one byte each (0 circle, 1 square, 2 triangle, 3 cross). The one exception is `samples`: its
 * file holds the 4 x 256 pixels as little-endian float32, exactly as SAMPLES has them (no labels, they are 0..3).
 */
import { writeFileSync } from 'node:fs';
import { mulberry32 } from '../src/lib/ml/rng.ts';
import { centeredShape } from '../src/lib/lessons/vision/autoencoder/data.ts';
import { randomShape, SAMPLES } from '../src/lib/lessons/vision/cnn-features/data.ts';

const [out, countArg, seedArg, generator = 'centered'] = process.argv.slice(2);
const count = Number(countArg);
const seed = Number(seedArg);
if (
	!out ||
	!Number.isInteger(count) ||
	count <= 0 ||
	!Number.isInteger(seed) ||
	!['centered', 'random', 'samples'].includes(generator)
) {
	console.error('usage: export-shapes.ts <out file> <count> <seed> [centered|random|samples]');
	process.exit(1);
}

const PIXELS = 256;
if (generator === 'samples') {
	// Exact float32 pixels (little-endian), so a reference computed from them matches the lesson's own SAMPLES.
	const floats = new Float32Array(SAMPLES.length * PIXELS);
	SAMPLES.forEach((image, i) => floats.set(image, i * PIXELS));
	writeFileSync(out, new Uint8Array(floats.buffer));
	process.exit(0);
}

const bytes = new Uint8Array(count * (PIXELS + 1));
const rand = mulberry32(seed);
const draw = generator === 'random' ? randomShape : centeredShape;
for (let i = 0; i < count; i++) {
	const { image, label } = draw(rand);
	for (let p = 0; p < PIXELS; p++) bytes[i * PIXELS + p] = Math.round(image[p] * 255);
	bytes[count * PIXELS + i] = label;
}
writeFileSync(out, bytes);

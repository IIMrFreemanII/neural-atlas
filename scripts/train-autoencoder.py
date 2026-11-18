"""
Trains the autoencoder of the "Latent space" lesson with PyTorch, on the Mac's GPU (MPS) when there is one,
and writes its weights to src/lib/lessons/vision/autoencoder/weights.ts. Run it from the project root:

    uv run --no-project --with torch python scripts/train-autoencoder.py

The drawings come from the lessons' own generator (scripts/export-shapes.ts, run through Node), so the
network learns exactly the shapes the lesson shows. The lesson itself runs the network in TypeScript
(src/lib/ml/autoencoder.ts); this script only has to write the weights in the order that file reads them.
Results on the GPU are not bit-for-bit repeatable, so a rerun gives slightly different weights.
"""

import argparse
import base64
import re
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
MODEL_TS = ROOT / 'src/lib/ml/autoencoder.ts'
OUT = ROOT / 'src/lib/lessons/vision/autoencoder/weights.ts'
PIXELS = 256
LATENT = 2
# Read from the TypeScript model, so the two cannot disagree.
HIDDEN = int(re.search(r'export const HIDDEN = (\d+)', MODEL_TS.read_text()).group(1))

TRAIN_COUNT = 200_000
TRAIN_SEED = 1001
EVAL_COUNT = 2_000
EVAL_SEED = 2002


def drawings(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
	"""`count` drawings (count x 256, values 0..1) and their labels, from the lessons' generator."""
	with tempfile.TemporaryDirectory() as tmp:
		path = Path(tmp) / 'shapes.bin'
		subprocess.run(
			['node', '--experimental-strip-types', 'scripts/export-shapes.ts', str(path), str(count), str(seed)],
			cwd=ROOT,
			check=True,
		)
		raw = bytearray(path.read_bytes())
	data = torch.frombuffer(raw, dtype=torch.uint8)
	images = data[: count * PIXELS].reshape(count, PIXELS).float() / 255
	labels = data[count * PIXELS :].long()
	return images, labels


def build() -> nn.Sequential:
	# The layers of createAutoencoder in src/lib/ml/autoencoder.ts: tanh on every hidden layer, so the code
	# is in -1..1, and a sigmoid on the output (inside the loss here, applied by the lesson).
	return nn.Sequential(
		nn.Linear(PIXELS, HIDDEN), nn.Tanh(),
		nn.Linear(HIDDEN, LATENT), nn.Tanh(),
		nn.Linear(LATENT, HIDDEN), nn.Tanh(),
		nn.Linear(HIDDEN, PIXELS),
	)


def pack(model: nn.Sequential) -> str:
	"""W0, b0, W1, b1, W2, b2, W3, b3 as little-endian float32, base64. A torch weight is [out, in], row-major,
	which is the layout of the TypeScript MLP: W[l][j * in + i] connects input i to output j."""
	values: list[float] = []
	for layer in model:
		if isinstance(layer, nn.Linear):
			values += layer.weight.detach().cpu().flatten().tolist()
			values += layer.bias.detach().cpu().flatten().tolist()
	return base64.b64encode(struct.pack(f'<{len(values)}f', *values)).decode('ascii')


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--steps', type=int, default=20_000)
	parser.add_argument('--batch', type=int, default=256)
	parser.add_argument('--lr', type=float, default=0.003)
	parser.add_argument('--lr-end', type=float, default=0.0003)
	parser.add_argument('--seed', type=int, default=7)
	parser.add_argument('--out', type=Path, default=OUT)
	parser.add_argument('--cpu', action='store_true', help='train on the CPU even when a GPU is available')
	args = parser.parse_args()

	device = torch.device('mps' if torch.backends.mps.is_available() and not args.cpu else 'cpu')
	print(f'device {device}, hidden {HIDDEN}')
	torch.manual_seed(args.seed)

	t0 = time.time()
	train_x, _ = drawings(TRAIN_COUNT, TRAIN_SEED)
	eval_x, _ = drawings(EVAL_COUNT, EVAL_SEED)
	train_x = train_x.to(device)
	eval_x = eval_x.to(device)
	print(f'{TRAIN_COUNT} training drawings ready ({time.time() - t0:.1f} s)')

	model = build().to(device)
	loss_fn = nn.BCEWithLogitsLoss()
	opt = torch.optim.Adam(model.parameters(), lr=args.lr)
	sched = torch.optim.lr_scheduler.LambdaLR(
		opt, lambda step: 1 + (args.lr_end / args.lr - 1) * min(1.0, step / args.steps)
	)
	order = torch.Generator().manual_seed(args.seed)

	t0 = time.time()
	perm = torch.randperm(TRAIN_COUNT, generator=order).to(device)
	pos = 0
	run_loss = torch.zeros((), device=device)
	for step in range(1, args.steps + 1):
		if pos + args.batch > TRAIN_COUNT:
			perm = torch.randperm(TRAIN_COUNT, generator=order).to(device)
			pos = 0
		x = train_x[perm[pos : pos + args.batch]]
		pos += args.batch
		loss = loss_fn(model(x), x)
		opt.zero_grad(set_to_none=True)
		loss.backward()
		opt.step()
		sched.step()
		run_loss += loss.detach()
		if step % 2500 == 0:
			with torch.no_grad():
				held_out = loss_fn(model(eval_x), eval_x).item()
			print(f'step {step}  loss {run_loss.item() / 2500:.4f}  held-out {held_out:.4f}  ({time.time() - t0:.1f} s)')
			run_loss.zero_()
	print(f'trained in {time.time() - t0:.1f} s')

	with torch.no_grad():
		held_out = loss_fn(model(eval_x), eval_x).item()
	args.out.write_text(
		'// Written by scripts/train-autoencoder.py. Do not edit by hand.\n'
		f'// PyTorch on {device}: Adam, learning rate {args.lr} falling to {args.lr_end}, batch {args.batch}, '
		f'{args.steps} steps, {TRAIN_COUNT} generated drawings (seed {TRAIN_SEED}), init seed {args.seed}.\n'
		f'// Held-out loss (binary cross-entropy per pixel, {EVAL_COUNT} drawings, seed {EVAL_SEED}): {held_out:.4f}.\n'
		"import { unpackWeights } from '../../../ml/autoencoder.ts';\n\n"
		f"export const WEIGHTS = unpackWeights('{pack(model)}');\n"
	)
	print(f'wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}')


if __name__ == '__main__':
	main()

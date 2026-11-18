"""
Trains the vision transformer of the "Vision transformer" lesson with PyTorch, on the Mac's GPU (MPS) when there
is one, and writes its weights to src/lib/lessons/vision/vit/weights.ts. Run it from the project root:

    uv run --no-project --with torch python scripts/train-vit.py

The drawings come from the lessons' own generator (scripts/export-shapes.ts, run through Node: randomShape of the
"Features" lesson), so the network learns exactly the shapes the lesson shows. The lesson runs the network in
TypeScript (src/lib/ml/vit.ts); the model below is written by hand with the same layout, and this script writes
the weights in the order that file reads them. It also writes the logits of the four clean SAMPLES, computed here
in PyTorch, so a test can check that the TypeScript forward pass gives the same numbers.
Results on the GPU are not bit-for-bit repeatable, so a rerun gives slightly different weights.
"""

import argparse
import base64
import math
import re
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
MODEL_TS = ROOT / 'src/lib/ml/vit.ts'
OUT = ROOT / 'src/lib/lessons/vision/vit/weights.ts'


def constant(name: str) -> int:
	# Read from the TypeScript model, so the two cannot disagree.
	return int(re.search(rf'export const {name} = (\d+)', MODEL_TS.read_text()).group(1))


IMAGE = constant('IMAGE')
PATCH = constant('PATCH')
GRID = constant('GRID')
PATCHES = constant('PATCHES')
TOKENS = constant('TOKENS')
PATCH_PIXELS = constant('PATCH_PIXELS')
DIM = constant('DIM')
HEADS = constant('HEADS')
HEAD_DIM = constant('HEAD_DIM')
HIDDEN = constant('HIDDEN')
LAYERS = constant('LAYERS')
CLASSES = constant('CLASSES')
PIXELS = IMAGE * IMAGE

TRAIN_COUNT = 200_000
TRAIN_SEED = 1001
EVAL_COUNT = 2_000
EVAL_SEED = 2002


def export_shapes(generator: str, count: int, seed: int) -> bytes:
	with tempfile.TemporaryDirectory() as tmp:
		path = Path(tmp) / 'shapes.bin'
		subprocess.run(
			['node', '--experimental-strip-types', 'scripts/export-shapes.ts', str(path), str(count), str(seed), generator],
			cwd=ROOT,
			check=True,
		)
		return path.read_bytes()


def drawings(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
	"""`count` drawings (count x 256, values 0..1) and their labels, from the lessons' generator."""
	data = torch.frombuffer(bytearray(export_shapes('random', count, seed)), dtype=torch.uint8)
	images = data[: count * PIXELS].reshape(count, PIXELS).float() / 255
	labels = data[count * PIXELS :].long()
	return images, labels


def samples() -> torch.Tensor:
	"""The four clean SAMPLES of the "Features" lesson, 4 x 256, exact float32."""
	raw = bytearray(export_shapes('samples', 1, 0))
	return torch.frombuffer(raw, dtype=torch.float32).reshape(-1, PIXELS).clone()


class Block(nn.Module):
	"""One pre-norm block: x += attention(LN1(x)); x += mlp(LN2(x))."""

	def __init__(self) -> None:
		super().__init__()
		self.ln1 = nn.LayerNorm(DIM, eps=1e-5)
		self.q = nn.Linear(DIM, DIM)
		self.k = nn.Linear(DIM, DIM)
		self.v = nn.Linear(DIM, DIM)
		self.o = nn.Linear(DIM, DIM)
		self.ln2 = nn.LayerNorm(DIM, eps=1e-5)
		self.fc1 = nn.Linear(DIM, HIDDEN)
		self.fc2 = nn.Linear(HIDDEN, DIM)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		n = x.shape[0]
		h = self.ln1(x)
		# (n, TOKENS, DIM) -> (n, HEADS, TOKENS, HEAD_DIM): head j uses channels [16j, 16j + 16).
		split = lambda t: t.reshape(n, TOKENS, HEADS, HEAD_DIM).transpose(1, 2)
		q, k, v = split(self.q(h)), split(self.k(h)), split(self.v(h))
		weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM), dim=-1)
		a = (weights @ v).transpose(1, 2).reshape(n, TOKENS, DIM)
		x = x + self.o(a)
		return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate='tanh'))


class Vit(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.embed = nn.Linear(PATCH_PIXELS, DIM)
		self.cls = nn.Parameter(torch.randn(DIM) * 0.02)
		self.pos = nn.Parameter(torch.randn(TOKENS, DIM) * 0.02)
		self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
		self.ln_f = nn.LayerNorm(DIM, eps=1e-5)
		self.head = nn.Linear(DIM, CLASSES)

	def forward(self, images: torch.Tensor) -> torch.Tensor:
		"""images: n x 256, row-major, row 0 at the top. Returns n x CLASSES logits."""
		n = images.shape[0]
		# Patch p = pr * GRID + pc, its pixels row-major: image[(pr*4 + r) * 16 + pc*4 + c].
		patches = (
			images.reshape(n, GRID, PATCH, GRID, PATCH).permute(0, 1, 3, 2, 4).reshape(n, PATCHES, PATCH_PIXELS)
		)
		x = torch.cat([self.cls.expand(n, 1, DIM), self.embed(patches)], dim=1) + self.pos
		for block in self.blocks:
			x = block(x)
		return self.head(self.ln_f(x[:, 0]))


def tensors(model: Vit) -> list[torch.Tensor]:
	"""E, bE, cls, pos, each layer's g1, b1, Wq, bq, Wk, bk, Wv, bv, Wo, bo, g2, b2, W1, b1m, W2, b2m, then gf, bf,
	Wh, bh. A torch Linear weight is [out, in], row-major, the layout of vit.ts."""
	out = [model.embed.weight, model.embed.bias, model.cls, model.pos]
	for b in model.blocks:
		out += [b.ln1.weight, b.ln1.bias]
		for lin in (b.q, b.k, b.v, b.o):
			out += [lin.weight, lin.bias]
		out += [b.ln2.weight, b.ln2.bias, b.fc1.weight, b.fc1.bias, b.fc2.weight, b.fc2.bias]
	out += [model.ln_f.weight, model.ln_f.bias, model.head.weight, model.head.bias]
	return out


def pack(model: Vit) -> str:
	"""All weights in that order as little-endian float32, base64."""
	values: list[float] = []
	for t in tensors(model):
		values += t.detach().cpu().flatten().tolist()
	return base64.b64encode(struct.pack(f'<{len(values)}f', *values)).decode('ascii')


def build_optimizer(model: Vit, lr: float) -> torch.optim.AdamW:
	# Weight decay on the matrices only: not on biases, layer-norm gains, the class token or the positions.
	decay = [p for p in model.parameters() if p.ndim == 2 and p is not model.pos]
	plain = [p for p in model.parameters() if not (p.ndim == 2 and p is not model.pos)]
	return torch.optim.AdamW(
		[{'params': decay, 'weight_decay': 0.05}, {'params': plain, 'weight_decay': 0.0}], lr=lr
	)


def accuracy(model: Vit, x: torch.Tensor, y: torch.Tensor) -> float:
	model.eval()
	with torch.no_grad():
		correct = sum((model(x[i : i + 1000]).argmax(1) == y[i : i + 1000]).sum().item() for i in range(0, len(x), 1000))
	model.train()
	return correct / len(x)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--steps', type=int, default=20_000)
	parser.add_argument('--batch', type=int, default=256)
	parser.add_argument('--lr', type=float, default=0.002)
	parser.add_argument('--lr-end', type=float, default=0.00001)
	parser.add_argument('--warmup', type=int, default=500)
	parser.add_argument('--seed', type=int, default=7)
	parser.add_argument(
		'--label-smoothing',
		type=float,
		default=0.1,
		help='keeps the model from answering with 100%% certainty on every drawing, scribbles included',
	)
	parser.add_argument('--out', type=Path, default=OUT)
	parser.add_argument('--cpu', action='store_true', help='train on the CPU even when a GPU is available')
	args = parser.parse_args()

	device = torch.device('mps' if torch.backends.mps.is_available() and not args.cpu else 'cpu')
	print(f'device {device}, dim {DIM}, heads {HEADS}, layers {LAYERS}')
	torch.manual_seed(args.seed)

	t0 = time.time()
	train_x, train_y = drawings(TRAIN_COUNT, TRAIN_SEED)
	eval_x, eval_y = drawings(EVAL_COUNT, EVAL_SEED)
	train_x, train_y, eval_x, eval_y = (t.to(device) for t in (train_x, train_y, eval_x, eval_y))
	print(f'{TRAIN_COUNT} training drawings ready ({time.time() - t0:.1f} s)')

	model = Vit().to(device)
	print(f'{sum(p.numel() for p in model.parameters())} parameters')
	opt = build_optimizer(model, args.lr)

	def lr_factor(step: int) -> float:
		if step < args.warmup:
			return (step + 1) / args.warmup
		t = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
		end = args.lr_end / args.lr
		return end + (1 - end) * 0.5 * (1 + math.cos(math.pi * t))

	sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
	order = torch.Generator().manual_seed(args.seed)

	t0 = time.time()
	perm = torch.randperm(TRAIN_COUNT, generator=order).to(device)
	pos = 0
	run_loss = torch.zeros((), device=device)
	for step in range(1, args.steps + 1):
		if pos + args.batch > TRAIN_COUNT:
			perm = torch.randperm(TRAIN_COUNT, generator=order).to(device)
			pos = 0
		idx = perm[pos : pos + args.batch]
		pos += args.batch
		loss = F.cross_entropy(model(train_x[idx]), train_y[idx], label_smoothing=args.label_smoothing)
		opt.zero_grad(set_to_none=True)
		loss.backward()
		opt.step()
		sched.step()
		run_loss += loss.detach()
		if step % 2000 == 0:
			print(
				f'step {step}  loss {run_loss.item() / 2000:.4f}  eval accuracy {accuracy(model, eval_x, eval_y):.4f}'
				f'  ({time.time() - t0:.1f} s)'
			)
			run_loss.zero_()
	print(f'trained in {time.time() - t0:.1f} s')

	final = accuracy(model, eval_x, eval_y)
	print(f'final eval accuracy {final:.4f}')

	# Reference logits for the TypeScript test, from float32 on the CPU.
	model = model.to('cpu').eval()
	with torch.no_grad():
		reference = model(samples()).tolist()
	rows = ',\n'.join('\t[' + ', '.join(f'{v:.6f}' for v in row) + ']' for row in reference)

	args.out.write_text(
		'// Written by scripts/train-vit.py. Do not edit by hand.\n'
		f'// PyTorch: AdamW (weight decay 0.05 on matrices), learning rate {args.lr} with {args.warmup} warm-up steps, '
		f'cosine to {args.lr_end}, label smoothing {args.label_smoothing}, batch {args.batch}, {args.steps} steps, {TRAIN_COUNT} generated drawings '
		f'(seed {TRAIN_SEED}), init seed {args.seed}.\n'
		f'// Eval accuracy ({EVAL_COUNT} drawings, seed {EVAL_SEED}): {final:.4f}.\n'
		"import { unpackVit } from '../../../ml/vit.ts';\n\n"
		f"export const WEIGHTS = unpackVit('{pack(model)}');\n\n"
		'/** Logits of the four clean SAMPLES (cnn-features/data.ts), computed by PyTorch in float32 on the CPU. */\n'
		f'export const REFERENCE_LOGITS: number[][] = [\n{rows}\n];\n'
	)
	print(f'wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}')


if __name__ == '__main__':
	main()

import { cosine, dot } from './vec.ts';
import type { Mat, Vec } from './vec.ts';

/** The k rows most similar to `query` by cosine similarity, best first, skipping the indices in `exclude`. */
export function nearest(
	vectors: Mat,
	query: Vec,
	k: number,
	exclude: readonly number[] = []
): { index: number; similarity: number }[] {
	const skip = new Set(exclude);
	const scored: { index: number; similarity: number }[] = [];
	for (let i = 0; i < vectors.length; i++) {
		if (skip.has(i)) continue;
		scored.push({ index: i, similarity: cosine(vectors[i], query) });
	}
	scored.sort((p, q) => q.similarity - p.similarity || p.index - q.index);
	return scored.slice(0, Math.max(0, k));
}

/** b - a + c: "a is to b as c is to ?" */
export function analogy(a: Vec, b: Vec, c: Vec): Vec {
	return a.map((_, i) => b[i] - a[i] + c[i]);
}

/** Eigen decomposition of a symmetric matrix by cyclic Jacobi rotations. Returns eigenvalues and eigenvectors (columns of `vectors`). */
function jacobiEigen(sym: Mat): { values: Vec; vectors: Mat } {
	const n = sym.length;
	const a = sym.map((row) => row.slice());
	const v: Mat = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
	for (let sweep = 0; sweep < 60; sweep++) {
		let off = 0;
		for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
		if (off < 1e-24) break;
		for (let p = 0; p < n - 1; p++) {
			for (let q = p + 1; q < n; q++) {
				if (Math.abs(a[p][q]) < 1e-300) continue;
				const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
				const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
				const c = 1 / Math.sqrt(t * t + 1);
				const s = t * c;
				for (let k = 0; k < n; k++) {
					const akp = a[k][p];
					const akq = a[k][q];
					a[k][p] = c * akp - s * akq;
					a[k][q] = s * akp + c * akq;
				}
				for (let k = 0; k < n; k++) {
					const apk = a[p][k];
					const aqk = a[q][k];
					a[p][k] = c * apk - s * aqk;
					a[q][k] = s * apk + c * aqk;
				}
				for (let k = 0; k < n; k++) {
					const vkp = v[k][p];
					const vkq = v[k][q];
					v[k][p] = c * vkp - s * vkq;
					v[k][q] = s * vkp + c * vkq;
				}
			}
		}
	}
	return { values: a.map((row, i) => row[i]), vectors: v };
}

/**
 * Principal component analysis down to 3 dimensions. Centers the rows, finds the top 3 eigenvectors of the
 * covariance matrix (Jacobi eigenvalue iteration on the d x d symmetric matrix; deterministic), and projects.
 * Each component's sign is fixed so its largest-magnitude entry is positive.
 */
export function pca3(vectors: Mat): { points: Mat; components: Mat; mean: Vec; explained: [number, number, number] } {
	const n = vectors.length;
	const d = vectors[0]?.length ?? 0;
	const mean: Vec = new Array<number>(d).fill(0);
	for (const row of vectors) for (let j = 0; j < d; j++) mean[j] += row[j] / n;
	const centered = vectors.map((row) => row.map((x, j) => x - mean[j]));
	const cov: Mat = Array.from({ length: d }, () => new Array<number>(d).fill(0));
	for (const row of centered) for (let i = 0; i < d; i++) for (let j = i; j < d; j++) cov[i][j] += row[i] * row[j];
	for (let i = 0; i < d; i++) {
		for (let j = i; j < d; j++) {
			cov[i][j] /= n;
			cov[j][i] = cov[i][j];
		}
	}
	let total = 0;
	for (let i = 0; i < d; i++) total += cov[i][i];
	const { values, vectors: eig } = jacobiEigen(cov);
	const order = values.map((_, i) => i).sort((p, q) => values[q] - values[p] || p - q);
	const components: Mat = [];
	const frac: number[] = [];
	for (let r = 0; r < 3; r++) {
		const col = order[r];
		const comp: Vec = eig.map((row) => row[col]);
		let big = 0;
		for (let j = 1; j < d; j++) if (Math.abs(comp[j]) > Math.abs(comp[big]) + 1e-12) big = j;
		if (comp[big] < 0) for (let j = 0; j < d; j++) comp[j] = -comp[j];
		components.push(comp);
		frac.push(total > 0 ? Math.max(0, values[col]) / total : 0);
	}
	const points = centered.map((row) => components.map((comp) => dot(row, comp)));
	return { points, components, mean, explained: [frac[0], frac[1], frac[2]] };
}

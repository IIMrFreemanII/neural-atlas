export type Vec2 = [number, number];

export interface Surface {
	id: 0 | 1 | 2;
	name: string;
	f(x: number, y: number): number;
	/** Analytic gradient. */
	grad(x: number, y: number): Vec2;
	/** Default starting point. */
	start: Vec2;
	/** Global minimum first. */
	minima: Vec2[];
}

/** Newton's method for the stationary points of the two-valleys surface along y = 0. */
function twoValleyMinima(): Vec2[] {
	const d = (x: number) => 2 * x * x * x - 2 * x + 0.3;
	const dd = (x: number) => 6 * x * x - 2;
	const solve = (x0: number) => {
		let x = x0;
		for (let i = 0; i < 50; i++) x -= d(x) / dd(x);
		return x;
	};
	const left = solve(-1.1);
	const right = solve(0.9);
	return [
		[left, 0],
		[right, 0]
	];
}

const bowl: Surface = {
	id: 0,
	name: 'Bowl',
	f: (x, y) => 0.5 * (x * x + y * y),
	grad: (x, y) => [x, y],
	start: [2, 1.2],
	minima: [[0, 0]]
};

const ravine: Surface = {
	id: 1,
	name: 'Ravine',
	f: (x, y) => 0.1 * x * x + 2 * y * y,
	grad: (x, y) => [0.2 * x, 4 * y],
	start: [-2.2, 1.0],
	minima: [[0, 0]]
};

const valleys: Surface = {
	id: 2,
	name: 'Two valleys',
	f: (x, y) => 0.5 * (x * x - 1) * (x * x - 1) + y * y + 0.3 * x,
	grad: (x, y) => [2 * x * (x * x - 1) + 0.3, 2 * y],
	start: [0.3, 1.2],
	minima: twoValleyMinima()
};

export const SURFACES: readonly [Surface, Surface, Surface] = [bowl, ravine, valleys];

export const DOMAIN: { x: [number, number]; y: [number, number] } = {
	x: [-2.5, 2.5],
	y: [-1.6, 1.6]
};

export type OptimizerId = 'sgd' | 'momentum' | 'adam';

export interface OptState {
	p: Vec2;
	/** Momentum velocity. */
	v: Vec2;
	/** Adam first moment. */
	m: Vec2;
	/** Adam second moment. */
	s: Vec2;
	t: number;
}

export function initState(p: Vec2): OptState {
	return { p: [p[0], p[1]], v: [0, 0], m: [0, 0], s: [0, 0], t: 0 };
}

const B1 = 0.9;
const B2 = 0.999;
const EPS = 1e-8;

/** Returns a new state after one update. Does not mutate the input. */
export function step(kind: OptimizerId, st: OptState, g: Vec2, lr: number, beta: number): OptState {
	const next: OptState = {
		p: [st.p[0], st.p[1]],
		v: [st.v[0], st.v[1]],
		m: [st.m[0], st.m[1]],
		s: [st.s[0], st.s[1]],
		t: st.t + 1
	};
	for (let i = 0; i < 2; i++) {
		if (kind === 'sgd') {
			next.p[i] = st.p[i] - lr * g[i];
		} else if (kind === 'momentum') {
			next.v[i] = beta * st.v[i] + g[i];
			next.p[i] = st.p[i] - lr * next.v[i];
		} else {
			next.m[i] = B1 * st.m[i] + (1 - B1) * g[i];
			next.s[i] = B2 * st.s[i] + (1 - B2) * g[i] * g[i];
			const mh = next.m[i] / (1 - Math.pow(B1, next.t));
			const sh = next.s[i] / (1 - Math.pow(B2, next.t));
			next.p[i] = st.p[i] - (lr * mh) / (Math.sqrt(sh) + EPS);
		}
	}
	return next;
}

/** The 1D problem used by the first half of the lesson. */
export const curve = { f: (w: number) => w * w, grad: (w: number) => 2 * w, start: 1.8 };

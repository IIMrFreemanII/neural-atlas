import { tracks } from '../../content/curriculum.ts';
import { OrbitRig, type Pose } from '../../gpu/camera.ts';
import { mixColor, sheetFill, tone } from '../../gpu/colors.ts';
import { Lines, type SegmentStyle } from '../../gpu/primitives/lines.ts';
import { SurfaceMesh } from '../../gpu/primitives/mesh.ts';
import { Points, type PointStyle } from '../../gpu/primitives/points.ts';
import { Shadows } from '../../gpu/primitives/shadows.ts';
import { Sheets, type RGBA } from '../../gpu/primitives/sheets.ts';
import type { Vec3 } from '../../gpu/space.ts';
import type { Scene, SceneContext, Tone } from '../../gpu/types.ts';
import { createShader } from '../../gpu/util.ts';
import { projectToCss } from '../../gpu/view.ts';
import { SceneOrbit } from '../shared/orbit.ts';

export const regions = [
	{ id: 'fundamentals', label: 'Neural network fundamentals', tone: 'data', center: [-5.1, 0, -0.85] },
	{ id: 'classic-ml', label: 'Classic machine learning', tone: 'ok', center: [-5.9, 0, 4.35] },
	{ id: 'llm', label: 'Large language models', tone: 'param', center: [5.1, 0, -1.65] },
	{ id: 'vision', label: 'Vision models', tone: 'grad', center: [5.9, 0, 3.9] }
] as const satisfies readonly { id: string; label: string; tone: Tone; center: Readonly<Vec3> }[];

const leftOrbit: Vec3[] = [
	[-2.8, 1.14, -1.82], [-3.18, 0.62, -0.69], [-3.2, 0.32, 0.8], [-2.56, 0.06, 2.13]
];
const rightOrbit: Vec3[] = [[2.5, 0.07, 2.1], [3.15, 0.57, 0.22], [2.8, 1.11, -1.73]];

function orbitPoints(profile: Vec3[], count: number): Vec3[] {
	return Array.from({ length: count }, (_, i) => {
		const t = count === 1 ? (profile.length - 1) / 2 : i * (profile.length - 1) / (count - 1);
		const start = Math.floor(t);
		const end = Math.ceil(t);
		const blend = t - start;
		const a = profile[start];
		const b = profile[end];
		return [a[0] + (b[0] - a[0]) * blend, a[1] + (b[1] - a[1]) * blend, a[2] + (b[2] - a[2]) * blend];
	});
}

const lessonLayouts = regions.map(({ id, center }) => {
	const lessons = tracks.find((track) => track.id === id)?.lessons ?? [];
	const split = Math.ceil(lessons.length / 2);
	const offsets = [...orbitPoints(leftOrbit, split), ...orbitPoints(rightOrbit, lessons.length - split)];
	return { lessons, split, offsets, points: offsets.map((offset) => plus(center, offset)) };
});
const TAU = Math.PI * 2;
const FOV = 0.63;

export interface AtlasAnchor {
	id: string;
	kind: 'region' | 'lesson';
	region: (typeof regions)[number]['id'];
	x: number;
	y: number;
	label: string;
	tone: Tone;
}

function plus(a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 {
	return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function path(list: SegmentStyle[], points: Vec3[], style: Omit<SegmentStyle, 'a' | 'b' | 't0' | 't1'>) {
	for (let i = 1; i < points.length; i++) {
		list.push({ ...style, a: points[i - 1], b: points[i], t0: (i - 1) / (points.length - 1), t1: i / (points.length - 1) });
	}
}

function sample(count: number, value: (t: number) => Vec3): Vec3[] {
	return Array.from({ length: count + 1 }, (_, i) => value(i / count));
}

function ring(center: Readonly<Vec3>, radius: number, y: number, ratio = 0.82): Vec3[] {
	return sample(80, (t) => plus(center, [radius * Math.cos(TAU * t), y, radius * ratio * Math.sin(TAU * t)]));
}

/** Matte, depth-writing triangles for the beveled foundations. Colors carry the cut of each face. */
function foundations(ctx: SceneContext, vertices: number[]) {
	const { device } = ctx.gpu;
	const data = new Float32Array(vertices);
	const buffer = device.createBuffer({
		label: 'atlas sculpture foundations',
		size: data.byteLength,
		usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
	});
	device.queue.writeBuffer(buffer, 0, data);
	const module = createShader(device, 'atlas foundation shader', `
		@group(0) @binding(0) var<uniform> viewProj: mat4x4f;
		struct Vertex {
			@builtin(position) clip: vec4f,
			@location(0) color: vec3f,
		};
		@vertex fn vs(@location(0) position: vec3f, @location(1) color: vec3f) -> Vertex {
			return Vertex(viewProj * vec4f(position, 1.0), color);
		}
		@fragment fn fs(in: Vertex) -> @location(0) vec4f {
			return vec4f(in.color, 1.0);
		}
	`);
	const pipeline = device.createRenderPipeline({
		label: 'atlas foundation pipeline',
		layout: device.createPipelineLayout({ bindGroupLayouts: [ctx.view.bindGroupLayout] }),
		vertex: {
			module, entryPoint: 'vs', buffers: [{
				arrayStride: 24,
				attributes: [
					{ shaderLocation: 0, offset: 0, format: 'float32x3' },
					{ shaderLocation: 1, offset: 12, format: 'float32x3' }
				]
			}]
		},
		fragment: { module, entryPoint: 'fs', targets: [{ format: ctx.target.format }] },
		primitive: { topology: 'triangle-list' },
		depthStencil: { format: ctx.target.depthFormat, depthWriteEnabled: true, depthCompare: 'less' },
		multisample: { count: ctx.target.sampleCount }
	});
	return {
		draw(pass: GPURenderPassEncoder) {
			pass.setPipeline(pipeline);
			pass.setBindGroup(0, ctx.view.bindGroup);
			pass.setVertexBuffer(0, buffer);
			pass.draw(data.length / 6);
		},
		destroy() { buffer.destroy(); }
	};
}

function plinth(center: Readonly<Vec3>, tint: Tone, vertices: number[], lines: SegmentStyle[]) {
	const paper = sheetFill();
	const pigment = tone(tint);
	const profile = [
		{ radius: 2.08, y: 0.24 }, { radius: 2.19, y: 0.15 }, { radius: 2.19, y: -0.09 },
		{ radius: 2.13, y: -0.16 }, { radius: 2.0, y: -0.17 }
	].map(({ radius, y }) => ring(center, radius, y));
	const plate = mixColor(paper, pigment, 0.065);
	for (let i = 0; i < profile[0].length - 1; i++) {
		const a = profile[0][i];
		const b = profile[0][i + 1];
		for (const p of [plus(center, [0, 0.24, 0]), a, b]) vertices.push(...p, ...plate.slice(0, 3));
		for (let j = 1; j < profile.length; j++) {
			const row = profile[j];
			const previous = profile[j - 1];
			const face = 0.5 + 0.5 * Math.cos(i / 80 * TAU + 0.8);
			const color = mixColor(paper, pigment, [0, 0.13, 0.24, 0.37, 0.32][j] + face * 0.11);
			for (const p of [previous[i], row[i], previous[i + 1], previous[i + 1], row[i], row[i + 1]]) {
				vertices.push(...p, ...color.slice(0, 3));
			}
		}
	}
	for (const radius of [0.48, 0.94, 1.4, 1.82, 2.03]) {
		path(lines, ring(center, radius, 0.246), { width: radius > 2 ? 0.95 : 0.63, color: tone(tint, radius > 2 ? 0.34 : 0.14) });
	}
	for (let i = 0; i < 48; i++) {
		const a = i / 48 * TAU;
		const p = (r: number) => plus(center, [r * Math.cos(a), 0.25, r * 0.82 * Math.sin(a)]);
		path(lines, [p(i % 4 === 0 ? 1.88 : 1.95), p(2.025)], { width: 0.72, color: tone(tint, 0.42) });
	}
}

function point(points: PointStyle[], p: Vec3, radius: number, color: RGBA) {
	points.push({ x: p[0], y: p[1], z: p[2], radius, color });
}

function neuralNetwork(lines: SegmentStyle[], points: PointStyle[], reducedMotion: boolean) {
	const { center } = regions[0];
	const layers = [3, 4, 4, 2].map((n, layer) =>
		Array.from({ length: n }, (_, i): Vec3 => {
			const level = i - (n - 1) / 2;
			return plus(center, [
				-1.56 + layer * 1.02,
				1.41 + level * 0.53 + 0.1 * Math.cos(i * 2 + layer),
				0.3 * Math.sin(i * 2.1 + layer * 1.28) - level * 0.2
			]);
		})
	);
	layers.forEach((layer, l) => {
		layer.forEach((p, i) => {
			point(points, p, l === 3 ? 0.164 : 0.13 + 0.015 * (i % 3), mixColor(tone(l === 3 ? 'ok' : 'data'), sheetFill(), 0.2 + 0.05 * i));
			if (l === 3) return;
			layers[l + 1].forEach((next, j) => {
				const charge = (i * 9 + j * 3 + l) % 7;
				const bright = charge < 2;
				const arch = 0.12 * Math.sin(i * 2.6 + j * 1.2 + l);
				path(lines, sample(16, (t) => {
					const bend = Math.sin(t * Math.PI);
					const ease = t * t * (3 - 2 * t);
					return [
						p[0] + (next[0] - p[0]) * t,
						p[1] + (next[1] - p[1]) * ease + arch * bend,
						p[2] + (next[2] - p[2]) * ease + (arch * 1.5 + 0.12) * bend
					];
				}), {
					width: bright ? 1.34 : 0.83,
					color: tone(charge === 0 ? 'param' : 'data', bright ? 0.65 : 0.35),
					pulseSpeed: reducedMotion || charge !== 0 ? 0 : 0.26 + 0.02 * i,
					pulseCount: reducedMotion || charge !== 0 ? 0 : 1,
					pulseGain: reducedMotion || charge !== 0 ? 0 : 0.45
				});
			});
		});
		const x = layer[0][0];
		path(lines, [[x, 0.25, center[2]], [x, layer[0][1], layer[0][2]]], { width: 0.9, color: tone('muted', 0.32) });
		path(lines, sample(36, (t) => [x, 1.42 + Math.cos(t * TAU) * 1.02, center[2] + Math.sin(t * TAU) * 0.58]), {
			width: 0.6, color: tone('data', 0.2)
		});
	});
}

function decisionSurface(ctx: SceneContext, lines: SegmentStyle[], points: PointStyle[]) {
	const { center } = regions[1];
	const mesh = new SurfaceMesh(ctx, { cols: 49, rows: 49, style: 'ink' });
	const extent = 1.38;
	const height = (x: number, z: number) =>
		0.52 + 0.87 / (1 + Math.exp(-4.2 * (x - 0.58 * Math.sin(2.4 * z)))) + 0.17 * Math.sin(2.6 * z + 1.5 * x);
	mesh.rect = [center[0] - extent, center[2] - extent, center[0] + extent, center[2] + extent];
	mesh.range = [0.34, 1.59];
	mesh.bands = 10;
	mesh.setInk({
		fill: mixColor(sheetFill(), tone('ok'), 0.18),
		ink: tone('ok', 0.56),
		contourInk: tone('ok', 0.53),
		gridEvery: 4,
		hatch: false
	});
	for (let row = 0; row < mesh.rows; row++) {
		for (let col = 0; col < mesh.cols; col++) {
			mesh.heights[row * mesh.cols + col] = height((2 * col / (mesh.cols - 1) - 1) * extent, (2 * row / (mesh.rows - 1) - 1) * extent);
		}
	}
	mesh.upload();
	const rim = (x: number, z: number): Vec3 => plus(center, [x, height(x, z) + 0.002, z]);
	for (const sign of [-1, 1]) {
		path(lines, sample(36, (t) => rim(extent * (2 * t - 1), sign * extent)), { width: 1.3, color: tone('ok', 0.7) });
		path(lines, sample(36, (t) => rim(sign * extent, extent * (2 * t - 1))), { width: 1.3, color: tone('ok', 0.7) });
	}
	path(lines, sample(56, (t) => {
		const z = extent * (2 * t - 1);
		const x = 0.58 * Math.sin(2.4 * z);
		return plus(center, [x, height(x, z) + 0.021, z]);
	}), { width: 2, color: tone('ok', 0.91) });
	for (let i = 0; i < 24; i++) {
		const z = ((i * 13) % 23 / 23 - 0.5) * 2.32;
		const x = (i % 2 ? 1 : -1) * (0.46 + ((i * 19) % 17) / 24);
		const y = height(x, z);
		const lift = 0.11 + (i % 4) * 0.028;
		const top = plus(center, [x, y + lift, z]);
		path(lines, [plus(center, [x, y, z]), top], { width: 0.74, color: tone(i % 2 ? 'ok' : 'grad', 0.35) });
		point(points, top, 0.047 + 0.004 * (i % 3), tone(i % 2 ? 'ok' : 'grad'));
	}
	return mesh;
}

function tileStack(ctx: SceneContext, lines: SegmentStyle[], vision: boolean) {
	const count = vision ? 3 : 4;
	const sheets = new Sheets(ctx, { capacity: count });
	const { center, tone: tint } = regions[vision ? 3 : 2];
	for (let level = 0; level < count; level++) {
		const n = vision ? 7 + level * 2 : 8;
		const dimension = 2.71 - level * 0.115;
		const yaw = vision ? -0.12 - level * 0.1 : 0.045 * (level - 1);
		const x0 = -dimension / 2 + (vision ? -0.04 : -0.12) * level;
		const z0 = -dimension / 2 + (vision ? -0.03 : -0.05) * level;
		const y = 0.51 + level * (vision ? 0.52 : 0.46);
		const rotate = (x: number, z: number): Vec3 => plus(center, [
			x * Math.cos(yaw) + z * Math.sin(yaw), y, -x * Math.sin(yaw) + z * Math.cos(yaw)
		]);
		const origin = rotate(x0, z0);
		const u: Vec3 = [dimension * Math.cos(yaw), 0, -dimension * Math.sin(yaw)];
		const v: Vec3 = [dimension * Math.sin(yaw), 0, dimension * Math.cos(yaw)];
		const tex = sheets.createTexture(n * 5, n * 5);
		tex.write((i, j) => {
			const x = Math.floor(i / 5);
			const z = Math.floor(j / 5);
			const noise = (((x + 3) * 103 + (z + 5) * 79 + (level + 2) * 73) ** 2 % 137) / 137;
			const line = i % 5 === 0 || j % 5 === 0;
			if (line) return mixColor(sheetFill(), tone(tint), 0.17);
			if (!vision) {
				const active = x <= z;
				const score = x === z ? 0.85 : 0.13 + 0.53 * noise;
				return mixColor(sheetFill(), tone((x + level) % 4 === 0 ? 'data' : tint), active ? score : 0.065);
			}
			const px = (x / (n - 1) - 0.5) * 2;
			const pz = (z / (n - 1) - 0.5) * 2;
			const distance = Math.hypot(px * 0.95, pz + 0.24);
			const horizon = 0.36 + 0.28 * Math.sin(2.6 * px);
			const motif = distance < 0.44 ? 0.89 : pz > horizon ? 0.65 : 0.16;
			const order = 0.13 + level * 0.41;
			return mixColor(sheetFill(), tone(pz > horizon ? 'data' : tint), 0.09 + 0.82 * (order * motif + (1 - order) * noise));
		});
		sheets.set(sheets.count++, {
			origin, u, v, texture: tex, fill: sheetFill(), edge: tone(tint, 0.82), edgePx: 1.18
		});
		if (level < count - 1) {
			for (const x of [x0 + 0.14, x0 + dimension - 0.14]) {
				const p = rotate(x, z0 + dimension - 0.14);
				path(lines, [p, plus(p, [-0.11, vision ? 0.51 : 0.45, -0.04])], { width: 1.17, color: tone(tint, 0.47) });
			}
		}
	}
	return sheets;
}

function lineLayer(ctx: SceneContext, strokes: SegmentStyle[]) {
	const lines = new Lines(ctx, { capacity: strokes.length, depthTest: true });
	strokes.forEach((s, i) => lines.set(i, s));
	lines.count = strokes.length;
	lines.upload();
	return lines;
}

function atlasRoutes(ctx: SceneContext) {
	const strokes: SegmentStyle[] = [];
	for (const [a, b] of [[0, 1], [0, 2], [1, 3], [2, 3]]) {
		const start = regions[a].center;
		const end = regions[b].center;
		const route = sample(64, (t) => [
			start[0] + (end[0] - start[0]) * t,
			-0.095,
			start[2] + (end[2] - start[2]) * t + 0.24 * Math.sin(TAU * t)
		]);
		for (let i = 1; i < route.length; i += 2) {
			path(strokes, [route[i - 1], route[i]], { width: 0.92, color: tone('muted', 0.28) });
		}
	}
	return lineLayer(ctx, strokes);
}

function sculpture(ctx: SceneContext, index: number) {
	const vertices: number[] = [];
	const strokes: SegmentStyle[] = [];
	const dots: PointStyle[] = [];
	const { center, tone: tint } = regions[index];
	plinth(center, tint, vertices, strokes);
	const casts = new Shadows(ctx, { capacity: 1 });
	casts.blob(0, { center: plus(center, [0.22, -0.19, 0.13]), radius: 2.42, squash: 0.79, alpha: 0.42 });
	casts.count = 1;

	if (index === 0) neuralNetwork(strokes, dots, ctx.reducedMotion);
	const surface = index === 1 ? decisionSurface(ctx, strokes, dots) : undefined;
	const sheets = index >= 2 ? tileStack(ctx, strokes, index === 3) : undefined;
	const bases = foundations(ctx, vertices);
	const lines = lineLayer(ctx, strokes);
	const spheres = dots.length > 0
		? new Points(ctx, { capacity: dots.length, shade: 'sphere', size: 'world', depthTest: true })
		: undefined;
	if (spheres) {
		dots.forEach((p, i) => spheres.set(i, p));
		spheres.count = dots.length;
		spheres.upload();
	}
	return {
		draw(pass: GPURenderPassEncoder) {
			casts.draw(pass);
			bases.draw(pass);
			surface?.draw(pass);
			sheets?.draw(pass);
			lines.draw(pass);
			spheres?.draw(pass);
		},
		dispose() {
			casts.dispose();
			bases.destroy();
			surface?.destroy();
			sheets?.dispose();
			lines.destroy();
			spheres?.destroy();
		}
	};
}

function satellites(ctx: SceneContext, regionIndex: number) {
	const { points, split } = lessonLayouts[regionIndex];
	const tint = regions[regionIndex].tone;
	const strokes: SegmentStyle[] = [];
	for (let i = 1; i < points.length; i++) {
		const a = points[i - 1];
		const b = points[i];
		if (i === split) {
			path(strokes, sample(36, (t) => [
				a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
				a[2] + (b[2] - a[2]) * t + 0.66 * Math.sin(t * Math.PI)
			]), { width: 1.1, color: tone(tint, 0.6) });
		} else path(strokes, [a, b], { width: 1.1, color: tone(tint, 0.6) });
	}
	const lines = lineLayer(ctx, strokes);
	const nodes = new Points(ctx, { capacity: points.length, shade: 'sphere', size: 'world', depthTest: true });
	return {
		set(index: number, completed: number) {
			points.forEach((p, i) => nodes.set(i, {
				x: p[0], y: p[1], z: p[2], radius: i === index ? 0.15 : completed & (1 << i) ? 0.12 : 0.095,
				color: tone(i === index ? 'text' : completed & (1 << i) ? 'ok' : tint)
			}));
			nodes.count = points.length;
			nodes.upload();
		},
		draw(pass: GPURenderPassEncoder) {
			lines.draw(pass);
			nodes.draw(pass);
		},
		dispose() {
			lines.destroy();
			nodes.destroy();
		}
	};
}

const regionAnchors = regions.map(({ center }) => plus(center, [0, -0.13, 1.96]));
const bounds = regions.map(({ center }) => [
	...ring(center, 2.19, -0.17), ...ring(center, 2.19, 0.24), ...ring(center, 1.75, 2.34, 0.67)
]);
const layoutRig = new OrbitRig({ fovY: FOV });

function fitPose(ctx: SceneContext, region: number, lesson: number): Pose {
	const view = ctx.view;
	const overview = region === 0;
	const fundamentals = region === 1;
	const wide = view.width > 630;
	const layout = lessonLayouts[region - 1];
	const selected = layout?.offsets[lesson];
	const target: Vec3 = overview
		? [0.02, 0.16, 1.42]
		: plus(regions[region - 1].center, [0, fundamentals ? 0.91 : 0.89, 0.34]);
	if (selected) {
		target[0] += 0.095 * selected[0];
		target[1] += 0.075 * (selected[1] - 0.62);
		target[2] += 0.079 * selected[2];
	}
	const pose: Pose = {
		target,
		azimuth: overview ? (wide ? 0.19 : 0.13) : (fundamentals ? 0.16 : 0.25) + 0.014 * (selected?.[0] ?? 0),
		elevation: overview ? (wide ? 0.54 : 0.82) : 0.64,
		distance: overview ? 15 : 6
	};
	const shape = overview ? bounds.flat() : bounds[region - 1];
	const markers = overview ? regionAnchors : layout.points;
	const left = view.width * (overview ? 0.06 : 0.055);
	const right = view.width - left;
	const top = 52;
	const bottom = Math.max(top, view.height - 44);
	const labelMargin = overview ? Math.min(85, view.width * 0.18) : Math.min(81, view.width * 0.14);
	for (let attempt = 0; attempt < 46; attempt++) {
		layoutRig.setPose(pose);
		const matrix = layoutRig.matrix(view.aspect);
		const inside = (point: Vec3, padding: number, down: number) => {
			const [x, y] = projectToCss(matrix, view.width, view.height, ...point);
			const w = matrix[3] * point[0] + matrix[7] * point[1] + matrix[11] * point[2] + matrix[15];
			return w > 0 && x > left + padding && x < right - padding && y > top && y < bottom - down;
		};
		if (shape.every((p) => inside(p, 0, 0)) && markers.every((p) => inside(p, labelMargin * 0.61, 17))) break;
		pose.distance *= 1.055;
	}
	// Keep clearance for all lesson labels while moving into the selected chapter.
	if (!overview && lesson < 0) pose.distance *= 1.048;
	return pose;
}

export function createAtlasScene(onanchors: (anchors: AtlasAnchor[]) => void, reducedMotion: boolean): Scene {
	let ctx: SceneContext;
	const sculptures: ReturnType<typeof sculpture>[] = [];
	let routes: Lines | undefined;
	let travelRemaining = 0;
	const lessonNodes: ReturnType<typeof satellites>[] = [];
	let orbit: SceneOrbit;
	let region = 0;
	let lesson = -1;
	let completed = 0;
	let poseChanged = true;
	let previousWidth = 0;
	let previousHeight = 0;
	let anchors: AtlasAnchor[] = [];
	const rig = new OrbitRig({ fovY: FOV, maxDrag: 0.28, minElevation: 0.48, maxElevation: 0.95, dragSpeed: 0.004 });
	const constrain = (v: number, min: number, max: number) => Math.max(min, Math.min(v, max));

	function publish() {
		const { view } = ctx;
		const width = view.width;
		const compact = width < 630;
		const layout = lessonLayouts[region - 1];
		const focused = regions[region - 1];
		const projected = focused
			? layout.lessons.map(({ title }, i): AtlasAnchor => {
				const [px, py] = view.project(...layout.points[i]);
				const left = i < layout.split;
				const horizontal = compact ? 66 : 89;
				return {
					id: String(i), kind: 'lesson', region: focused.id, label: title,
					tone: completed & (1 << i) ? 'ok' : focused.tone,
					x: constrain(
						px + (left ? -12 : 12),
						left ? horizontal : width / 2 + horizontal,
						left ? width / 2 - horizontal : width - horizontal
					),
					y: constrain(py + 18, 51, view.height - 52)
				};
			})
			: regions.map(({ id, label, tone: tint }, i): AtlasAnchor => {
				const [px, py] = view.project(...regionAnchors[i]);
				const horizontal = compact ? 63 : 95;
				return {
					id, kind: 'region', region: id, label, tone: tint,
					x: constrain(px, horizontal, width - horizontal),
					y: constrain(py + 15, 58, view.height - 55)
				};
			});

		// Preserve the spatial order when orbiting brings two captions closer than their hit areas.
		if (focused) {
			const spacing = compact ? 56 : 47;
			for (const side of [projected.slice(0, layout.split), projected.slice(layout.split).reverse()]) {
				if (!side.length) continue;
				for (let i = 1; i < side.length; i++) side[i].y = Math.max(side[i].y, side[i - 1].y + spacing);
				const overflow = side[side.length - 1].y - (view.height - 53);
				if (overflow > 0) side.forEach((anchor) => { anchor.y -= overflow; });
			}
		} else if (compact) {
			for (const [top, bottom] of [[0, 1], [2, 3]]) {
				projected[bottom].y = Math.max(projected[bottom].y, projected[top].y + 55);
			}
		}
		if (
			anchors.length === projected.length &&
			projected.every((item, i) =>
				item.id === anchors[i].id && item.kind === anchors[i].kind && item.region === anchors[i].region &&
				item.tone === anchors[i].tone &&
				Math.abs(item.x - anchors[i].x) < 0.25 && Math.abs(item.y - anchors[i].y) < 0.25
			)
		) return;
		anchors = projected;
		onanchors(anchors);
	}

	return {
		init(context) {
			ctx = { ...context, reducedMotion };
			orbit = new SceneOrbit(ctx, rig);
			routes = atlasRoutes(ctx);
			regions.forEach((_, i) => {
				sculptures.push(sculpture(ctx, i));
				lessonNodes.push(satellites(ctx, i));
			});
		},
		setParams(params) {
			const nextRegion = params.region ?? 0;
			const nextLesson = params.lesson ?? -1;
			const nextCompleted = params.completed ?? 0;
			if (region === nextRegion && lesson === nextLesson && completed === nextCompleted) return;
			if (region !== nextRegion && previousWidth > 0) travelRemaining = reducedMotion ? 0 : 0.8;
			if (region !== nextRegion || lesson !== nextLesson) poseChanged = true;
			region = nextRegion;
			lesson = nextLesson;
			completed = nextCompleted;
			lessonNodes[region - 1]?.set(lesson, completed);
		},
		update({ dt }) {
			if (poseChanged || ctx.view.width !== previousWidth || ctx.view.height !== previousHeight) {
				const pose = fitPose(ctx, region, lesson);
				if (poseChanged) orbit.stepPose(pose);
				else rig.glideTo(pose, { instant: reducedMotion, duration: 0.28 });
				poseChanged = false;
				previousWidth = ctx.view.width;
				previousHeight = ctx.view.height;
			}
			orbit.update(dt);
			travelRemaining = Math.max(0, travelRemaining - dt);
			ctx.view.setViewProj(rig.matrix(ctx.view.aspect));
			publish();
		},
		draw(pass) {
			if (region === 0 || travelRemaining > 0) {
				routes?.draw(pass);
				sculptures.forEach((part) => part.draw(pass));
			} else {
				sculptures[region - 1]?.draw(pass);
			}
			lessonNodes[region - 1]?.draw(pass);
		},
		pointer(event) { orbit.pointer(event); },
		command(id) {
			if (id === 'view:cancel-drag') orbit.pointer({ type: 'leave', x: 0, y: 0 });
			else orbit.command(id);
		},
		dispose() {
			routes?.destroy();
			sculptures.forEach((part) => part.dispose());
			lessonNodes.forEach((part) => part.dispose());
		}
	};
}

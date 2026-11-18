/**
 * Shared contracts between the WebGPU engine, the Stage host component and lesson scenes.
 * This file is types only. Do not add runtime code here.
 */

/** The shared WebGPU device for the whole app. */
export interface Gpu {
	device: GPUDevice;
	/** Preferred canvas format from navigator.gpu.getPreferredCanvasFormat(). */
	format: GPUTextureFormat;
}

/** Formats every scene pipeline must target (the offscreen HDR scene pass). */
export interface TargetInfo {
	/** Color format of the scene pass ('rgba16float'). */
	format: GPUTextureFormat;
	/** Depth attachment format of the scene pass ('depth24plus'). */
	depthFormat: GPUTextureFormat;
	/** MSAA sample count of the scene pass (4). */
	sampleCount: number;
}

export interface FrameInfo {
	/** Seconds of animated time since the scene started. Does not advance while paused. */
	time: number;
	/** Seconds since the previous frame, clamped to 0.1. Zero while paused. */
	dt: number;
}

/** Semantic colors. Each maps to a CSS token `--c-<tone>` and a linear RGB value in gpu/colors.ts. */
export type Tone = 'text' | 'muted' | 'data' | 'param' | 'grad' | 'neg' | 'ok';

/**
 * Camera + viewport state shared by all primitives (bind group 0).
 * WGSL layout: struct ViewU { viewProj: mat4x4f, viewport: vec2f, dpr: f32, time: f32 }
 * where viewport is the target size in device pixels.
 */
export interface View {
	readonly bindGroupLayout: GPUBindGroupLayout;
	readonly bindGroup: GPUBindGroup;
	/** Stage width in CSS pixels. */
	readonly width: number;
	/** Stage height in CSS pixels. */
	readonly height: number;
	readonly dpr: number;
	/** width / height */
	readonly aspect: number;
	/** Set the column-major 4x4 view-projection matrix. Scenes call this from update(). */
	setViewProj(m: Float32Array): void;
	/** World position -> CSS pixels relative to the stage's top-left corner. */
	project(x: number, y: number, z?: number): [number, number];
	/** CSS pixels relative to the stage's top-left -> world position on the z = 0 plane. Valid for orthographic (2D) cameras. */
	unproject(cssX: number, cssY: number): [number, number];
}

/** A text label drawn as HTML on top of the canvas. Positions are CSS pixels relative to the stage's top-left. */
export interface Label {
	/** Stable key across frames. */
	id: string;
	x: number;
	y: number;
	text: string;
	/** Default 'text'. */
	tone?: Tone;
	/** Horizontal anchor. Default 'center'. Labels are always vertically centered on y. */
	align?: 'left' | 'center' | 'right';
	/** Use the monospace font (for numbers). Default false. */
	mono?: boolean;
	/** Font size in CSS px. Default 12. */
	size?: number;
	/** 0..1, default 1. */
	opacity?: number;
}

export interface LabelSink {
	/** Replace the full set of labels. Scenes call this from update(), every frame or only when labels change. */
	set(labels: Label[]): void;
}

export interface ScenePointer {
	type: 'move' | 'down' | 'up' | 'leave';
	/** CSS pixels relative to the stage's top-left. */
	x: number;
	y: number;
}

/** Everything a scene receives from the Stage. */
export interface SceneContext {
	gpu: Gpu;
	target: TargetInfo;
	view: View;
	labels: LabelSink;
	/**
	 * Send a named value to the UI (shown as a readout chip), e.g. emit('loss', 0.42).
	 * A key of the form 'param:<key>' with a number instead writes that lesson param, e.g. emit('param:w', 0.8),
	 * so its slider follows a value the scene is animating. The scene then receives it back through setParams.
	 */
	emit(key: string, value: number | string): void;
	/** True when the user prefers reduced motion. Scenes should then avoid continuous decorative animation. */
	reducedMotion: boolean;
}

/**
 * A lesson visual. Lifecycle, driven by the Stage:
 *   init(ctx) once -> setParams(initial) -> every frame: update(frame) then draw(pass) -> dispose() once.
 * setParams is called again whenever any param changes (step change tween, or a control).
 */
export interface Scene {
	init(ctx: SceneContext): void | Promise<void>;
	/** Receives the complete current param set, never a partial one. */
	setParams(params: Readonly<Record<string, number>>): void;
	/** Advance simulation, write instance buffers, call ctx.view.setViewProj, set labels, emit readouts. */
	update(frame: FrameInfo): void;
	/** Issue draw calls into the scene pass (HDR, MSAA). */
	draw(pass: GPURenderPassEncoder): void;
	/** A lesson 'action' control was pressed, e.g. 'run', 'step', 'reset'. */
	command?(id: string): void;
	/** The learner is hovering a term of this tone in the narration (null = none). Scenes may emphasize matching elements. */
	highlight?(tone: Tone | null): void;
	pointer?(e: ScenePointer): void;
	dispose(): void;
}

export type SceneFactory = () => Scene;

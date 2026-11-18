import type { OrbitRig, Pose } from '../../gpu/camera.ts';
import type { SceneContext, ScenePointer } from '../../gpu/types.ts';

/** A press that moves less than this many CSS px is a click, not a drag. */
const CLICK_SLOP = 4;

type OrbitCtx = Pick<SceneContext, 'emit' | 'reducedMotion'>;

/**
 * Wires an OrbitRig to the lesson UI: glides to each step's pose, turns the figure on drag, and talks to the
 * page through 'view:drag' (show the drag hint) and 'view:moved' (show the "Back to this step's view" chip).
 */
export class SceneOrbit {
	readonly rig: OrbitRig;
	private ctx: OrbitCtx;
	private draggable: boolean;
	private posed = false;
	private lastMoved = false;
	private pressed = false;
	private dragging = false;
	private lastX = 0;
	private lastY = 0;
	private downX = 0;
	private downY = 0;

	constructor(ctx: OrbitCtx, rig: OrbitRig, opts: { draggable?: boolean } = {}) {
		this.ctx = ctx;
		this.rig = rig;
		this.draggable = opts.draggable ?? true;
		ctx.emit('view:drag', this.draggable ? 1 : 0);
		ctx.emit('view:moved', 0);
	}

	/** Call when the step's pose changes. The first call jumps (and so does reduced motion); later calls glide. */
	stepPose(p: Pose): void {
		const instant = !this.posed || this.ctx.reducedMotion;
		this.posed = true;
		this.rig.glideTo(p, { instant });
	}

	setDraggable(b: boolean): void {
		this.draggable = b;
		if (!b) {
			this.pressed = false;
			this.dragging = false;
		}
		this.ctx.emit('view:drag', b ? 1 : 0);
	}

	/**
	 * Feed every pointer event. Returns 'drag' while the press is turning the figure (including the release that
	 * ends it), 'click' on the release of a press that moved less than 4 CSS px, and null otherwise.
	 */
	pointer(e: ScenePointer): 'drag' | 'click' | null {
		if (e.type === 'down') {
			this.pressed = true;
			this.dragging = false;
			this.lastX = this.downX = e.x;
			this.lastY = this.downY = e.y;
			return null;
		}
		if (!this.pressed) return null;
		if (e.type === 'move') {
			if (!this.dragging) {
				if (!this.draggable || Math.hypot(e.x - this.downX, e.y - this.downY) < CLICK_SLOP) return null;
				this.dragging = true;
			}
			// The first drag move carries the distance travelled since the press, so there is no jump.
			this.rig.drag(e.x - this.lastX, e.y - this.lastY);
			this.lastX = e.x;
			this.lastY = e.y;
			return 'drag';
		}
		// up or leave
		const wasDragging = this.dragging;
		const wasClick = !wasDragging && e.type === 'up' && Math.hypot(e.x - this.downX, e.y - this.downY) < CLICK_SLOP;
		this.pressed = false;
		this.dragging = false;
		if (wasDragging) return 'drag';
		return wasClick ? 'click' : null;
	}

	/** Handles 'view:reset'. Returns true if the command was one of ours. */
	command(id: string): boolean {
		if (id !== 'view:reset') return false;
		this.rig.reset(this.ctx.reducedMotion);
		return true;
	}

	/** Advance the glide; tells the page when the learner has turned away from (or back to) the step's view. */
	update(dt: number): void {
		this.rig.update(dt);
		if (this.rig.moved !== this.lastMoved) {
			this.lastMoved = this.rig.moved;
			this.ctx.emit('view:moved', this.lastMoved ? 1 : 0);
		}
	}
}

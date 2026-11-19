import{r as e}from"./BOlS45Hx.js";import{n as t}from"./Dy4eWgNh.js";var n=`// Particle cloud: one compute pass eases positions and colors, one render pass draws soft discs.

struct Particle {
	pos: vec2f,
	group: f32,
	seed: f32,
	color: vec4f,
};

struct SimU {
	rates: vec4f,    // dt, time, follow, colorFollow
	misc: vec4f,     // jitter, groupCount, count, teleportCount
	neutral: vec4f,
	centers: array<vec4f, 4>, // two centers per entry: (x0, y0, x1, y1)
	palette: array<vec4f, 8>,
};

@group(0) @binding(0) var<uniform> sim: SimU;
@group(0) @binding(1) var<storage, read> targets: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> state: array<Particle>;
@group(0) @binding(3) var<storage, read> teleport: array<vec2f>;

fn centerAt(i: u32) -> vec2f {
	let c = sim.centers[i / 2u];
	return select(c.xy, c.zw, (i & 1u) == 1u);
}

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) gid: vec3u) {
	let i = gid.x;
	if (f32(i) >= sim.misc.z) {
		return;
	}
	var p = state[i];
	let tgt = targets[i];
	if (f32(i) < sim.misc.w) {
		p.pos = teleport[i];
	}

	let dt = sim.rates.x;
	let time = sim.rates.y;
	let jitter = sim.misc.x;
	let drift = jitter * vec2f(sin(time * 0.7 + p.seed * 61.0), cos(time * 0.55 + p.seed * 97.0));
	let kp = 1.0 - exp(-sim.rates.z * dt);
	p.pos = p.pos + (tgt + drift - p.pos) * kp;

	let groups = u32(sim.misc.y + 0.5);
	var goal = sim.neutral;
	var g = -1.0;
	if (groups > 0u) {
		var best = 1e30;
		var bi = 0u;
		for (var c = 0u; c < groups; c = c + 1u) {
			let d = tgt - centerAt(c);
			let d2 = dot(d, d);
			if (d2 < best) {
				best = d2;
				bi = c;
			}
		}
		g = f32(bi);
		goal = sim.palette[bi];
	}
	p.group = g;
	let kc = 1.0 - exp(-sim.rates.w * dt);
	p.color = p.color + (goal - p.color) * kc;
	state[i] = p;
}

struct ViewU {
	viewProj: mat4x4f,
	viewport: vec2f,
	dpr: f32,
	time: f32,
};
@group(0) @binding(0) var<uniform> view: ViewU;

struct VsIn {
	@builtin(vertex_index) vi: u32,
	@location(0) posMisc: vec4f, // x, y, group, seed
	@location(1) color: vec4f,
};

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) local: vec2f,
	@location(1) color: vec4f,
	@location(2) @interpolate(flat) shape: vec2f, // radius dev px, softness
};

// radius (css px) and softness come from the render uniform.
struct RenderU {
	radius: f32,
	softness: f32,
	pad0: f32,
	pad1: f32,
};
@group(1) @binding(0) var<uniform> style: RenderU;

@vertex
fn vs(in: VsIn) -> VsOut {
	var corners = array<vec2f, 6>(
		vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
		vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
	);
	let corner = corners[in.vi];
	let radius = style.radius * view.dpr;
	let extent = radius + 1.0;
	let clip = view.viewProj * vec4f(in.posMisc.xy, 0.0, 1.0);
	let offsetNdc = corner * extent * 2.0 / view.viewport;

	var o: VsOut;
	o.pos = vec4f(clip.xy + offsetNdc * clip.w, clip.z, clip.w);
	o.local = corner * extent;
	o.color = in.color;
	o.shape = vec2f(radius, style.softness);
	return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	let d = length(in.local);
	let r = max(in.shape.x, 0.001);
	let crisp = clamp(r - d + 0.5, 0.0, 1.0);
	let x = clamp(d / r, 0.0, 1.0);
	let e4 = exp(-4.0);
	let glow = (exp(-4.0 * x * x) - e4) / (1.0 - e4);
	let cov = mix(crisp, glow, clamp(in.shape.y, 0.0, 1.0));
	let a = in.color.a * cov;
	return vec4f(in.color.rgb * a, a);
}
`,r=8,i=60,a=64,o=class{capacity;count;targets;centers=new Float32Array(16);groupCount=0;palette=new Float32Array(32);neutralColor=[.3,.35,.5,.5];radius=1.6;softness=0;follow=5;colorFollow=8;jitter=0;device;view;stateBuf;targetBuf;teleportBuf;simBuf;styleBuf;computePipeline;computeGroup;renderPipeline;styleGroup;simData=new Float32Array(i);styleData=new Float32Array(4);teleport;teleportCount=0;constructor(i,a){let o=i.gpu.device;this.device=o,this.view=i.view,this.capacity=a.capacity,this.count=a.capacity,this.targets=new Float32Array(a.capacity*2),this.teleport=new Float32Array(a.capacity*2);let s=e=>Math.max(16,e*4);this.stateBuf=o.createBuffer({label:`particles state`,size:s(a.capacity*r),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),this.targetBuf=o.createBuffer({label:`particles targets`,size:s(a.capacity*2),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.teleportBuf=o.createBuffer({label:`particles teleport`,size:s(a.capacity*2),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.simBuf=o.createBuffer({label:`particles sim uniforms`,size:240,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.styleBuf=o.createBuffer({label:`particles style uniforms`,size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});let c=new Float32Array(a.capacity*r),l=t(2654435761);for(let e=0;e<a.capacity;e++){let t=e*r;c[t+2]=-1,c[t+3]=l(),c[t+4]=this.neutralColor[0],c[t+5]=this.neutralColor[1],c[t+6]=this.neutralColor[2],c[t+7]=this.neutralColor[3]}o.queue.writeBuffer(this.stateBuf,0,c);let u=e(o,`particles shader`,n),d=o.createBindGroupLayout({label:`particles compute layout`,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:`uniform`}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:`storage`}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:`read-only-storage`}}]});this.computePipeline=o.createComputePipeline({label:`particles compute pipeline`,layout:o.createPipelineLayout({bindGroupLayouts:[d]}),compute:{module:u,entryPoint:`simulate`}}),this.computeGroup=o.createBindGroup({label:`particles compute group`,layout:d,entries:[{binding:0,resource:{buffer:this.simBuf}},{binding:1,resource:{buffer:this.targetBuf}},{binding:2,resource:{buffer:this.stateBuf}},{binding:3,resource:{buffer:this.teleportBuf}}]});let f=o.createBindGroupLayout({label:`particles style layout`,entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:`uniform`}}]});this.styleGroup=o.createBindGroup({layout:f,entries:[{binding:0,resource:{buffer:this.styleBuf}}]});let p=`one-minus-src-alpha`;this.renderPipeline=o.createRenderPipeline({label:`particles pipeline (alpha)`,layout:o.createPipelineLayout({bindGroupLayouts:[i.view.bindGroupLayout,f]}),vertex:{module:u,entryPoint:`vs`,buffers:[{arrayStride:32,stepMode:`instance`,attributes:[{shaderLocation:0,offset:0,format:`float32x4`},{shaderLocation:1,offset:16,format:`float32x4`}]}]},fragment:{module:u,entryPoint:`fs`,targets:[{format:i.target.format,blend:{color:{srcFactor:`one`,dstFactor:p,operation:`add`},alpha:{srcFactor:`one`,dstFactor:p,operation:`add`}}}]},primitive:{topology:`triangle-list`},depthStencil:{format:i.target.depthFormat,depthWriteEnabled:!1,depthCompare:`always`},multisample:{count:i.target.sampleCount}})}uploadTargets(){this.device.queue.writeBuffer(this.targetBuf,0,this.targets,0,this.capacity*2)}setPositions(e){let t=Math.min(this.capacity,Math.floor(e.length/2));this.teleport.set(e.subarray(0,t*2)),this.device.queue.writeBuffer(this.teleportBuf,0,this.teleport,0,t*2),this.teleportCount=t}step(e,t){if(e<=0&&this.teleportCount===0)return;let n=Math.min(this.count,this.capacity);if(n<=0)return;let r=this.simData;r[0]=Math.max(0,e),r[1]=t,r[2]=this.follow,r[3]=this.colorFollow,r[4]=this.jitter,r[5]=Math.max(0,Math.min(8,this.groupCount)),r[6]=n,r[7]=this.teleportCount,r.set(this.neutralColor,8),r.set(this.centers,12),r.set(this.palette,28),this.device.queue.writeBuffer(this.simBuf,0,r),this.teleportCount=0;let i=this.device.createCommandEncoder({label:`particles step`}),o=i.beginComputePass({label:`particles compute`});o.setPipeline(this.computePipeline),o.setBindGroup(0,this.computeGroup),o.dispatchWorkgroups(Math.ceil(n/a)),o.end(),this.device.queue.submit([i.finish()])}draw(e){let t=Math.min(this.count,this.capacity);t!==0&&(this.styleData[0]=this.radius,this.styleData[1]=this.softness,this.device.queue.writeBuffer(this.styleBuf,0,this.styleData),e.setPipeline(this.renderPipeline),e.setBindGroup(0,this.view.bindGroup),e.setBindGroup(1,this.styleGroup),e.setVertexBuffer(0,this.stateBuf),e.draw(6,t))}destroy(){this.stateBuf.destroy(),this.targetBuf.destroy(),this.teleportBuf.destroy(),this.simBuf.destroy(),this.styleBuf.destroy()}};export{o as t};
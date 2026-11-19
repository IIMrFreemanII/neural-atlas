import{r as e,t}from"./BOlS45Hx.js";import{i as n,p as r}from"./MfLy5J65.js";var i=`struct ViewU {
	viewProj: mat4x4f,
	viewport: vec2f,
	dpr: f32,
	time: f32,
};
@group(0) @binding(0) var<uniform> view: ViewU;

struct ShadowU {
	color: vec4f, // rgb linear, a = cast alpha
};
@group(1) @binding(0) var<uniform> sh: ShadowU;

struct VsIn {
	@builtin(vertex_index) vi: u32,
	// blob: center xyz, kind 0          quad: center xyz, kind 1
	@location(0) a: vec4f,
	// blob: dir x, dir z, semiMajor, semiMinor      quad: c0 x, c0 z, c1 x, c1 z
	@location(1) b: vec4f,
	// blob: unused                                  quad: c2 x, c2 z, c3 x, c3 z
	@location(2) c: vec4f,
	// alpha, blur, half extent of the drawn square, 0
	@location(3) d: vec4f,
};

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) xz: vec2f,
	@location(1) @interpolate(flat) a: vec4f,
	@location(2) @interpolate(flat) b: vec4f,
	@location(3) @interpolate(flat) c: vec4f,
	@location(4) @interpolate(flat) d: vec4f,
};

@vertex
fn vs(in: VsIn) -> VsOut {
	var corners = array<vec2f, 6>(
		vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
		vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
	);
	let k = corners[in.vi] * in.d.z;
	let w = vec3f(in.a.x + k.x, in.a.y, in.a.z + k.y);
	var o: VsOut;
	o.pos = view.viewProj * vec4f(w, 1.0);
	o.xz = w.xz;
	o.a = in.a;
	o.b = in.b;
	o.c = in.c;
	o.d = in.d;
	return o;
}

// Signed distance from p to the convex quad c0..c3 (ring order), negative inside.
fn edgeDist(p: vec2f, p0: vec2f, p1: vec2f, centroid: vec2f) -> f32 {
	let e = p1 - p0;
	var n = normalize(vec2f(e.y, -e.x));
	if (dot(n, centroid - p0) > 0.0) {
		n = -n;
	}
	return dot(p - p0, n);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	var cov = 0.0;
	if (in.a.w < 0.5) {
		// Blob: gaussian-ish falloff over an ellipse, zero at the edge.
		let rel = in.xz - in.a.xz;
		let dir = in.b.xy;
		let along = dot(rel, dir) / max(in.b.z, 1e-5);
		let across = dot(rel, vec2f(-dir.y, dir.x)) / max(in.b.w, 1e-5);
		let q2 = min(along * along + across * across, 1.0);
		let e3 = exp(-3.0);
		cov = (exp(-3.0 * q2) - e3) / (1.0 - e3);
	} else {
		let p0 = in.b.xy;
		let p1 = in.b.zw;
		let p2 = in.c.xy;
		let p3 = in.c.zw;
		let cen = (p0 + p1 + p2 + p3) * 0.25;
		let d = max(
			max(edgeDist(in.xz, p0, p1, cen), edgeDist(in.xz, p1, p2, cen)),
			max(edgeDist(in.xz, p2, p3, cen), edgeDist(in.xz, p3, p0, cen))
		);
		// Soft edge centred on the border, about one blur unit each side.
		let blur = max(in.d.y, 1e-4);
		let x = clamp(0.5 - d / (2.0 * blur), 0.0, 1.0);
		cov = x * x * (3.0 - 2.0 * x);
	}
	let a = sh.color.a * in.d.x * cov;
	return vec4f(sh.color.rgb * a, a);
}
`,a=16;function o(){let e=Math.hypot(r[0],r[2])||1;return[-r[0]/e,-r[2]/e]}function s(e){let t=e.reduce((e,t)=>e+t[0],0)/e.length,n=e.reduce((e,t)=>e+t[2],0)/e.length;return e.map(e=>[...e]).sort((e,r)=>Math.atan2(e[2]-n,e[0]-t)-Math.atan2(r[2]-n,r[0]-t))}var c=class{capacity;data;count=0;device;view;buffer;uniform;group;pipeline;dirty=!0;constructor(r,o){let s=r.gpu.device;this.device=s,this.view=r.view,this.capacity=o.capacity,this.data=new Float32Array(o.capacity*a),this.buffer=s.createBuffer({label:`shadows instances`,size:Math.max(16,this.data.byteLength),usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});let c=t(),[l,u,d]=n(c.cast);this.uniform=s.createBuffer({label:`shadows color`,size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),s.queue.writeBuffer(this.uniform,0,new Float32Array([l,u,d,c.castAlpha]));let f=s.createBindGroupLayout({label:`shadows bgl`,entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]});this.group=s.createBindGroup({label:`shadows group`,layout:f,entries:[{binding:0,resource:{buffer:this.uniform}}]});let p=e(s,`shadows shader`,i),m=`one-minus-src-alpha`;this.pipeline=s.createRenderPipeline({label:`shadows pipeline`,layout:s.createPipelineLayout({label:`shadows pipeline layout`,bindGroupLayouts:[r.view.bindGroupLayout,f]}),vertex:{module:p,entryPoint:`vs`,buffers:[{arrayStride:64,stepMode:`instance`,attributes:[0,1,2,3].map(e=>({shaderLocation:e,offset:e*16,format:`float32x4`}))}]},fragment:{module:p,entryPoint:`fs`,targets:[{format:r.target.format,blend:{color:{srcFactor:`one`,dstFactor:m,operation:`add`},alpha:{srcFactor:`one`,dstFactor:m,operation:`add`}}}]},primitive:{topology:`triangle-list`,cullMode:`none`},depthStencil:{format:r.target.depthFormat,depthWriteEnabled:!1,depthCompare:`less-equal`,depthBias:-4,depthBiasSlopeScale:-2},multisample:{count:r.target.sampleCount}})}blob(e,t){let n=Math.max(.05,Math.min(1,t.squash??r[1])),i=t.radius/n,[a,s]=o();this.write(e,[t.center[0],t.center[1],t.center[2],0,a,s,i,t.radius,0,0,0,0,t.alpha??1,0,i,0])}quad(e,t){let n=s(t.corners),r=(n[0][0]+n[1][0]+n[2][0]+n[3][0])/4,i=(n[0][1]+n[1][1]+n[2][1]+n[3][1])/4,a=(n[0][2]+n[1][2]+n[2][2]+n[3][2])/4,o=0;for(let e of n)o=Math.max(o,Math.abs(e[0]-r),Math.abs(e[2]-a));let c=Math.max(t.blur,1e-4);this.write(e,[r,i,a,1,n[0][0],n[0][2],n[1][0],n[1][2],n[2][0],n[2][2],n[3][0],n[3][2],t.alpha??1,c,o+2*c,0])}write(e,t){this.data.set(t,e*a),this.dirty=!0}upload(){this.dirty=!1;let e=Math.min(this.count,this.capacity)*a;e>0&&this.device.queue.writeBuffer(this.buffer,0,this.data,0,e)}draw(e){let t=Math.min(this.count,this.capacity);t!==0&&(this.dirty&&this.upload(),e.setPipeline(this.pipeline),e.setBindGroup(0,this.view.bindGroup),e.setBindGroup(1,this.group),e.setVertexBuffer(0,this.buffer),e.draw(6,t))}dispose(){this.buffer.destroy(),this.uniform.destroy()}},l=4,u=class{rig;ctx;draggable;posed=!1;lastMoved=!1;pressed=!1;dragging=!1;lastX=0;lastY=0;downX=0;downY=0;constructor(e,t,n={}){this.ctx=e,this.rig=t,this.draggable=n.draggable??!0,e.emit(`view:drag`,+!!this.draggable),e.emit(`view:moved`,0)}stepPose(e){let t=!this.posed||this.ctx.reducedMotion;this.posed=!0,this.rig.glideTo(e,{instant:t})}setDraggable(e){this.draggable=e,e||(this.pressed=!1,this.dragging=!1),this.ctx.emit(`view:drag`,+!!e)}pointer(e){if(e.type===`down`)return this.pressed=!0,this.dragging=!1,this.lastX=this.downX=e.x,this.lastY=this.downY=e.y,null;if(!this.pressed)return null;if(e.type===`move`){if(!this.dragging){if(!this.draggable||Math.hypot(e.x-this.downX,e.y-this.downY)<l)return null;this.dragging=!0}return this.rig.drag(e.x-this.lastX,e.y-this.lastY),this.lastX=e.x,this.lastY=e.y,`drag`}let t=this.dragging,n=!t&&e.type===`up`&&Math.hypot(e.x-this.downX,e.y-this.downY)<l;return this.pressed=!1,this.dragging=!1,t?`drag`:n?`click`:null}command(e){return e===`view:reset`&&(this.rig.reset(this.ctx.reducedMotion),!0)}update(e){this.rig.update(e),this.rig.moved!==this.lastMoved&&(this.lastMoved=this.rig.moved,this.ctx.emit(`view:moved`,+!!this.lastMoved))}};export{c as n,u as t};
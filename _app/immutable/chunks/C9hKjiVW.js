import{r as e}from"./BOlS45Hx.js";var t=`struct ViewU {
	viewProj: mat4x4f,
	viewport: vec2f,
	dpr: f32,
	time: f32,
};
@group(0) @binding(0) var<uniform> view: ViewU;

struct VsIn {
	@builtin(vertex_index) vi: u32,
	@location(0) posR: vec4f,   // x, y, z, radius (css px)
	@location(1) size: vec4f,   // w, h (world), 0, 0
	@location(2) color: vec4f,  // linear, straight alpha
};

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) local: vec2f,                      // device px from center
	@location(1) color: vec4f,
	@location(2) @interpolate(flat) shape: vec3f,   // half width px, half height px, radius px
};

@vertex
fn vs(in: VsIn) -> VsOut {
	var corners = array<vec2f, 6>(
		vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
		vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
	);
	let corner = corners[in.vi];
	let half_vp = view.viewport * 0.5;
	let c = in.posR.xyz;
	let hw = in.size.x * 0.5;
	let hh = in.size.y * 0.5;

	let cc = view.viewProj * vec4f(c, 1.0);
	let cx = view.viewProj * vec4f(c + vec3f(hw, 0.0, 0.0), 1.0);
	let cy = view.viewProj * vec4f(c + vec3f(0.0, hh, 0.0), 1.0);
	let pc = cc.xy / cc.w * half_vp;
	let vx = cx.xy / cx.w * half_vp - pc;
	let vy = cy.xy / cy.w * half_vp - pc;
	let lx = length(vx);
	let ly = length(vy);
	let dx = select(vec2f(1.0, 0.0), vx / max(lx, 1e-6), lx > 1e-6);
	let dy = select(vec2f(0.0, 1.0), vy / max(ly, 1e-6), ly > 1e-6);

	let pad = 1.0;
	let ex = lx + pad;
	let ey = ly + pad;
	let p = pc + dx * corner.x * ex + dy * corner.y * ey;

	var o: VsOut;
	o.pos = vec4f(p / half_vp * cc.w, cc.z, cc.w);
	o.local = vec2f(corner.x * ex, corner.y * ey);
	o.color = in.color;
	o.shape = vec3f(lx, ly, min(in.posR.w * view.dpr, min(lx, ly)));
	return o;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	let r = in.shape.z;
	let q = abs(in.local) - (in.shape.xy - vec2f(r));
	let d = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
	let cov = clamp(0.5 - d, 0.0, 1.0);
	let a = in.color.a * cov;
	return vec4f(in.color.rgb * a, a);
}
`,n=12,r=class{capacity;data;count=0;device;view;buffer;pipeline;constructor(r,i){let a=r.gpu.device;this.device=a,this.view=r.view,this.capacity=i.capacity,this.data=new Float32Array(i.capacity*n),this.buffer=a.createBuffer({label:`quads instances`,size:Math.max(16,this.data.byteLength),usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});let o=e(a,`quads shader`,t),s=`one-minus-src-alpha`;this.pipeline=a.createRenderPipeline({label:`quads pipeline (alpha)`,layout:a.createPipelineLayout({label:`quads pipeline layout`,bindGroupLayouts:[r.view.bindGroupLayout]}),vertex:{module:o,entryPoint:`vs`,buffers:[{arrayStride:48,stepMode:`instance`,attributes:[{shaderLocation:0,offset:0,format:`float32x4`},{shaderLocation:1,offset:16,format:`float32x4`},{shaderLocation:2,offset:32,format:`float32x4`}]}]},fragment:{module:o,entryPoint:`fs`,targets:[{format:r.target.format,blend:{color:{srcFactor:`one`,dstFactor:s,operation:`add`},alpha:{srcFactor:`one`,dstFactor:s,operation:`add`}}}]},primitive:{topology:`triangle-list`},depthStencil:{format:r.target.depthFormat,depthWriteEnabled:!1,depthCompare:`always`},multisample:{count:r.target.sampleCount}})}set(e,t){let r=this.data,i=e*n;r[i]=t.x,r[i+1]=t.y,r[i+2]=t.z??0,r[i+3]=t.radius??0,r[i+4]=t.w,r[i+5]=t.h,r[i+6]=0,r[i+7]=0,r[i+8]=t.color[0],r[i+9]=t.color[1],r[i+10]=t.color[2],r[i+11]=t.color[3]}upload(){let e=Math.min(this.count,this.capacity)*n;e>0&&this.device.queue.writeBuffer(this.buffer,0,this.data,0,e)}draw(e){let t=Math.min(this.count,this.capacity);t!==0&&(e.setPipeline(this.pipeline),e.setBindGroup(0,this.view.bindGroup),e.setVertexBuffer(0,this.buffer),e.draw(6,t))}destroy(){this.buffer.destroy()}};export{r as t};
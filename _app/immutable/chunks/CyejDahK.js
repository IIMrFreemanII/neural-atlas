import{r as e}from"./BOlS45Hx.js";var t=`struct ViewU {
	viewProj: mat4x4f,
	viewport: vec2f,
	dpr: f32,
	time: f32,
};
@group(0) @binding(0) var<uniform> view: ViewU;

struct VsIn {
	@builtin(vertex_index) vi: u32,
	@location(0) posR: vec4f,   // x, y, z, radius (css px, or world units for vs_world)
	@location(1) color: vec4f,  // linear, straight alpha
	@location(2) misc: vec4f,   // ring (css px), softness, 0, 0
};

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) local: vec2f,                 // device px from center
	@location(1) color: vec4f,
	@location(2) @interpolate(flat) shape: vec4f, // radius dev px, ring dev px, softness, 0
	@location(3) @interpolate(flat) lightV: vec3f, // direction to the light in view space (x right, y up, z toward the eye)
};

// Direction toward the light, world space. Keep in sync with LIGHT in gpu/space.ts and mesh.wgsl.
const LIGHT = vec3f(-0.5, 0.8, 0.35);

fn corners6(vi: u32) -> vec2f {
	var corners = array<vec2f, 6>(
		vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
		vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
	);
	return corners[vi];
}

// World directions of screen right / up, recovered from the viewProj rows (the view is a rotation).
fn lightInView() -> vec3f {
	let m = view.viewProj;
	let rx = normalize(vec3f(m[0][0], m[1][0], m[2][0]));
	let ry = normalize(vec3f(m[0][1], m[1][1], m[2][1]));
	let rz = cross(rx, ry);
	let l = normalize(LIGHT);
	return vec3f(dot(l, rx), dot(l, ry), dot(l, rz));
}

fn finishVs(in: VsIn, radius: f32) -> VsOut {
	let extent = radius + 1.0;
	let clip = view.viewProj * vec4f(in.posR.xyz, 1.0);
	let offsetNdc = corners6(in.vi) * extent * 2.0 / view.viewport;

	var o: VsOut;
	o.pos = vec4f(clip.xy + offsetNdc * clip.w, clip.z, clip.w);
	o.local = corners6(in.vi) * extent;
	o.color = in.color;
	o.shape = vec4f(radius, in.misc.x * view.dpr, in.misc.y, 0.0);
	o.lightV = lightInView();
	return o;
}

@vertex
fn vs(in: VsIn) -> VsOut {
	return finishVs(in, in.posR.w * view.dpr);
}

@vertex
fn vs_world(in: VsIn) -> VsOut {
	// Device px per world unit at this depth: |d clip.y / d world| is the projection's y scale (the view is a rotation).
	let m = view.viewProj;
	let yScale = length(vec3f(m[0][1], m[1][1], m[2][1]));
	let clip = m * vec4f(in.posR.xyz, 1.0);
	let pxPerUnit = yScale * view.viewport.y * 0.5 / max(clip.w, 1e-4);
	return finishVs(in, in.posR.w * pxPerUnit);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	let d = length(in.local);
	let r = max(in.shape.x, 0.001);
	let ringW = in.shape.y;
	let soft = clamp(in.shape.z, 0.0, 1.0);

	var crisp = clamp(r - d + 0.5, 0.0, 1.0);
	if (ringW > 0.0) {
		crisp = crisp * clamp(d - (r - ringW) + 0.5, 0.0, 1.0);
	}

	let x = clamp(d / r, 0.0, 1.0);
	let e4 = exp(-4.0);
	let glow = (exp(-4.0 * x * x) - e4) / (1.0 - e4);

	let cov = mix(crisp, glow, soft);
	let a = in.color.a * cov;
	return vec4f(in.color.rgb * a, a);
}

// Matte ball: Lambert from a fake sphere normal, ambient 0.35, thin darker ink rim, no specular.
@fragment
fn fs_sphere(in: VsOut) -> @location(0) vec4f {
	let d = length(in.local);
	let r = max(in.shape.x, 0.001);
	let cov = clamp(r - d + 0.5, 0.0, 1.0);

	let q = clamp(in.local / r, vec2f(-1.0), vec2f(1.0));
	let nz = sqrt(max(1.0 - dot(q, q), 0.0));
	let n = vec3f(q, nz);
	let lambert = max(dot(n, normalize(in.lightV)), 0.0);
	let lit = in.color.rgb * (0.35 + 0.65 * lambert);

	let rim = clamp(d - (r - view.dpr) + 0.5, 0.0, 1.0);
	let rgb = mix(lit, in.color.rgb * 0.55, rim);

	let a = in.color.a * cov;
	return vec4f(rgb * a, a);
}
`,n=12,r=class{capacity;data;count=0;device;view;buffer;pipeline;constructor(r,i){let a=r.gpu.device;this.device=a,this.view=r.view,this.capacity=i.capacity,this.data=new Float32Array(i.capacity*n),this.buffer=a.createBuffer({label:`points instances`,size:Math.max(16,this.data.byteLength),usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});let o=e(a,`points shader`,t),s=`one-minus-src-alpha`;this.pipeline=a.createRenderPipeline({label:`points pipeline (${i.shade??`flat`}, ${i.size??`px`})`,layout:a.createPipelineLayout({label:`points pipeline layout`,bindGroupLayouts:[r.view.bindGroupLayout]}),vertex:{module:o,entryPoint:i.size===`world`?`vs_world`:`vs`,buffers:[{arrayStride:48,stepMode:`instance`,attributes:[{shaderLocation:0,offset:0,format:`float32x4`},{shaderLocation:1,offset:16,format:`float32x4`},{shaderLocation:2,offset:32,format:`float32x4`}]}]},fragment:{module:o,entryPoint:i.shade===`sphere`?`fs_sphere`:`fs`,targets:[{format:r.target.format,blend:{color:{srcFactor:`one`,dstFactor:s,operation:`add`},alpha:{srcFactor:`one`,dstFactor:s,operation:`add`}}}]},primitive:{topology:`triangle-list`},depthStencil:{format:r.target.depthFormat,depthWriteEnabled:i.depthWrite??!1,depthCompare:i.depthTest?`less-equal`:`always`},multisample:{count:r.target.sampleCount}})}set(e,t){let r=this.data,i=e*n;r[i]=t.x,r[i+1]=t.y,r[i+2]=t.z??0,r[i+3]=t.radius,r[i+4]=t.color[0],r[i+5]=t.color[1],r[i+6]=t.color[2],r[i+7]=t.color[3],r[i+8]=t.ring??0,r[i+9]=t.softness??0,r[i+10]=0,r[i+11]=0}upload(){let e=Math.min(this.count,this.capacity)*n;e>0&&this.device.queue.writeBuffer(this.buffer,0,this.data,0,e)}draw(e){let t=Math.min(this.count,this.capacity);t!==0&&(e.setPipeline(this.pipeline),e.setBindGroup(0,this.view.bindGroup),e.setVertexBuffer(0,this.buffer),e.draw(6,t))}destroy(){this.buffer.destroy()}};export{r as t};
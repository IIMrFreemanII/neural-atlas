import{r as e}from"./BOlS45Hx.js";var t=`struct ViewU {
	viewProj: mat4x4f,
	viewport: vec2f,
	dpr: f32,
	time: f32,
};
@group(0) @binding(0) var<uniform> view: ViewU;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

struct VsIn {
	@builtin(vertex_index) vi: u32,
	@location(0) originEdge: vec4f, // origin xyz, edgePx (css)
	@location(1) uOpacity: vec4f,   // u xyz, opacity
	@location(2) v: vec4f,          // v xyz, 0
	@location(3) fill: vec4f,       // linear, straight alpha
	@location(4) edge: vec4f,       // linear, straight alpha
};

struct VsOut {
	@builtin(position) pos: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) fill: vec4f,
	@location(2) @interpolate(flat) edge: vec4f,
	@location(3) @interpolate(flat) misc: vec2f, // edge device px, opacity
};

@vertex
fn vs(in: VsIn) -> VsOut {
	var corners = array<vec2f, 6>(
		vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
		vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
	);
	let c = corners[in.vi];
	let w = in.originEdge.xyz + in.uOpacity.xyz * c.x + in.v.xyz * c.y;
	var o: VsOut;
	o.pos = view.viewProj * vec4f(w, 1.0);
	o.uv = c;
	o.fill = in.fill;
	o.edge = in.edge;
	o.misc = vec2f(in.originEdge.w * view.dpr, in.uOpacity.w);
	return o;
}

// Premultiplied "inkOver" of a straight-alpha color onto a premultiplied destination.
fn inkOver(dst: vec4f, rgb: vec3f, a: f32) -> vec4f {
	return vec4f(rgb * a + dst.rgb * (1.0 - a), a + dst.a * (1.0 - a));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	// Derivatives first, in uniform control flow.
	let gu = vec2f(dpdx(in.uv.x), dpdy(in.uv.x));
	let gv = vec2f(dpdx(in.uv.y), dpdy(in.uv.y));
	let t = textureSample(tex, samp, in.uv);

	var col = vec4f(in.fill.rgb * in.fill.a, in.fill.a);
	col = inkOver(col, t.rgb, t.a);

	// Distance inside the border, in device px, along each sheet axis.
	let pxU = min(in.uv.x, 1.0 - in.uv.x) / max(length(gu), 1e-6);
	let pxV = min(in.uv.y, 1.0 - in.uv.y) / max(length(gv), 1e-6);
	let d = min(pxU, pxV);
	let ecov = clamp(in.misc.x - d + 0.5, 0.0, 1.0) * select(0.0, 1.0, in.misc.x > 0.0);
	col = inkOver(col, in.edge.rgb, in.edge.a * ecov);

	return col * in.misc.y;
}
`,n=20;function r(e){let t=Math.max(0,Math.min(1,e));return t<=.0031308?t*12.92:1.055*t**(1/2.4)-.055}function i(e,t,n){let i=new Uint8Array(e*t*4),a=0;for(let o=0;o<t;o++)for(let t=0;t<e;t++){let e=n(t,o);i[a++]=Math.round(r(e[0])*255),i[a++]=Math.round(r(e[1])*255),i[a++]=Math.round(r(e[2])*255),i[a++]=Math.round(Math.max(0,Math.min(1,e[3]))*255)}return i}var a=class{width;height;bindGroup;texture;device;constructor(e,t,n,r,i){this.device=e,this.width=r,this.height=i,this.texture=e.createTexture({label:`sheet texture`,size:[r,i],format:`rgba8unorm-srgb`,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST}),this.bindGroup=e.createBindGroup({label:`sheet texture group`,layout:t,entries:[{binding:0,resource:this.texture.createView()},{binding:1,resource:n}]})}write(e){let t=i(this.width,this.height,e);this.device.queue.writeTexture({texture:this.texture},t,{bytesPerRow:this.width*4},[this.width,this.height])}destroy(){this.texture.destroy()}},o=class{capacity;data;count=0;device;view;buffer;pipeline;layout;sampler;blank;textures=[];slots;dirty=!0;constructor(r,i){let o=r.gpu.device;this.device=o,this.view=r.view,this.capacity=i.capacity,this.data=new Float32Array(i.capacity*n),this.slots=Array(i.capacity).fill(null),this.buffer=o.createBuffer({label:`sheets instances`,size:Math.max(16,this.data.byteLength),usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),this.layout=o.createBindGroupLayout({label:`sheets bgl`,entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`}},{binding:1,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`non-filtering`}}]}),this.sampler=o.createSampler({magFilter:`nearest`,minFilter:`nearest`}),this.blank=new a(o,this.layout,this.sampler,1,1),this.blank.write(()=>[0,0,0,0]);let s=e(o,`sheets shader`,t),c=`one-minus-src-alpha`;this.pipeline=o.createRenderPipeline({label:`sheets pipeline`,layout:o.createPipelineLayout({label:`sheets pipeline layout`,bindGroupLayouts:[r.view.bindGroupLayout,this.layout]}),vertex:{module:s,entryPoint:`vs`,buffers:[{arrayStride:80,stepMode:`instance`,attributes:[0,1,2,3,4].map(e=>({shaderLocation:e,offset:e*16,format:`float32x4`}))}]},fragment:{module:s,entryPoint:`fs`,targets:[{format:r.target.format,blend:{color:{srcFactor:`one`,dstFactor:c,operation:`add`},alpha:{srcFactor:`one`,dstFactor:c,operation:`add`}}}]},primitive:{topology:`triangle-list`,cullMode:`none`},depthStencil:{format:r.target.depthFormat,depthWriteEnabled:i.depthWrite??!0,depthCompare:`less-equal`},multisample:{count:r.target.sampleCount}})}createTexture(e,t){let n=new a(this.device,this.layout,this.sampler,e,t);return this.textures.push(n),n}set(e,t){let r=this.data,i=e*n;r[i]=t.origin[0],r[i+1]=t.origin[1],r[i+2]=t.origin[2],r[i+3]=t.edgePx??1,r[i+4]=t.u[0],r[i+5]=t.u[1],r[i+6]=t.u[2],r[i+7]=t.opacity??1,r[i+8]=t.v[0],r[i+9]=t.v[1],r[i+10]=t.v[2],r[i+11]=0,r.set(t.fill,i+12),r.set(t.edge,i+16),this.slots[e]=t.texture??null,this.dirty=!0}upload(){this.dirty=!1;let e=Math.min(this.count,this.capacity)*n;e>0&&this.device.queue.writeBuffer(this.buffer,0,this.data,0,e)}draw(e){let t=Math.min(this.count,this.capacity);if(t!==0){this.dirty&&this.upload(),e.setPipeline(this.pipeline),e.setBindGroup(0,this.view.bindGroup),e.setVertexBuffer(0,this.buffer);for(let n=0;n<t;n++)e.setBindGroup(1,(this.slots[n]??this.blank).bindGroup),e.draw(6,1,0,n)}}dispose(){this.buffer.destroy(),this.blank.destroy();for(let e of this.textures)e.destroy();this.textures=[]}};export{o as t};
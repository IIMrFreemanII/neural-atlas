/** Create a shader module and log any compile errors to the console. */
export function createShader(device: GPUDevice, label: string, code: string): GPUShaderModule {
	const module = device.createShaderModule({ label, code });
	void module.getCompilationInfo().then((info) => {
		for (const m of info.messages) {
			if (m.type === 'error') {
				console.error(`[gpu] ${label}`, `${m.lineNum}:${m.linePos} ${m.message}`);
			}
		}
	});
	return module;
}

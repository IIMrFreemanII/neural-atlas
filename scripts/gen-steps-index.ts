// Regenerates src/lib/content/steps-index.ts. Lesson modules use Vite-only imports (?url), so plain Node
// cannot load them; this runs the index spec under vitest in write mode instead.
// Run: node --experimental-strip-types scripts/gen-steps-index.ts
import { spawnSync } from 'node:child_process';

const r = spawnSync('npx', ['vitest', '--run', 'src/lib/content/steps-index.spec.ts'], {
	stdio: 'inherit',
	env: { ...process.env, GEN_STEPS_INDEX: '1' }
});
process.exit(r.status ?? 1);

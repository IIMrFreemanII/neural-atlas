# Neural Atlas

A desktop web app that teaches how AI, machine learning and language models work through interactive
visuals. Every lesson is a live WebGPU scene you can poke at, with short narration beside it and the math
and code behind an optional "Go deeper" panel.

"Neural Atlas" is a working title.

## What is in it

Nineteen lessons in four tracks:

- **Neural net fundamentals**: the neuron, activation functions, forward pass, loss landscape, gradient
  descent, backpropagation, train a network.
- **Classic machine learning**: linear regression, decision boundaries, k-means clustering, overfitting and
  underfitting.
- **LLMs and transformers**: tokenization, embedding space, self-attention, inside a transformer block,
  next-token sampling.
- **Vision and generative**: convolution, CNN feature maps, diffusion.

Plus a constellation map of the lessons on the home page and a glossary of the terms they use.

The models are small on purpose so the mechanism stays visible, and the math on screen is really computed in
the browser. Each lesson says what is simplified about its model.

## Requirements

- Node 24 (pinned with Volta in `package.json`).
- A browser with WebGPU: a current Chrome, Edge or Safari. There is no WebGL fallback; without WebGPU the
  visuals show a notice instead.
- A desktop-sized window. The layout does not adapt below 1024 px wide.

## Commands

```sh
npm install      # once
npm run dev      # development server at http://localhost:5173
npm run check    # type check (svelte-check)
npm run test     # unit tests (Vitest)
npm run build    # static site in build/
npm run preview  # serve the built site
```

`scripts/train-shapes-cnn.ts` retrains the small network used by the CNN feature maps lesson and rewrites its
weights file (about 40 seconds, deterministic):

```sh
node --experimental-strip-types scripts/train-shapes-cnn.ts
```

## How it is built

- **SvelteKit 3 + Svelte 5 + TypeScript**, built as a static, client-rendered site (`adapter-static`, no
  backend).
- **Rendering**: raw WebGPU and WGSL through a small in-house engine in `src/lib/gpu` (points, lines, quads,
  meshes, scalar fields, compute-shader particles, an HDR pass with bloom). No 3D library.
- **Math**: `src/lib/ml` holds the models as plain TypeScript with no DOM or GPU code, each with a Vitest spec
  next to it.

```
src/
  app.css                       design tokens and base styles
  lib/
    gpu/                        the WebGPU engine
    ml/                         models and math, unit-tested
    lessons/<track>/<lesson>/   lesson.ts (steps, narration, controls) and scene.ts (the visual)
    lessons/types.ts            the lesson contract
    content/                    curriculum, constellation layout, glossary
    components/lesson/          lesson page: stage, narration, controls
    components/shell/           top bar, track list, constellation map
    state/                      progress and settings, kept in localStorage
  routes/                       home, /learn, /learn/<track>/<lesson>, /glossary
```

A lesson is a `LessonDef` (`src/lib/lessons/types.ts`): a list of steps, each with narration and the scene
parameters for that step, plus a scene factory. The scene implements `Scene` (`src/lib/gpu/types.ts`). To add a
lesson, create its folder and register it in `src/lib/content/curriculum.ts`, give it a position in
`src/lib/content/constellation.ts`, and add its terms to `src/lib/content/glossary.ts`.

## Using a lesson

- Left and right arrow keys change the step. The space bar presses the lesson's main action (Run, Train,
  Play).
- Hover a colored term in the narration to highlight the matching parts of the visual.
- **Reduce motion** in the top bar stops decorative animation and makes transitions instant. It starts from
  the operating system's setting.
- Progress is stored in the browser only.

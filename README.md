# Neural Atlas

A desktop web app that teaches how AI, machine learning and language models work through interactive
visuals. Every lesson is a live WebGPU scene you can poke at, with short narration beside it and the math
and code behind an optional "Go deeper" panel.

"Neural Atlas" is a working title.

## What is in it

Twenty lessons in four tracks:

- **Neural net fundamentals**: the neuron, activation functions, forward pass, loss landscape, gradient
  descent, backpropagation, train a network.
- **Classic machine learning**: linear regression, decision boundaries, k-means clustering, overfitting and
  underfitting.
- **LLMs and transformers**: tokenization, embedding space, self-attention, inside a transformer block,
  next-token sampling, watching a model learn.
- **Vision and generative**: convolution, CNN feature maps, diffusion.

Every lesson ends with three questions that check your understanding. Some ask you to pick an answer, and each
answer explains why it is right or wrong. Others are tasks you solve in the live visual, and the page notices
when you get there.

Plus a constellation map of the lessons on the home page and a glossary of the terms they use.

The models are small on purpose so the mechanism stays visible, and the math on screen is really computed in
the browser. Each lesson says what is simplified about its model.

Two things in the language-model track are real data, not toys:

- **A trained transformer.** The sampling, transformer block and self-attention lessons run a small GPT-style
  model (4 blocks, 4 heads each, 64 numbers per vector, 2,048-word vocabulary, about 330 thousand weights)
  that this repo trains itself on the TinyStories text. Its predictions and attention patterns on screen are
  the model's own. The last lesson of the track shows the same model at 11 saved moments of that training
  run, from random weights to the finished model.
- **GloVe word vectors.** The embedding space lesson places 74 words by their published GloVe vectors (100
  numbers per word).

The first eight steps of the self-attention lesson still use three hand-designed heads, because hand-set
numbers explain the mechanism best. The lesson says so, and its last step shows the trained model's heads.

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
npm run test:e2e # smoke test (Playwright): builds the site, then loads every page and lesson
npm run build    # static site in build/
npm run preview  # serve the built site
```

The smoke test drives the Chrome installed on your machine, headless, and fails on any console error. For
each lesson it checks that the visual starts, draws something, and survives a walk through every step. In a
browser without WebGPU it only checks that the "needs WebGPU" notice appears. To use Playwright's own
browser instead, run `npx playwright install chromium` once and set `PW_CHANNEL=chromium`.

## Regenerating the model data

The trained weights and vectors are committed, so none of this is needed to run the app. Each script is
deterministic: the same inputs give the same files.

`scripts/train-shapes-cnn.ts` retrains the small network used by the CNN feature maps lesson and rewrites its
weights file (about 40 seconds):

```sh
node --experimental-strip-types scripts/train-shapes-cnn.ts
```

`scripts/train-tinystories.ts` trains the transformer in plain TypeScript on 8 worker threads (about 40 minutes
on a 16-core machine) and writes `src/lib/lessons/llm/shared/tinystories.bin` and `lm-meta.ts`. It needs
`TinyStoriesV2-GPT4-valid.txt` (22.5 MB) from the
[TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories):

```sh
node --experimental-strip-types scripts/train-tinystories.ts path/to/TinyStoriesV2-GPT4-valid.txt --steps 12000 --checkpoint-dir path/to/checkpoints
```

The shipped model was trained with `--steps 12000`. `--checkpoint-dir` is optional and does not change the
result: it also saves the weights at 11 moments of the run (14 MB in all) and the loss of every step.
`scripts/build-training-log.ts` turns that folder into the data of the "Watching a model learn" lesson,
`src/lib/lessons/llm/training/log.ts` (the loss curve, and what the model writes and predicts at each
checkpoint). It refuses a run whose last checkpoint is not byte for byte the shipped `tinystories.bin`, so the
log always describes the model the other lessons run:

```sh
node --experimental-strip-types scripts/build-training-log.ts path/to/checkpoints
```

Four lessons describe what this particular model does (which word it predicts, what three of its attention
heads look at, what it writes early in training). Unit tests named "what the narration says about the trained weights" pin those facts, so if you
retrain with other settings and one fails, update that lesson's narration to match the new model.

`scripts/extract-glove.ts` copies the vectors of the embedding lesson's words out of `glove-wiki-gigaword-100.gz`
(134 MB, from the [gensim-data releases](https://github.com/piskvorky/gensim-data/releases)) into
`src/lib/lessons/llm/embeddings/vectors.ts`:

```sh
node --experimental-strip-types scripts/extract-glove.ts path/to/glove-wiki-gigaword-100.gz
```

## Data sources

- **TinyStories** (Eldan and Li, 2023): short stories in simple English written by GPT-3.5 and GPT-4. The
  dataset card lists its license as CDLA-Sharing-1.0. This repo contains weights trained on it and a vocabulary
  list, not the text.
- **GloVe** (Pennington, Socher and Manning, 2014), the 100-dimensional vectors trained on Wikipedia 2014 and
  Gigaword 5, released under the Public Domain Dedication and License. This repo contains the vectors of 74
  words, unchanged.

## How it is built

- **SvelteKit 3 + Svelte 5 + TypeScript**, built as a static, client-rendered site (`adapter-static`, no
  backend).
- **Rendering**: raw WebGPU and WGSL through a small in-house engine in `src/lib/gpu` (points, lines, quads,
  meshes, scalar fields, compute-shader particles, an HDR pass with bloom). No 3D library.
- **Math**: `src/lib/ml` holds the models as plain TypeScript with no DOM or GPU code, each with a Vitest spec
  next to it. `gpt.ts` is the transformer: forward pass, a hand-written backward pass (checked against
  numerical gradients) and the helpers the lessons draw from.

```
src/
  app.css                       design tokens and base styles
  lib/
    gpu/                        the WebGPU engine
    ml/                         models and math, unit-tested
    lessons/<track>/<lesson>/   lesson.ts (steps, narration, controls), scene.ts (the visual) and
                                questions.ts (the questions after the last step)
    lessons/types.ts            the lesson contract
    content/                    curriculum, constellation layout, glossary
    components/lesson/          lesson page: stage, narration, controls
    components/shell/           top bar, track list, constellation map
    state/                      progress and settings, kept in localStorage
  routes/                       home, /learn, /learn/<track>/<lesson>, /glossary
e2e/                            Playwright smoke test
```

A lesson is a `LessonDef` (`src/lib/lessons/types.ts`): a list of steps, each with narration and the scene
parameters for that step, plus a scene factory. The scene implements `Scene` (`src/lib/gpu/types.ts`). To add a
lesson, create its folder and register it in `src/lib/content/curriculum.ts`, give it a position in
`src/lib/content/constellation.ts`, and add its terms to `src/lib/content/glossary.ts`.

A lesson's questions are a list of `Question` objects (`src/lib/lessons/types.ts`). A task question has a
`goal` function over the lesson's params, readouts and typed text, and an `example` of control settings that
reach it. `src/lib/lessons/questions.spec.ts` checks every lesson's questions: the texts, that each task starts
unsolved, and that each example solves its task with controls that are on screen.

## Using a lesson

- Left and right arrow keys change the step. The space bar presses the lesson's main action (Run, Train,
  Play).
- Hover a colored term in the narration to highlight the matching parts of the visual.
- **Reduce motion** in the top bar stops decorative animation and makes transitions instant. It starts from
  the operating system's setting.
- Progress, including which questions you solved, is stored in the browser only.

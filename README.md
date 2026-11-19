# Neural Atlas

A desktop web app that teaches how AI, machine learning and language models work through interactive
visuals. Every lesson is a live WebGPU scene you can poke at, with short narration beside it and the math
and code behind an optional "Go deeper" panel.

Try it at <https://iimrfreemanii.github.io/neural-atlas/>.

"Neural Atlas" is a working title.

## What is in it

Twenty-three lessons in four tracks:

- **Neural net fundamentals**: the neuron, activation functions, forward pass, loss landscape, gradient
  descent, backpropagation, train a network.
- **Classic machine learning**: linear regression, decision boundaries, k-means clustering, overfitting and
  underfitting, decision trees and random forests.
- **LLMs and transformers**: tokenization, embedding space, self-attention, inside a transformer block,
  next-token sampling, watching a model learn.
- **Vision and generative**: convolution, CNN feature maps, vision transformer, latent space (an
  autoencoder), diffusion.

Every lesson ends with three questions that check your understanding. Some ask you to pick an answer, and each
answer explains why it is right or wrong. Others are tasks you solve in the live visual, and the page notices
when you get there. The lesson list and the map on the home page show how many questions of each lesson you
have solved.

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
npm run deploy   # publish the site to GitHub Pages
```

The smoke test drives the Chrome installed on your machine, headless, and fails on any console error. For
each lesson it checks that the visual starts, draws something, and survives a walk through every step. In a
browser without WebGPU it only checks that the "needs WebGPU" notice appears. To use Playwright's own
browser instead, run `npx playwright install chromium` once and set `PW_CHANNEL=chromium`.

## Publishing

The site is a static single page app on GitHub Pages. `npm run deploy` (`scripts/deploy-pages.sh`) builds
it and publishes the result to the `gh-pages` branch of `origin`, which GitHub serves. It refuses to run
with uncommitted changes, so that the published site always matches a commit.

- Pages serves a project at `/<repository>/`, so the script builds with `BASE_PATH` set to the repository's
  name. Internal links must therefore go through `resolve()` from `$app/paths`, or `lessonLink()` from
  `src/lib/link.ts`; a link written as `href="/learn"` leaves the site there.
- The build writes an HTML file for every page, each lesson included, so a link straight to a lesson is
  answered with status 200. The pages are empty shells that the app fills in the browser (there is no
  server rendering, since every visual needs WebGPU). Lessons are listed for the build by `entries()` in
  `src/routes/learn/[track]/[lesson]/+page.ts`. Any other address gets `404.html`, the app's own "not
  found" page.
- Link previews and search engines do not run the app, so `src/hooks.server.ts` writes each page's title
  and description (a lesson's title and summary from `curriculum.ts`) into its HTML while the build
  prerenders it. The texts are in `src/lib/page-meta.ts`.
- Each lesson's preview image is a screenshot of its scene in `static/previews/`. Previews need full
  addresses, so the script builds with `SITE_ORIGIN` (`https://<owner>.github.io`); a build without it has
  no preview images. After adding a lesson or changing how one looks, build and take the images again (it
  needs Chrome with WebGPU; a lesson key such as `llm/sampling` takes only that one):

  ```bash
  npm run build && node --experimental-strip-types scripts/capture-previews.ts
  ```

  `STEP` in that script picks the step shown, for lessons whose first step is a poor picture.
- Each deploy replaces the single commit on `gh-pages` with a forced push. The branch holds build output
  only.

## Regenerating the model data

The trained weights and vectors are committed, so none of this is needed to run the app. Each script is
deterministic, except the autoencoder's when it runs on a GPU: the same inputs give the same files.

`scripts/train-shapes-cnn.ts` retrains the small network used by the CNN feature maps lesson and rewrites its
weights file (about 40 seconds):

```sh
node --experimental-strip-types scripts/train-shapes-cnn.ts
```

`scripts/train-autoencoder.py` retrains the autoencoder of the latent space lesson with PyTorch and rewrites
`src/lib/lessons/vision/autoencoder/weights.ts`. It trains on the Mac's GPU (Metal) when there is one, and
on the CPU otherwise (`--cpu` forces it); on an Apple GPU the run takes about 15 seconds. It needs Python 3
and Node. The drawings come from the lesson's own generator through `scripts/export-shapes.ts`, and the
lesson runs the network in TypeScript (`src/lib/ml/autoencoder.ts`), so PyTorch is needed only for training.
With [uv](https://docs.astral.sh/uv/), nothing has to be installed into the project:

```sh
uv run --no-project --with torch python scripts/train-autoencoder.py
```

Unlike the other scripts, a GPU run is not repeatable bit for bit, so a rerun gives slightly different
weights. The tests in `src/lib/lessons/vision/autoencoder/` check that any new weights still separate the
four shapes on the map, rebuild drawings well, and agree with what the narration says.

`scripts/train-vit.py` trains the small vision transformer of the vision transformer lesson the same way
(PyTorch, about 90 seconds on an Apple GPU) on the CNN lesson's generated drawings, and rewrites
`src/lib/lessons/vision/vit/weights.ts`. The browser runs the model in TypeScript (`src/lib/ml/vit.ts`);
the weights file also holds PyTorch's answers for four drawings, and a test checks that the TypeScript
model gives the same ones. It trains with label smoothing, which is why the lesson's model never claims
more than about 92%:

```sh
uv run --no-project --with torch python scripts/train-vit.py
```

As with the autoencoder, a GPU rerun gives slightly different weights. `trained.spec.ts` in the lesson's
folder pins what the narration says about the model (how much it relies on position vectors, and which
head of the class token looks at the ink).

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
    components/home/            WebGPU atlas host, homepage and lesson previews
    components/shell/           top bar and track list
    state/                      progress and settings, kept in localStorage
  routes/                       home, /learn, /learn/<track>/<lesson>, /glossary
e2e/                            Playwright smoke test
```

A lesson is a `LessonDef` (`src/lib/lessons/types.ts`): a list of steps, each with narration and the scene
parameters for that step, plus a scene factory. A step can also list which controls, legend chips and
readouts it shows, so that a chip never names something that is not on screen yet. The scene implements `Scene` (`src/lib/gpu/types.ts`). To add a
lesson, create its folder and register it in `src/lib/content/curriculum.ts`, give it a position in
`src/lib/content/constellation.ts`, and add its terms to `src/lib/content/glossary.ts`.

A lesson's questions are a list of `Question` objects (`src/lib/lessons/types.ts`). A task question has a
`goal` function over the lesson's params, readouts and typed text, and an `example` of control settings that
reach it. `src/lib/lessons/questions.spec.ts` checks every lesson's questions: the texts, that each task starts
unsolved, and that each example solves its task with controls that are on screen.

## Exploring the atlas

The homepage prototype uses the same WebGPU engine as the lessons. Its overview shows four scientific
sculptures; click a region to move closer, or drag to turn the view. **Whole atlas** restores the overview
and **Reset view** restores the current camera. Paper and Night apply to both the scene and its controls.

Neural networks is the first complete guided region: all seven lesson markers open previews with the
lesson summary, prerequisites and a **Start lesson** link. **New to AI** starts with the neuron;
**I know the basics** starts with forward propagation. The other three regions show their sculptures and
link to existing lessons. The directory beneath the atlas keeps every lesson accessible, including when
WebGPU is unavailable. Reduced motion stops decorative signals and makes camera transitions immediate.

## Using a lesson

- Left and right arrow keys change the step. The space bar presses the lesson's main action (Run, Train,
  Play).
- Hover a colored term in the narration to highlight the matching parts of the visual.
- **Reduce motion** in the top bar stops decorative animation and makes transitions instant. It starts from
  the operating system's setting.
- Progress, including which questions you solved, is stored in the browser only.

## License

The code is under the MIT License; see `LICENSE`. The word vectors and the model trained on TinyStories
come from the sources named under "Data sources", and those sources' terms apply to them.

import type { Tone } from '../gpu/types.ts';

export interface GlossaryEntry {
	/** URL fragment, kebab-case, unique: /glossary#<id>. */
	id: string;
	/** As written in running text: lower case unless it is a name or an acronym ('ReLU', 'Adam'). */
	term: string;
	/** Other names or spellings the search should match, e.g. ['BPE']. */
	aliases?: string[];
	/** One or two plain sentences, at most 240 characters, ending with a period. No markup. */
	definition: string;
	/** The term's semantic color in the lessons, when it has one (weight: 'param', gradient: 'grad'). */
	tone?: Tone;
	/** Lesson keys ('<track>/<id>') where the term is taught or used, the lesson that introduces it first. */
	lessons: string[];
	/** Ids of related entries. */
	related?: string[];
}

const entries: GlossaryEntry[] = [
	{
		id: 'activation-function',
		term: 'activation function',
		aliases: ['activation'],
		definition:
			'The function a neuron applies to its sum before passing the result on. It adds a bend, which lets a network learn curves instead of only straight lines.',
		lessons: [
			'fundamentals/neuron',
			'fundamentals/activations',
			'fundamentals/forward-pass',
			'fundamentals/train-a-network',
			'vision/cnn-features'
		],
		related: ['relu', 'sigmoid', 'tanh']
	},
	{
		id: 'adam',
		term: 'Adam',
		definition:
			'An optimizer that keeps a running velocity like momentum and also scales each weight\'s step by how large its recent gradients were.',
		lessons: ['fundamentals/gradient-descent', 'llm/training'],
		related: ['optimizer', 'momentum']
	},
	{
		id: 'attention-head',
		term: 'attention head',
		aliases: ['head'],
		definition:
			'One copy of the attention mechanism with its own weights. A layer runs several side by side, and each head can look for a different kind of link between tokens.',
		lessons: ['llm/attention', 'llm/transformer-block'],
		related: ['self-attention']
	},
	{
		id: 'autoencoder',
		term: 'autoencoder',
		definition:
			'A network trained to rebuild its own input after squeezing it through a narrow middle layer. The encoder half makes a short code, the decoder half draws the input back from it.',
		lessons: ['vision/autoencoder'],
		related: ['latent-space', 'hidden-layer']
	},
	{
		id: 'backpropagation',
		term: 'backpropagation',
		aliases: ['backprop', 'backward pass'],
		definition:
			'The method that finds the gradient of every weight in one backward sweep, passing each neuron\'s share of the blame from the output back toward the inputs.',
		lessons: ['fundamentals/backpropagation', 'fundamentals/train-a-network', 'llm/training'],
		related: ['chain-rule', 'gradient', 'forward-pass']
	},
	{
		id: 'bagging',
		term: 'bagging',
		aliases: ['bootstrap aggregating'],
		definition:
			'Training each model on a random resample of the data, with some examples used twice and some left out, then averaging their answers. The average is steadier than any one model.',
		lessons: ['classic-ml/decision-trees'],
		related: ['random-forest', 'bias-and-variance']
	},
	{
		id: 'batch',
		term: 'batch',
		definition:
			'A group of training examples used together in one training step. The loss and the gradients are averaged over the batch, which gives a steadier nudge than a single example.',
		lessons: ['llm/training'],
		related: ['training', 'gradient-descent']
	},
	{
		id: 'bias',
		term: 'bias',
		definition:
			'A number a neuron adds to its sum whatever the inputs are. It shifts the baseline and sets how easily the neuron responds.',
		tone: 'param',
		lessons: ['fundamentals/neuron', 'fundamentals/forward-pass', 'fundamentals/loss-landscape'],
		related: ['weight', 'neuron', 'parameter']
	},
	{
		id: 'bias-and-variance',
		term: 'bias and variance',
		aliases: ['variance'],
		definition:
			'Two sources of error. Bias comes from a model too rigid to match the pattern. Variance comes from a model so flexible that its answer depends on which sample it saw.',
		lessons: ['classic-ml/overfitting'],
		related: ['overfitting', 'underfitting', 'model-complexity']
	},
	{
		id: 'byte-pair-encoding',
		term: 'byte-pair encoding',
		aliases: ['BPE', 'byte pair encoding'],
		definition:
			'A way to build a tokenizer. Start from single characters, merge the most common adjacent pair into a new token, and repeat. The learned merges then cut any new text.',
		lessons: ['llm/tokenization'],
		related: ['tokenization', 'token', 'vocabulary']
	},
	{
		id: 'causal-mask',
		term: 'causal mask',
		aliases: ['no peeking ahead', 'hide later words'],
		definition:
			'A rule that lets each token look only at itself and the tokens before it, so a model that predicts the next token cannot peek at the answer.',
		lessons: ['llm/attention', 'llm/transformer-block'],
		related: ['self-attention', 'attention-head']
	},
	{
		id: 'centroid',
		term: 'centroid',
		aliases: ['center', 'cluster center'],
		definition:
			'The middle of a cluster: the average position of its points. K-means keeps one per cluster and moves it to the average of its points in every round.',
		lessons: ['classic-ml/k-means'],
		related: ['k-means', 'clustering']
	},
	{
		id: 'chain-rule',
		term: 'chain rule',
		definition:
			'A rule from calculus: to find how one quantity affects another through a chain of steps, combine how each step affects the next. Backpropagation uses it to reuse work.',
		lessons: ['fundamentals/backpropagation'],
		related: ['backpropagation', 'gradient']
	},
	{
		id: 'channel',
		term: 'channel',
		definition:
			'One layer of values across an image, such as the red values. A grayscale image has one channel and a color photo has three.',
		lessons: ['vision/convolution'],
		related: ['pixel', 'convolution']
	},
	{
		id: 'checkpoint',
		term: 'checkpoint',
		definition:
			'A saved copy of a model\'s weights at one moment of training. Comparing checkpoints from different steps shows how the model changed while it learned.',
		lessons: ['llm/training'],
		related: ['training', 'weight']
	},
	{
		id: 'classification',
		term: 'classification',
		aliases: ['classifier', 'class'],
		definition:
			'The task of choosing a label for each input, such as class A or class B. A classifier has to answer for every possible position, not only for the examples it saw.',
		lessons: ['classic-ml/decision-boundaries', 'fundamentals/train-a-network'],
		related: ['decision-boundary', 'logistic-regression']
	},
	{
		id: 'clustering',
		term: 'clustering',
		aliases: ['cluster'],
		definition:
			'Finding groups in data that has no labels. Points in the same cluster are more alike than points in different ones.',
		lessons: ['classic-ml/k-means'],
		related: ['k-means', 'centroid']
	},
	{
		id: 'convolution',
		term: 'convolution',
		definition:
			'Sliding a small kernel over an image. At every position the pixels under it are multiplied by its weights and added up, and the sums form a new image that shows where a pattern appears.',
		lessons: ['vision/convolution', 'vision/cnn-features'],
		related: ['kernel', 'feature-map', 'stride']
	},
	{
		id: 'convolutional-neural-network',
		term: 'convolutional neural network',
		aliases: ['CNN', 'convnet'],
		definition:
			'A neural network built from convolution layers whose kernels are learned. Early layers respond to simple strokes, and later layers respond to larger shapes.',
		lessons: ['vision/cnn-features', 'vision/convolution'],
		related: ['convolution', 'feature-map', 'receptive-field']
	},
	{
		id: 'cosine-similarity',
		term: 'cosine similarity',
		aliases: ['cosine'],
		definition:
			'A measure of how closely two vectors point the same way. It is 1 for the same direction, 0 for unrelated vectors and negative for opposite ones.',
		lessons: ['llm/embeddings'],
		related: ['vector', 'embedding']
	},
	{
		id: 'cross-entropy',
		term: 'cross-entropy loss',
		aliases: ['cross-entropy'],
		definition:
			'A loss for classifiers. It is small when the model gives a high probability to the correct class and large when it gives that class a low probability.',
		tone: 'grad',
		lessons: ['classic-ml/decision-boundaries', 'llm/training'],
		related: ['loss', 'logistic-regression']
	},
	{
		id: 'decision-boundary',
		term: 'decision boundary',
		definition:
			'The line or curve where a classifier\'s answer flips from one class to the other. Different models draw it differently: a line, soft blobs or boxes.',
		tone: 'text',
		lessons: ['classic-ml/decision-boundaries', 'fundamentals/train-a-network'],
		related: ['classification', 'logistic-regression', 'decision-tree']
	},
	{
		id: 'decision-tree',
		term: 'decision tree',
		definition:
			'A classifier that asks a series of yes or no questions about the inputs, each one cutting a region in two. Its boundary is a staircase of straight cuts.',
		lessons: ['classic-ml/decision-boundaries', 'classic-ml/decision-trees'],
		related: ['decision-boundary', 'classification', 'random-forest']
	},
	{
		id: 'denoising',
		term: 'denoising',
		definition:
			'Removing noise to get closer to the clean data. A diffusion model learns, for every noise level, which way is less noisy, and follows that direction step by step.',
		lessons: ['vision/diffusion'],
		related: ['diffusion-model', 'noise', 'sampling-step']
	},
	{
		id: 'diffusion-model',
		term: 'diffusion model',
		aliases: ['diffusion'],
		definition:
			'A generative model that learns to undo noise. It starts from pure noise and removes a little at each step until a new sample that fits the training pattern appears.',
		lessons: ['vision/diffusion'],
		related: ['denoising', 'noise', 'sampling']
	},
	{
		id: 'elbow',
		term: 'elbow',
		definition:
			'The bend in the curve of spread against the number of clusters, where adding more centers stops helping much. It is a hint for choosing k.',
		lessons: ['classic-ml/k-means'],
		related: ['k-means', 'inertia']
	},
	{
		id: 'embedding',
		term: 'embedding',
		aliases: ['embedding space'],
		definition:
			'The vector a model looks up for each token. Tokens that mean similar things get similar vectors, so they sit near each other in embedding space.',
		lessons: ['llm/embeddings'],
		related: ['vector', 'token', 'cosine-similarity']
	},
	{
		id: 'error',
		term: 'error',
		definition:
			'The gap between a model\'s guess and the true answer for one example. The loss combines the errors of all examples into a single number.',
		tone: 'grad',
		lessons: [
			'fundamentals/loss-landscape',
			'fundamentals/backpropagation',
			'classic-ml/linear-regression',
			'classic-ml/overfitting'
		],
		related: ['loss', 'mean-squared-error']
	},
	{
		id: 'feature-map',
		term: 'feature map',
		definition:
			'The output picture of one kernel. It shows where in the input that kernel\'s pattern appears and how strongly.',
		lessons: ['vision/cnn-features', 'vision/convolution'],
		related: ['kernel', 'convolution', 'pooling']
	},
	{
		id: 'feed-forward-layer',
		term: 'feed-forward layer',
		aliases: ['feed-forward'],
		definition:
			'A small network applied to each token\'s vector on its own. It stretches the vector to more numbers, applies a nonlinearity, squeezes it back and adds the result to the stream.',
		lessons: ['llm/transformer-block'],
		related: ['transformer-block', 'residual-connection']
	},
	{
		id: 'forward-pass',
		term: 'forward pass',
		definition:
			'One sweep of values through a network, from the inputs through each layer to the output. The result is the network\'s prediction.',
		lessons: [
			'fundamentals/forward-pass',
			'fundamentals/backpropagation',
			'fundamentals/train-a-network'
		],
		related: ['layer', 'backpropagation', 'neural-network']
	},
	{
		id: 'gini-impurity',
		term: 'Gini impurity',
		aliases: ['Gini'],
		definition:
			'A number for how mixed a group of examples is: 0 when they all share one class, 0.5 for an even mix of two. A tree picks the cut that gives its two sides the lowest weighted impurity.',
		lessons: ['classic-ml/decision-trees', 'classic-ml/decision-boundaries'],
		related: ['decision-tree', 'classification']
	},
	{
		id: 'gradient',
		term: 'gradient',
		definition:
			'How much and in which direction the loss changes when a weight grows a little. It points uphill, so training steps the other way.',
		tone: 'grad',
		lessons: [
			'fundamentals/gradient-descent',
			'fundamentals/backpropagation',
			'classic-ml/linear-regression'
		],
		related: ['gradient-descent', 'loss', 'learning-rate']
	},
	{
		id: 'gradient-descent',
		term: 'gradient descent',
		definition:
			'Training by repeated small steps downhill. At each step it finds the gradient and moves every weight a little the opposite way, so the loss goes down.',
		lessons: [
			'fundamentals/gradient-descent',
			'fundamentals/backpropagation',
			'classic-ml/linear-regression',
			'fundamentals/train-a-network',
			'llm/training'
		],
		related: ['gradient', 'learning-rate', 'optimizer']
	},
	{
		id: 'greedy-decoding',
		term: 'greedy decoding',
		definition:
			'Always picking the most likely next token. The text is predictable, but it often falls into a loop and repeats itself.',
		lessons: ['llm/sampling', 'llm/training'],
		related: ['sampling', 'temperature']
	},
	{
		id: 'hidden-layer',
		term: 'hidden layer',
		definition:
			'A layer between the input and the output. Its values are neither the data nor the answer. They are the network\'s own intermediate features.',
		lessons: [
			'fundamentals/forward-pass',
			'fundamentals/activations',
			'fundamentals/backpropagation',
			'fundamentals/train-a-network',
			'vision/autoencoder'
		],
		related: ['layer', 'neural-network']
	},
	{
		id: 'inertia',
		term: 'inertia',
		aliases: ['spread'],
		definition:
			'The total squared distance from every point to its cluster center. K-means lowers it with every assign and every move, so the run must stop somewhere.',
		tone: 'grad',
		lessons: ['classic-ml/k-means'],
		related: ['k-means', 'elbow', 'local-minimum']
	},
	{
		id: 'k-means',
		term: 'k-means',
		aliases: ['k-means clustering'],
		definition:
			'A clustering method that alternates two steps: assign each point to its nearest center, then move each center to the average of its points. It repeats until nothing changes.',
		lessons: ['classic-ml/k-means'],
		related: ['clustering', 'centroid', 'local-minimum']
	},
	{
		id: 'k-nearest-neighbors',
		term: 'k-nearest neighbors',
		aliases: ['kNN', 'k-NN'],
		definition:
			'A classifier with no training step. To label a spot, it finds the k closest examples and takes a vote among them.',
		lessons: ['classic-ml/decision-boundaries'],
		related: ['classification', 'decision-boundary', 'overfitting']
	},
	{
		id: 'kernel',
		term: 'kernel',
		aliases: ['filter'],
		definition:
			'A small grid of weights, such as 3 by 3, laid on a patch of an image to compute one number. Different kernels respond to different patterns, such as edges.',
		tone: 'param',
		lessons: ['vision/convolution', 'vision/cnn-features'],
		related: ['convolution', 'feature-map', 'weight']
	},
	{
		id: 'key',
		term: 'key',
		definition:
			'In attention, what a token offers to the others. When a query and a key line up, that token is worth looking at.',
		tone: 'param',
		lessons: ['llm/attention'],
		related: ['query', 'value', 'self-attention']
	},
	{
		id: 'language-model',
		term: 'language model',
		definition:
			'A model that gives a probability to every possible next token, given the text so far. Chat models write by picking from these probabilities one token at a time.',
		lessons: ['llm/tokenization', 'llm/sampling', 'llm/transformer-block', 'llm/training'],
		related: ['token', 'sampling', 'transformer']
	},
	{
		id: 'latent-space',
		term: 'latent space',
		aliases: ['latent', 'code'],
		definition:
			'The space of a network\'s inner codes. Similar inputs get nearby codes, and a decoder can turn any point of the space into an output, even one no example came from.',
		lessons: ['vision/autoencoder'],
		related: ['autoencoder', 'embedding']
	},
	{
		id: 'layer',
		term: 'layer',
		definition:
			'A group of neurons that work side by side on the same inputs. A network passes its values from one layer to the next.',
		lessons: [
			'fundamentals/forward-pass',
			'fundamentals/backpropagation',
			'fundamentals/train-a-network',
			'llm/transformer-block',
			'vision/cnn-features'
		],
		related: ['hidden-layer', 'neural-network']
	},
	{
		id: 'layer-normalization',
		term: 'layer normalization',
		aliases: ['normalize'],
		definition:
			'Rescaling a vector to a fixed center and spread before a layer uses it. It keeps the numbers in a workable range, however deep the stack gets.',
		lessons: ['llm/transformer-block'],
		related: ['transformer-block', 'residual-connection']
	},
	{
		id: 'learning-rate',
		term: 'learning rate',
		aliases: ['eta', 'step size'],
		definition:
			'A number that scales how big each training step is. Too small and training crawls. Too large and it overshoots and may diverge.',
		tone: 'grad',
		lessons: [
			'fundamentals/backpropagation',
			'fundamentals/gradient-descent',
			'fundamentals/train-a-network',
			'llm/training'
		],
		related: ['gradient-descent', 'optimizer']
	},
	{
		id: 'least-squares',
		term: 'least squares',
		definition:
			'Choosing the line that makes the total of the squared misses as small as possible. For a straight line there is an exact answer, so no stepping is needed.',
		lessons: ['classic-ml/linear-regression'],
		related: ['linear-regression', 'mean-squared-error']
	},
	{
		id: 'linear-regression',
		term: 'linear regression',
		definition:
			'Fitting a straight line to points to predict one value from another. The best line is the one with the smallest mean squared error.',
		lessons: ['classic-ml/linear-regression'],
		related: ['least-squares', 'mean-squared-error', 'outlier']
	},
	{
		id: 'local-minimum',
		term: 'local minimum',
		aliases: ['minima'],
		definition:
			'A low point of the loss that is not the lowest one. A search that only goes downhill can settle in one and stay there.',
		lessons: ['fundamentals/gradient-descent', 'classic-ml/k-means'],
		related: ['gradient-descent', 'loss-landscape']
	},
	{
		id: 'logistic-regression',
		term: 'logistic regression',
		definition:
			'A classifier that scores a position with a weighted sum and squashes it into a probability. Its decision boundary is always a straight line.',
		lessons: ['classic-ml/decision-boundaries'],
		related: ['classification', 'decision-boundary', 'sigmoid']
	},
	{
		id: 'logits',
		term: 'logits',
		aliases: ['logit', 'raw scores'],
		definition:
			'The raw scores a model gives to every token before they are turned into probabilities. A larger score means a more likely token.',
		lessons: ['llm/sampling'],
		related: ['softmax', 'temperature']
	},
	{
		id: 'loss',
		term: 'loss',
		aliases: ['loss function'],
		definition:
			'One number that says how wrong a model is across all its examples. Zero is perfect and bigger is worse. Training tries to make it small.',
		tone: 'grad',
		lessons: [
			'fundamentals/loss-landscape',
			'fundamentals/gradient-descent',
			'fundamentals/backpropagation',
			'fundamentals/train-a-network',
			'classic-ml/decision-boundaries',
			'llm/training'
		],
		related: ['mean-squared-error', 'loss-landscape', 'error']
	},
	{
		id: 'loss-landscape',
		term: 'loss landscape',
		definition:
			'The surface you get by trying every setting of the weights and plotting the loss as a height. Training is a walk across it toward a low point.',
		lessons: ['fundamentals/loss-landscape', 'fundamentals/gradient-descent'],
		related: ['loss', 'gradient-descent', 'local-minimum']
	},
	{
		id: 'mean-squared-error',
		term: 'mean squared error',
		aliases: ['MSE'],
		definition:
			'The average of the squared gaps between a model\'s guesses and the true values. Squaring makes big misses count far more than small ones.',
		tone: 'grad',
		lessons: [
			'fundamentals/loss-landscape',
			'classic-ml/linear-regression',
			'classic-ml/overfitting'
		],
		related: ['loss', 'error', 'least-squares']
	},
	{
		id: 'model-complexity',
		term: 'model complexity',
		aliases: ['degree', 'flexibility', 'capacity'],
		definition:
			'How flexible a model is, meaning how many bends it can make. More helps up to a point, and then the model starts to chase noise.',
		lessons: ['classic-ml/overfitting', 'fundamentals/train-a-network'],
		related: ['overfitting', 'underfitting', 'bias-and-variance']
	},
	{
		id: 'momentum',
		term: 'momentum',
		definition:
			'An addition to gradient descent that keeps a running velocity. Zigzags cancel out while steady progress in one direction builds up.',
		lessons: ['fundamentals/gradient-descent'],
		related: ['optimizer', 'adam', 'gradient-descent']
	},
	{
		id: 'neural-network',
		term: 'neural network',
		aliases: ['network', 'net'],
		definition:
			'Many neurons arranged in layers, with a weight on every connection. The weights are what the network knows, and training adjusts them.',
		lessons: [
			'fundamentals/neuron',
			'fundamentals/forward-pass',
			'fundamentals/backpropagation',
			'fundamentals/train-a-network'
		],
		related: ['neuron', 'layer', 'weight']
	},
	{
		id: 'neuron',
		term: 'neuron',
		aliases: ['unit', 'hidden unit'],
		definition:
			'A tiny function. It multiplies each input by a weight, adds the results and a bias, and passes the sum through an activation function to give one output.',
		lessons: [
			'fundamentals/neuron',
			'fundamentals/activations',
			'fundamentals/forward-pass',
			'fundamentals/backpropagation'
		],
		related: ['weight', 'bias', 'activation-function']
	},
	{
		id: 'noise',
		term: 'noise',
		definition:
			'Random variation that hides a pattern. Real measurements contain it, and diffusion models add it on purpose to learn how to remove it.',
		tone: 'grad',
		lessons: [
			'classic-ml/linear-regression',
			'classic-ml/overfitting',
			'classic-ml/decision-boundaries',
			'fundamentals/train-a-network',
			'vision/diffusion'
		],
		related: ['overfitting', 'diffusion-model']
	},
	{
		id: 'optimizer',
		term: 'optimizer',
		definition:
			'The rule that turns gradients into weight updates. Plain gradient descent, momentum and Adam are all optimizers.',
		lessons: ['fundamentals/gradient-descent'],
		related: ['gradient-descent', 'momentum', 'adam']
	},
	{
		id: 'outlier',
		term: 'outlier',
		definition:
			'A point far from the rest. Because misses are squared, a single outlier can pull a fitted line hard toward itself.',
		lessons: ['classic-ml/linear-regression', 'classic-ml/decision-boundaries'],
		related: ['linear-regression', 'mean-squared-error']
	},
	{
		id: 'overfitting',
		term: 'overfitting',
		aliases: ['overfit'],
		definition:
			'When a model is so flexible that it chases the noise in its training data. Its training error is low, but it does badly on data it has not seen.',
		lessons: ['classic-ml/overfitting', 'classic-ml/decision-boundaries', 'classic-ml/decision-trees', 'llm/training'],
		related: ['underfitting', 'test-set', 'model-complexity']
	},
	{
		id: 'padding',
		term: 'padding',
		definition:
			'A border of zeros added around an image so a kernel can sit on the edge pixels. With it, the output can keep the size of the image.',
		lessons: ['vision/convolution'],
		related: ['stride', 'convolution']
	},
	{
		id: 'parameter',
		term: 'parameter',
		aliases: ['parameters'],
		definition:
			'A number a model learns from data, such as a weight or a bias. Training adjusts the parameters to make the loss smaller.',
		tone: 'param',
		lessons: ['classic-ml/linear-regression', 'fundamentals/loss-landscape'],
		related: ['weight', 'bias']
	},
	{
		id: 'penalty',
		term: 'penalty',
		aliases: ['regularization', 'ridge regression'],
		definition:
			'An extra cost for large coefficients that keeps a fitted curve smooth. It is one way to fight overfitting without collecting more data.',
		lessons: ['classic-ml/overfitting'],
		related: ['overfitting', 'model-complexity']
	},
	{
		id: 'perplexity',
		term: 'perplexity',
		definition:
			'A language model\'s loss written as a number of choices: e raised to the loss. A model with a perplexity of 12 is as unsure as someone choosing evenly among 12 tokens.',
		lessons: ['llm/training'],
		related: ['loss', 'cross-entropy', 'language-model']
	},
	{
		id: 'pixel',
		term: 'pixel',
		definition:
			'One dot of an image, stored as a number for its brightness or a few numbers for its color.',
		lessons: ['vision/convolution', 'vision/cnn-features', 'vision/autoencoder', 'vision/diffusion'],
		related: ['channel', 'convolution']
	},
	{
		id: 'pooling',
		term: 'pooling',
		aliases: ['max pooling'],
		definition:
			'Shrinking a feature map by keeping one number from each small block, usually the largest. A small shift in the image then changes the result very little.',
		lessons: ['vision/cnn-features'],
		related: ['feature-map', 'stride']
	},
	{
		id: 'probability',
		term: 'probability',
		definition:
			'A number from 0 to 1 for how likely something is. A model\'s probabilities over all the choices add up to 1.',
		lessons: [
			'llm/sampling',
			'fundamentals/forward-pass',
			'classic-ml/decision-boundaries',
			'vision/cnn-features'
		],
		related: ['softmax', 'sampling']
	},
	{
		id: 'query',
		term: 'query',
		definition:
			'In attention, what a token is looking for, written as a short list of numbers. A token\'s query is matched against the key of every token.',
		tone: 'ok',
		lessons: ['llm/attention'],
		related: ['key', 'value', 'self-attention']
	},
	{
		id: 'random-forest',
		term: 'random forest',
		definition:
			'Many decision trees, each grown on a random resample of the data and with random choices of question, whose answers are averaged. It overfits far less than a single deep tree.',
		lessons: ['classic-ml/decision-trees'],
		related: ['decision-tree', 'bagging', 'overfitting']
	},
	{
		id: 'receptive-field',
		term: 'receptive field',
		definition:
			'The part of the input that one cell of a feature map can see. It grows with depth, because each layer combines what the layer before it saw.',
		tone: 'ok',
		lessons: ['vision/cnn-features'],
		related: ['feature-map', 'convolutional-neural-network']
	},
	{
		id: 'relu',
		term: 'ReLU',
		definition:
			'An activation that passes positive inputs unchanged and outputs zero for negative ones. It is cheap to compute, and its slope does not fade on the positive side.',
		lessons: [
			'fundamentals/neuron',
			'fundamentals/activations',
			'fundamentals/train-a-network',
			'vision/cnn-features',
			'llm/transformer-block'
		],
		related: ['activation-function', 'sigmoid', 'tanh']
	},
	{
		id: 'residual-connection',
		term: 'residual connection',
		aliases: ['residual stream', 'skip connection', 'bypass'],
		definition:
			'A path that carries a layer\'s input around the layer and adds the layer\'s output to it. Old information survives unless an update writes over it.',
		lessons: ['llm/transformer-block'],
		related: ['transformer-block', 'layer-normalization']
	},
	{
		id: 'sampling',
		term: 'sampling',
		definition:
			'Picking from a set of options at random, with each chance set by its probability. A model that samples its next token can give a different text on every run.',
		lessons: ['llm/sampling', 'vision/diffusion', 'llm/training'],
		related: ['temperature', 'top-k', 'greedy-decoding']
	},
	{
		id: 'sampling-step',
		term: 'sampling step',
		aliases: ['reverse step', 'denoising step'],
		definition:
			'One small move from a noisier sample toward a cleaner one. A picture takes many such steps, and more steps give a sharper result at the cost of more computation.',
		lessons: ['vision/diffusion'],
		related: ['denoising', 'diffusion-model']
	},
	{
		id: 'self-attention',
		term: 'self-attention',
		aliases: ['attention'],
		definition:
			'The step where every token looks at the others and takes in a share of each, weighted by how well they match. It is how tokens exchange information.',
		lessons: ['llm/attention', 'llm/transformer-block', 'llm/embeddings'],
		related: ['query', 'key', 'value']
	},
	{
		id: 'sigmoid',
		term: 'sigmoid',
		definition:
			'An activation that squeezes any input into the range 0 to 1 with a smooth S shape. Its output can be read as a score or a probability.',
		lessons: [
			'fundamentals/neuron',
			'fundamentals/activations',
			'fundamentals/forward-pass',
			'fundamentals/backpropagation'
		],
		related: ['activation-function', 'tanh', 'vanishing-gradient']
	},
	{
		id: 'softmax',
		term: 'softmax',
		definition:
			'A function that turns a list of scores into shares that add up to 100%, giving the highest score the biggest share.',
		lessons: ['llm/attention', 'llm/sampling', 'llm/transformer-block', 'vision/cnn-features'],
		related: ['logits', 'probability', 'self-attention']
	},
	{
		id: 'stride',
		term: 'stride',
		definition:
			'How many pixels a kernel moves between positions. A stride of 2 visits every other position and halves the size of the output.',
		lessons: ['vision/convolution', 'vision/cnn-features'],
		related: ['padding', 'convolution', 'pooling']
	},
	{
		id: 'tanh',
		term: 'tanh',
		definition:
			'An activation with the same S shape as sigmoid, but its output runs from -1 to 1 and passes through zero in the middle.',
		lessons: [
			'fundamentals/activations',
			'fundamentals/loss-landscape',
			'fundamentals/backpropagation',
			'fundamentals/train-a-network',
			'vision/autoencoder'
		],
		related: ['activation-function', 'sigmoid']
	},
	{
		id: 'target',
		term: 'target',
		definition:
			'The answer we want the model to give for an input. The loss measures how far the model\'s guess is from it.',
		tone: 'ok',
		lessons: ['fundamentals/backpropagation'],
		related: ['loss', 'error']
	},
	{
		id: 'temperature',
		term: 'temperature',
		definition:
			'A setting that reshapes the odds of the next token before it is picked. Low values favor the top token and give safe, repetitive text. High values flatten the odds and give surprising text.',
		tone: 'grad',
		lessons: ['llm/sampling'],
		related: ['sampling', 'logits', 'top-p']
	},
	{
		id: 'test-set',
		term: 'test set',
		aliases: ['test data', 'test error'],
		definition:
			'Examples kept out of the fitting and used only to check how a model does on data it has never seen. The error on them is the test error.',
		tone: 'ok',
		lessons: ['classic-ml/overfitting', 'classic-ml/decision-trees', 'llm/training'],
		related: ['training-set', 'overfitting']
	},
	{
		id: 'token',
		term: 'token',
		definition:
			'A piece of text, such as a word, part of a word or a single character, that a language model treats as one unit. Each token gets an ID number.',
		lessons: [
			'llm/tokenization',
			'llm/embeddings',
			'llm/attention',
			'llm/transformer-block',
			'llm/sampling',
			'llm/training'
		],
		related: ['tokenization', 'vocabulary', 'embedding']
	},
	{
		id: 'tokenization',
		term: 'tokenization',
		aliases: ['tokenizer'],
		definition:
			'Cutting text into tokens and giving each one an ID number. It comes first because a model works with numbers, not letters.',
		lessons: ['llm/tokenization'],
		related: ['token', 'byte-pair-encoding', 'vocabulary']
	},
	{
		id: 'top-k',
		term: 'top-k',
		aliases: ['top k'],
		definition:
			'Keeping only the k most likely tokens and cutting off the rest before one is picked. It stops the text from wandering to very unlikely tokens.',
		lessons: ['llm/sampling'],
		related: ['top-p', 'temperature', 'sampling']
	},
	{
		id: 'top-p',
		term: 'top-p',
		aliases: ['top p', 'nucleus sampling'],
		definition:
			'Keeping the most likely tokens until their probabilities add up to p, and cutting off the rest. The number kept adapts to how sure the model is.',
		lessons: ['llm/sampling'],
		related: ['top-k', 'temperature', 'sampling']
	},
	{
		id: 'training',
		term: 'training',
		definition:
			'Repeatedly adjusting a model\'s weights so its loss gets smaller. Each lap runs a forward pass, measures the loss, finds the gradients and updates the weights.',
		tone: 'grad',
		lessons: [
			'fundamentals/loss-landscape',
			'fundamentals/gradient-descent',
			'fundamentals/backpropagation',
			'fundamentals/train-a-network',
			'llm/training'
		],
		related: ['gradient-descent', 'loss', 'backpropagation']
	},
	{
		id: 'training-set',
		term: 'training set',
		aliases: ['training data', 'training points', 'training error'],
		definition:
			'The examples a model is fitted to. The error on them is the training error, and it keeps falling as the model gets more flexible.',
		tone: 'data',
		lessons: ['classic-ml/overfitting'],
		related: ['test-set', 'overfitting']
	},
	{
		id: 'transformer',
		term: 'transformer',
		definition:
			'A network design for sequences, built from a stack of identical blocks. Each block lets tokens exchange information through attention and then refines each token on its own.',
		lessons: ['llm/transformer-block'],
		related: ['transformer-block', 'self-attention']
	},
	{
		id: 'transformer-block',
		term: 'transformer block',
		definition:
			'The repeating unit of a transformer: self-attention, a feed-forward layer, and residual paths with normalization. A model stacks many blocks, and each refines the stream it receives.',
		lessons: ['llm/transformer-block'],
		related: ['transformer', 'residual-connection', 'feed-forward-layer']
	},
	{
		id: 'underfitting',
		term: 'underfitting',
		aliases: ['underfit'],
		definition:
			'When a model is too simple for the pattern, so its error stays high even on the data it was fitted to.',
		lessons: ['classic-ml/overfitting'],
		related: ['overfitting', 'model-complexity']
	},
	{
		id: 'value',
		term: 'value',
		definition:
			'In attention, the information a token passes on. A token\'s output is the average of all the values, weighted by the attention shares.',
		tone: 'data',
		lessons: ['llm/attention'],
		related: ['query', 'key', 'self-attention']
	},
	{
		id: 'vanishing-gradient',
		term: 'vanishing gradient',
		aliases: ['vanishing gradients'],
		definition:
			'A problem where an activation\'s slope flattens toward zero, so the gradients reaching earlier weights shrink and learning slows down.',
		lessons: ['fundamentals/activations'],
		related: ['sigmoid', 'gradient', 'relu']
	},
	{
		id: 'vector',
		term: 'vector',
		definition:
			'A list of numbers. A model represents a token as a vector, and the numbers place it as a point in a space.',
		lessons: ['llm/embeddings', 'llm/attention', 'llm/transformer-block'],
		related: ['embedding', 'cosine-similarity']
	},
	{
		id: 'vocabulary',
		term: 'vocabulary',
		definition:
			'The set of all tokens a model knows, each with an ID. When the model predicts what comes next, it gives a probability to every one of them.',
		lessons: ['llm/tokenization', 'llm/sampling', 'llm/transformer-block', 'llm/training'],
		related: ['token', 'tokenization']
	},
	{
		id: 'weight',
		term: 'weight',
		aliases: ['weights'],
		definition:
			'A number on a connection that scales the signal passing through it. A bigger weight strengthens the signal and a negative weight flips its sign. Weights are what a network learns.',
		tone: 'param',
		lessons: [
			'fundamentals/neuron',
			'fundamentals/activations',
			'fundamentals/forward-pass',
			'fundamentals/loss-landscape',
			'fundamentals/gradient-descent',
			'fundamentals/backpropagation',
			'fundamentals/train-a-network',
			'llm/attention',
			'llm/transformer-block',
			'vision/convolution',
			'llm/training'
		],
		related: ['bias', 'neuron', 'parameter']
	}
];

function byTerm(a: GlossaryEntry, b: GlossaryEntry): number {
	const x = a.term.toLowerCase();
	const y = b.term.toLowerCase();
	return x < y ? -1 : x > y ? 1 : 0;
}

/** Every entry, sorted by term, ignoring case. */
export const glossary: GlossaryEntry[] = [...entries].sort(byTerm);

const byId = new Map(glossary.map((e) => [e.id, e]));

export function findEntry(id: string): GlossaryEntry | undefined {
	return byId.get(id);
}

/** Entries whose `lessons` include the key: those it introduces (first in `lessons`) first, then the rest; each group in glossary order. */
export function termsForLesson(key: string): GlossaryEntry[] {
	const used = glossary.filter((e) => e.lessons.includes(key));
	return [...used.filter((e) => e.lessons[0] === key), ...used.filter((e) => e.lessons[0] !== key)];
}

/**
 * Case-insensitive search. An empty or blank query returns every entry. Otherwise: entries whose term or an
 * alias starts with the query, then entries whose term or an alias contains it, then entries whose definition
 * contains it; each group in glossary order, no entry twice.
 */
export function searchGlossary(query: string): GlossaryEntry[] {
	const q = query.trim().toLowerCase();
	if (!q) return glossary;
	const names = (e: GlossaryEntry) => [e.term, ...(e.aliases ?? [])].map((s) => s.toLowerCase());
	const starts = glossary.filter((e) => names(e).some((n) => n.startsWith(q)));
	const hasName = glossary.filter(
		(e) => !starts.includes(e) && names(e).some((n) => n.includes(q))
	);
	const hasDefinition = glossary.filter(
		(e) => !starts.includes(e) && !hasName.includes(e) && e.definition.toLowerCase().includes(q)
	);
	return [...starts, ...hasName, ...hasDefinition];
}

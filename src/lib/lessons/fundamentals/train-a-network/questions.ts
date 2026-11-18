import type { Question } from '../../types.ts';

/** The accuracy readout is a text such as '91%'. A missing readout gives NaN, and NaN >= 100 is false. */
const percent = (v: unknown): number => parseInt(String(v), 10);

export const questions: Question[] = [
	{
		kind: 'choice',
		prompt:
			'You train the circles with 4 units and a learning rate of 0.03. After 600 steps the [[grad:loss]] has only gone from about 0.69 to 0.66 and the accuracy is 70%. What does that tell you?\n\nDecide first, then check with **Train** and the slider.',
		params: { scenario: 7, lr: 0.03 },
		controls: ['run', 'reset', 'lr'],
		options: [
			{
				text: 'These 4 units cannot learn this pattern, so the loss stays near 0.66.',
				why: 'The same 4 units reach 100% accuracy on these circles with a rate of 0.3. With 0.03 the loss keeps falling, only slowly: it is near 0.22 after 4000 steps.'
			},
			{
				text: 'Training has finished: the loss has reached its lowest point.',
				why: 'The loss is still falling. It passes 0.5 after about 2000 steps and the accuracy climbs with it. Each nudge is just tiny.'
			},
			{
				text: 'Learning works but is slow: each nudge is tiny, so it needs many more steps or a larger rate.',
				correct: true,
				why: 'Each step moves the [[param:weights]] by the learning rate times the gradient, so a small rate makes small moves. With 0.3 the same network reaches 100% accuracy in about 600 steps.'
			},
			{
				text: 'The learning rate is too high, so the weights keep jumping past the answer.',
				why: 'A rate that is too high makes the loss jump up and down. Here it falls steadily, and 0.03 is a small rate. Raising it to 0.3 makes the loss fall faster.'
			}
		]
	},
	{
		kind: 'task',
		prompt:
			'This network has to tell the XOR groups apart. Make it classify every dot correctly: reach an accuracy of 100%.',
		params: { scenario: 8, dataset: 2, hidden: 2 },
		controls: ['run', 'reset', 'hidden', 'lr'],
		goal: ({ readouts }) => percent(readouts.acc) >= 100,
		hint: 'Press Train and watch the accuracy. If it stalls while the loss stays high, ask what the network is able to draw, not how fast it learns.',
		solved:
			'With 2 units the network stalls well below 100% here, whatever the learning rate. Each hidden unit adds a bend to the boundary, and 4 units give enough bends to separate all four clusters of XOR.'
	},
	{
		kind: 'choice',
		prompt:
			'You raise the noise to 0.25 on the circles, with 4 units, and train for a long time. What do you expect at the end?\n\nDecide first, then check with the noise slider and **Train**.',
		params: { scenario: 9 },
		controls: ['run', 'reset', 'noise'],
		options: [
			{
				text: 'Accuracy reaches 100% again, but training takes longer.',
				why: 'With noise 0.25 the dots of the two groups overlap, so some sit among the other group. Training longer does not separate them: the accuracy ends near 90%.'
			},
			{
				text: 'Accuracy stays below 100%, because some dots now overlap the other group.',
				correct: true,
				why: 'Noise moves every dot at random, so some dots of one class end up among the other class, and no smooth boundary separates them cleanly. Here the loss settles near 0.3 and the accuracy near 90%. The data sets that limit, not the training.'
			},
			{
				text: 'The loss goes up, because noisy dots give the weights wrong gradients.',
				why: 'Each gradient is computed correctly from the dots as they are. The loss still falls, from about 0.69 to about 0.3. It just cannot fall as far as on clean data.'
			},
			{
				text: 'Nothing changes, because noise only blurs the picture and the network ignores it.',
				why: 'The network trains on the noisy dots, so the noise changes what it can reach. With 0.05 the accuracy gets to 100%. With 0.25 it stays near 90%.'
			}
		]
	}
];

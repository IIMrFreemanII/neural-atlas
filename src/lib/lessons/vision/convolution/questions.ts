import { outSize } from '../../../ml/conv.ts';
import { shown } from '../../questions.ts';
import type { Question } from '../../types.ts';

export const questions: Question[] = [
	{
		kind: 'task',
		prompt:
			'The image is 24 × 24 pixels and the kernel is 3 × 3. Make the [[data:output]] map exactly 12 × 12. The **Output** chip shows its size.',
		params: { scenario: 6, kernel: 3, image: 0, stride: 1, pad: 0 },
		controls: ['stride', 'pad'],
		goal: ({ params: p }) => outSize(24, 3, shown(p.stride, 0), shown(p.pad, 0)) === 12,
		hint: 'A bigger stride visits fewer positions, and padding adds a border of zeros before the kernel starts sliding. Try the combinations and watch the chip.',
		solved:
			'With a stride of 2 and padding, the kernel slides over a 26 × 26 image and visits every other position: ⌊(24 + 2 − 3) / 2⌋ + 1 = 12. Without padding the same stride gives 11.',
		example: { params: { stride: 2, pad: 1 } }
	},
	{
		kind: 'choice',
		prompt:
			'Hover the inside of the white rectangle, where every pixel is 1, so the patch under the kernel is flat. For which of these kernels does **This position** read exactly 0 there?\n\nDecide first, then check by choosing kernels and hovering.',
		params: { scenario: 4, image: 0, kernel: 0, stride: 1, pad: 0 },
		controls: ['kernel'],
		options: [
			{
				text: 'Blur',
				why: 'Blur adds up nine pixels. On a flat patch of 1s they add up to 9.0, so the output is large there, not 0.'
			},
			{
				text: 'Sharpen',
				why: 'Its weights are 5 in the middle and four times −1 around it, which add up to 1. A flat patch of 1s gives 1.0, so Sharpen leaves flat areas as they are.'
			},
			{
				text: 'Vertical',
				correct: true,
				why: 'Its weights are −1, 0, 1, −2, 0, 2, −1, 0, 1 and add up to 0. On a flat patch every pixel is the same, so the products add up to that pixel × 0 = 0. Such a kernel only reacts where pixels differ.'
			},
			{
				text: 'Same',
				why: 'Same has a single weight of 1 in the middle, so it copies the pixel under it. A flat patch of 1s gives 1.0.'
			}
		]
	},
	{
		kind: 'choice',
		prompt:
			'The white rectangle has four sides. Which kernel gives non-zero values along its top and bottom sides, but 0 along its left and right sides?\n\nDecide first, then check with the kernel selector.',
		params: { scenario: 5, image: 0, kernel: 0, stride: 1, pad: 0 },
		controls: ['kernel'],
		options: [
			{
				text: 'Vertical',
				why: 'This kernel compares the left and right columns of a patch. It gives 4.0 on the left side of the rectangle and −4.0 on the right side, and 0 on the top and bottom sides, where the pixels do not change from left to right.'
			},
			{
				text: 'Horizontal',
				correct: true,
				why: 'It compares the top and bottom rows of a patch. Along the top side of the rectangle it gives 4.0 and along the bottom side −4.0. Along the left and right sides every row of the patch looks alike, so the products cancel to 0.'
			},
			{
				text: 'Outline',
				why: 'Outline answers wherever a pixel differs from its neighbors in any direction. It gives 3.0 on all four sides, not only on two of them.'
			},
			{
				text: 'Blur',
				why: 'Blur does not look for changes. It adds up the nine pixels, which gives 6.0 on every side of the rectangle, so it does not single out two of them.'
			}
		]
	}
];

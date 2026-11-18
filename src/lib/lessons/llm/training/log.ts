// Written by scripts/build-training-log.ts. Do not edit by hand.
// The model of the language lessons at 11 moments of its training run (scripts/train-tinystories.ts).
import type { TrainLog } from './log-types.ts';

export const TRAIN_LOG: TrainLog = {
	prompts: ['once upon a time there was a', 'one day , lily went', 'the little girl wanted to'],
	probe: ['tom', 'and', 'his', 'dog', 'went', 'to', 'the', 'park', '.', 'they', 'saw', 'a', 'big', 'red', 'ball', '.'],
	vocab: 2048,
	uniformLoss: 7.6246,
	batch: 64,
	ctx: 32,
	steps: 12000,
	curveStep: [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 15, 16, 18, 20, 22, 23, 25, 27, 29, 32, 34, 36, 39, 41, 44, 46, 49, 52, 55, 58, 62, 65, 69, 72, 76, 80, 84, 89, 93, 98, 103, 108, 113, 119, 124, 130, 136, 143, 149, 156, 163, 171, 179, 187, 195, 204, 213, 222, 232, 242, 253, 264, 275, 287, 300, 312, 326, 340, 354, 369, 385, 401, 418, 435, 454, 473, 492, 513, 534, 556, 579, 603, 628, 654, 681, 709, 738, 768, 799, 832, 866, 901, 938, 976, 1016, 1057, 1100, 1144, 1191, 1239, 1289, 1341, 1395, 1451, 1510, 1571, 1634, 1699, 1768, 1839, 1913, 1989, 2069, 2152, 2238, 2328, 2421, 2518, 2619, 2723, 2832, 2945, 3063, 3185, 3312, 3445, 3582, 3725, 3873, 4028, 4188, 4355, 4529, 4709, 4897, 5091, 5294, 5505, 5724, 5951, 6188, 6434, 6690, 6956, 7233, 7520, 7819, 8129, 8452, 8788, 9137, 9500, 9877, 10270, 10677, 11101, 11542, 12000],
	curveLoss: [7.945, 7.884, 7.883, 7.873, 7.799, 7.717, 7.647, 7.592, 7.532, 7.454, 7.405, 7.268, 7.207, 7.13, 7.049, 6.998, 6.971, 6.911, 6.834, 6.761, 6.665, 6.582, 6.516, 6.393, 6.298, 6.188, 6.104, 5.955, 5.824, 5.732, 5.626, 5.504, 5.413, 5.34, 5.28, 5.178, 5.12, 5.04, 4.935, 4.903, 4.823, 4.731, 4.714, 4.648, 4.54, 4.438, 4.336, 4.287, 4.212, 4.171, 4.128, 4.092, 4.03, 3.968, 3.908, 3.855, 3.801, 3.748, 3.71, 3.686, 3.639, 3.587, 3.576, 3.558, 3.539, 3.485, 3.459, 3.427, 3.407, 3.394, 3.38, 3.365, 3.327, 3.289, 3.262, 3.232, 3.235, 3.228, 3.207, 3.172, 3.178, 3.139, 3.115, 3.103, 3.09, 3.078, 3.078, 3.068, 3.044, 3.031, 3.012, 3.008, 2.99, 2.964, 2.959, 2.959, 2.944, 2.918, 2.915, 2.904, 2.883, 2.87, 2.858, 2.85, 2.841, 2.837, 2.825, 2.809, 2.797, 2.784, 2.782, 2.776, 2.763, 2.751, 2.736, 2.726, 2.725, 2.718, 2.713, 2.701, 2.687, 2.682, 2.679, 2.669, 2.656, 2.645, 2.636, 2.63, 2.626, 2.621, 2.612, 2.603, 2.592, 2.584, 2.579, 2.568, 2.56, 2.552, 2.548, 2.539, 2.532, 2.523, 2.513, 2.504, 2.491, 2.483, 2.477, 2.473, 2.466, 2.46, 2.451, 2.442, 2.433, 2.429, 2.428, 2.425, 2.416, 2.413, 2.413],
	checkpoints: [
		{
			step: 0,
			trainLoss: 7.9454,
			validLoss: 7.927,
			greedy: [
				['faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'life', 'life', 'life', 'life', 'life', 'life', 'life', 'life', 'life', 'life', 'life', 'life', 'life', 'life', 'life', 'life', 'life', 'life'],
				['faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'faces', 'live', 'live'],
				['carefully', 'ways', 'ways', 'ways', 'ways', 'ways', 'ways', 'ways', 'ways', 'ways', 'ways', 'ways', 'ways', 'ways', 'ways', 'jelly', 'jelly', 'jelly', 'jelly', 'jelly', 'jelly', 'jelly', 'jelly', 'jelly', 'jelly', 'jelly', 'jelly', 'getting']
			],
			sampled: [
				['shout', 'crawl', 'troubled', 'crayon', 'tie', 'hats', 'promise', 'hat', 'lay', 'painting', 'drawing', 'stretch', 'anyone', 'gave', 'lucy\'s', 'hope', 'bull', 'used', 'it', 'swan', 'thermometer', 'trap', 'wrong', 'patch', 'that', 'delicious', 'leo', 'pool'],
				['upset', 'asleep', 'than', 'cold', 'independent', 'flame', 'animals', 'tiger', 'peaceful', 'higher', 'replied', 'triangle', 'rug', 'walk', 'figure', 'seeds', 'tap', 'daddy', 'farmer', 'used', 'wearing', 'gentle', 'books', 'what\'s', 'sighed', 'as', 'tower', 'tv'],
				['hop', 'faster', 'lose', 'rare', 'truck', 'climbed', 'independent', 'city', 'wall', 'them', 'put', 'wake', 'snack', 'picked', 'use', 'sue\'s', 'thinks', 'trap', 'stood', ',', 'lady', 'daisy', 'shrimp', 'makeup', 'mrs', 'dolphin', 'cut', 'chair']
			],
			probe: [0.0002, 0.0004, 0.0006, 0.0004, 0.0011, 0.0002, 0.0003, 0.0008, 0.0003, 0.0001, 0.0003, 0.0005, 0.0002, 0.0002, 0.0005],
			guess: ['roar', 'colors', 'roar', 'dinner', 'getting', 'lights', 'wake', 'on', 'holding', 'holding', 'bedroom', 'on', 'roar', 'garage', 'cave'],
			guessP: [0.0045, 0.0068, 0.0061, 0.0059, 0.0047, 0.005, 0.0058, 0.0039, 0.0063, 0.0079, 0.0043, 0.0054, 0.0051, 0.0046, 0.0051],
			probeLoss: 7.9762
		},
		{
			step: 25,
			trainLoss: 6.8828,
			validLoss: 6.8467,
			greedy: [
				['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
				['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
				['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.']
			],
			sampled: [
				['successful', 'leopard', 'goose', 'hay', 'adventures', 'mask', 'wall', 'decided', 'shook', 'suit', 'obedient', 'create', 'free', 'wanted', 'amazed', 'teacher', 'coins', 'outside', '.', 'smelled', 'thermometer', 'collect', 'john', 'ouch', '.', 'cookies', 'plan', 'confused'],
				['money', 'dig', 'clean', 'if', 'mushroom', 'graceful', 'help', 'mixed', 'pointed', 'waved', 'store', 'crawled', 'puts', 'again', 'guard', 'dead', 'flame', 'gone', 'clothes', 'outside', 'plants', 'later', 'wait', 'soup', 'attic', 'was', 'john', 'easy'],
				['snack', 'laugh', 'kick', 'famous', 'both', 'off', 'pepper', 'octopus', 'blocks', 'a', 'play', 'boring', 'continued', 'felt', 'everyone', 'bit', 'strange', 'view', 'tv', '<end>']
			],
			probe: [0.0025, 0.0011, 0.0012, 0.0005, 0.006, 0.0025, 0.0003, 0.0327, 0.0039, 0.0009, 0.0018, 0.0044, 0.0007, 0.0005, 0.0231],
			guess: ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
			guessP: [0.0232, 0.0262, 0.0304, 0.029, 0.0227, 0.0274, 0.0294, 0.0327, 0.029, 0.0279, 0.0288, 0.0252, 0.0272, 0.0231, 0.0231],
			probeLoss: 6.1793
		},
		{
			step: 50,
			trainLoss: 5.8998,
			validLoss: 5.8718,
			greedy: [
				['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'they', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'they', '.', '.', 'they', '.', '.', '.'],
				['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'they', '.', '.', '.', '.', '.', '.', 'they', '.', 'they', '.', 'they', '.', '.', 'they', '.'],
				['.', '.', 'they', '.', 'they', '.', 'they', '.', 'they', '.', 'they', '.', 'they', '.', 'they', '.', 'they', '.', 'they', '.', 'they', 'they', '.', 'they', '.', '<end>']
			],
			sampled: [
				['suit', 'spot\'s', 'pot', 'hop', 'can\'t', 'pet', 'toys', 'they', 'beautiful', 'bought', 'bone', 'jealous', 'mouse', '.', 'work', 'rock', 'low', 'to', '<end>'],
				['lots', 'fight', 'big', 'he', 'hugs', 'valuable', ',', 'swings', 'window', 'good', 'time', 'pointed', 'threw', ',', 'triangle', 'purple', 'tray', 'could', 'other', '"', 'table', 'ran', 'fun', 'both', 'scissors', '.', 'toys', 'doll'],
				['angry', 'max', 'storm', 'mimi', 'they', 'were', 'drive', 'ending', 'ball', 'the', 'to', 'push', 'fish', 'a', 'had', 'playing', 'its', 'log', 'tom\'s', '<end>']
			],
			probe: [0.0097, 0.0032, 0.0026, 0.0022, 0.0197, 0.0114, 0.001, 0.1031, 0.0145, 0.0031, 0.0082, 0.0094, 0.001, 0.0016, 0.0597],
			guess: ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
			guessP: [0.053, 0.0579, 0.0404, 0.0607, 0.0622, 0.0615, 0.0652, 0.1031, 0.0227, 0.0575, 0.0539, 0.053, 0.0784, 0.0773, 0.0597],
			probeLoss: 5.0429
		},
		{
			step: 100,
			trainLoss: 4.7801,
			validLoss: 4.7371,
			greedy: [
				['big', 'big', 'big', '.', '"', '"', '"', '"', '"', '"', '"', '"', '"', '"', '"', '"', 'i', ',', '"', '"', '"', '"', '"', '"', '"', '"', '"', '"'],
				['to', 'play', '.', '"', '"', 'i', '.', '"', '"', '"', '"', 'i', ',', '"', '"', '"', '"', '"', '"', '"', 'i', ',', '"', '"', '"', '"', '"', '"'],
				['the', 'bird', '.', 'the', 'bird', '.', 'they', 'was', 'a', 'big', 'big', ',', '"', '"', '"', '"', '"', '"', '"', '"', 'the', 'bird', '.', '"', '"', '"', '"', '"']
			],
			sampled: [
				['spring', 'fun', 'them', 'them', 'named', 'beautiful', 'the', 'bird', 'was', 'called', 'did', 'okay', 'and', 'and', 'him', ',', 'zip', '.', '<end>'],
				['said', 'they', 'she', 'with', 'good', 'saved', '.', 'can', 'ben', 'friends', 'and', 'being', 'of', 'the', 'dependable', 'were', 'potato', ',', 'he', 'with', 'play', 'in', 'the', 'cake', 'up', '.', 'they', 'thank'],
				['cat', 'and', 'will', 'for', 'they', 'with', 'an', 'morning', 'the', 'in', 'the', 'enough', ',', 'and', 'said', ',', 'day', 'splash', 'that', '<end>']
			],
			probe: [0.0722, 0.007, 0.0066, 0.0039, 0.1086, 0.0978, 0.0145, 0.3509, 0.0748, 0.0128, 0.1336, 0.0845, 0.0031, 0.0017, 0.1865],
			guess: ['.', 'the', 'big', '.', 'to', 'the', 'bird', '.', '"', 'was', 'a', 'big', ',', '.', '.'],
			guessP: [0.1138, 0.0387, 0.0274, 0.1955, 0.1086, 0.0978, 0.0291, 0.3509, 0.0802, 0.051, 0.1336, 0.0845, 0.0412, 0.0855, 0.1865],
			probeLoss: 3.5515
		},
		{
			step: 200,
			trainLoss: 3.8417,
			validLoss: 3.8263,
			greedy: [
				['little', 'boy', 'named', 'tim', '.', 'he', 'was', 'a', 'big', ',', 'tim', '.', 'he', 'was', 'very', 'happy', '.', 'he', 'was', 'very', 'happy', '.', 'he', 'was', 'very', 'happy', '.', 'he'],
				['to', 'the', 'park', '.', 'she', 'was', 'very', 'happy', '.', 'she', 'was', 'very', 'happy', '.', 'she', 'was', 'very', 'happy', '.', 'she', 'was', 'very', 'happy', '.', 'she', 'was', 'very', 'happy'],
				['play', 'with', 'the', 'park', '.', 'he', 'was', 'very', 'happy', '.', 'he', 'was', 'very', 'happy', '.', 'he', 'was', 'very', 'happy', '.', 'he', 'was', 'very', 'happy', '.', 'he', 'was', 'very']
			],
			sampled: [
				['wagged', 'boy', 'named', 'i\'m', 'special', 'garden', '.', '"', 'look', 'their', 'bee', 'smiled', 'and', 'said', ',', '"', 'large', ',', 'the', 'alone', 'motorcycle', 'beautiful', 'bird', 'by', '.', 'he', 'wanted', 'to'],
				['to', 'find', 'a', 'big', 'trees', 'tim\'s', 'mom', 'with', 'her', 'friend', '.', 'tim', 'could', 'she', 'called', 'ran', 'nose', ',', 'he', 'was', 'too', '.', 'he', 'wanted', 'to', 'the', 'sorry', 'to'],
				['look', 'and', 'laughed', 'with', 'the', 'park', 'to', 'cut', '.', '.', 'the', 'chair', 'said', ',', '"', 'help', 'on', 'cry', '!', '<end>']
			],
			probe: [0.102, 0.006, 0.0071, 0.0154, 0.2435, 0.3056, 0.0582, 0.5275, 0.0938, 0.04, 0.6852, 0.2441, 0.01, 0.0271, 0.447],
			guess: ['was', 'said', 'mom', '.', 'to', 'the', 'park', '.', 'he', 'were', 'a', 'big', 'tree', 'tree', '.'],
			guessP: [0.1398, 0.1178, 0.1836, 0.1688, 0.2435, 0.3056, 0.0582, 0.5275, 0.1773, 0.0898, 0.6852, 0.2441, 0.0629, 0.0428, 0.447],
			probeLoss: 2.5999
		},
		{
			step: 400,
			trainLoss: 3.3227,
			validLoss: 3.3198,
			greedy: [
				['little', 'girl', 'named', 'lily', '.', 'she', 'loved', 'to', 'play', 'with', 'her', 'toys', 'with', 'her', 'friends', '.', 'one', 'day', ',', 'she', 'saw', 'a', 'big', ',', 'little', 'girl', 'named', 'lily'],
				['to', 'the', 'park', '.', 'she', 'saw', 'a', 'big', ',', 'red', 'ball', 'in', 'the', 'park', '.', 'she', 'was', 'very', 'happy', 'and', 'wanted', 'to', 'play', 'with', 'her', 'friends', '.', 'she'],
				['play', 'with', 'the', 'ball', '.', 'she', 'said', ',', '"', 'i', 'can', 'be', 'careful', ',', 'i', 'will', 'be', 'careful', '.', '"', 'lily', 'said', ',', '"', 'i', 'can', 'be', 'careful']
			],
			sampled: [
				['dirty', 'picture', 'of', 'full', 'of', 'dance', '.', 'the', 'people', 'thought', 'that', 'started', 'to', 'play', 'with', 'it', 'outside', '.', '<end>'],
				['to', 'it', ',', 'a', 'long', 'roof', '.', 'she', 'looked', 'at', 'the', 'bench', 'and', 'said', 'she', 'would', 'goes', 'in', 'the', 'box', '.', 'she', 'saw', 'a', 'mouse', '.', 'she', 'asked'],
				['get', 'the', 'bowl', 'to', 'the', 'box', 'and', 'bigger', '.', 'the', 'little', 'girl', 'saw', 'the', 'bird', 'and', 'wanted', 'to', 'listen', '.', 'the', 'dog', 'laughed', 'soon', 'helped', 'them', 'and', 'played']
			],
			probe: [0.1104, 0.0168, 0.0038, 0.023, 0.1362, 0.5308, 0.1668, 0.5325, 0.4037, 0.0731, 0.581, 0.3948, 0.0257, 0.0734, 0.2056],
			guess: ['was', 'said', 'mom', '.', 'back', 'the', 'park', '.', 'they', 'were', 'a', 'big', ',', 'ball', 'in'],
			guessP: [0.2869, 0.168, 0.2291, 0.3037, 0.1589, 0.5308, 0.1668, 0.5325, 0.4037, 0.118, 0.581, 0.3948, 0.0782, 0.0734, 0.2236],
			probeLoss: 2.2353
		},
		{
			step: 800,
			trainLoss: 3.0145,
			validLoss: 3.0328,
			greedy: [
				['little', 'girl', 'named', 'lily', '.', 'she', 'loved', 'to', 'play', 'with', 'her', 'friends', '.', 'one', 'day', ',', 'she', 'saw', 'a', 'big', ',', 'red', 'ball', '.', 'she', 'wanted', 'to', 'play'],
				['to', 'the', 'park', '.', 'she', 'saw', 'a', 'big', ',', 'red', 'ball', '.', 'she', 'wanted', 'to', 'play', 'with', 'her', 'friends', ',', 'but', 'she', 'could', 'not', 'find', 'it', '.', 'she'],
				['play', 'with', 'her', 'friends', '.', 'she', 'said', ',', '"', 'i', 'want', 'to', 'play', 'with', 'you', '.', '"', 'the', 'little', 'girl', 'smiled', 'and', 'said', ',', '"', 'yes', ',', 'tim']
			],
			sampled: [
				['cute', 'man', 'named', 'spot', '.', 'max', 'loved', 'to', 'everyone', 'every', 'day', ',', 'his', 'mom', 'and', 'dad', 'as', 'they', ',', 'she', 'stretched', 'if', 'they', 'showed', 'the', 'broken', 'place', 'to'],
				['to', 'a', 'time', ',', 'there', 'found', 'a', 'tree', 'and', 'she', 'had', 'long', 'colors', '.', 'lucy', 'went', 'by', 'her', 'friend', ',', 'a', 'little', 'girl', 'named', 'lucy', '.', 'she', 'loved'],
				['make', 'a', 'sweet', 'butterfly', '.', 'the', 'boat', 'lived', 'in', 'the', 'tree', 'and', 'had', 'a', 'big', 'box', '.', 'inside', 'the', ',', 'the', 'box', 'made', 'bright', 'painting', 'of', 'the', 'forest']
			],
			probe: [0.1816, 0.0327, 0.0409, 0.058, 0.3256, 0.5456, 0.2519, 0.6932, 0.3264, 0.2413, 0.6435, 0.5248, 0.0265, 0.4146, 0.2957],
			guess: ['and', 'lily', 'mom', '.', 'to', 'the', 'park', '.', 'they', 'saw', 'a', 'big', ',', 'ball', '.'],
			guessP: [0.1816, 0.2002, 0.3927, 0.2379, 0.3256, 0.5456, 0.2519, 0.6932, 0.3264, 0.2413, 0.6435, 0.5248, 0.3035, 0.4146, 0.2957],
			probeLoss: 1.6001
		},
		{
			step: 1600,
			trainLoss: 2.796,
			validLoss: 2.8439,
			greedy: [
				['little', 'boy', 'named', 'tim', '.', 'tim', 'loved', 'to', 'play', 'with', 'his', 'friends', '.', 'one', 'day', ',', 'he', 'saw', 'a', 'big', 'tree', 'with', 'a', 'big', 'tree', '.', 'he', 'wanted'],
				['to', 'the', 'park', 'to', 'play', 'with', 'her', 'mom', '.', 'they', 'saw', 'a', 'big', 'tree', 'with', 'a', 'big', ',', 'red', 'ball', '.', 'they', 'were', 'very', 'happy', 'and', 'played', 'together'],
				['play', 'with', 'her', 'friends', '.', 'she', 'said', ',', '"', 'thank', 'you', ',', 'mom', ',', 'for', 'helping', 'me', '.', '"', 'the', 'little', 'girl', 'was', 'happy', 'and', 'said', ',', '"']
			],
			sampled: [
				['huge', 'chair', ',', 'who', 'loved', 'to', 'go', 'to', 'the', 'road', '.', 'one', 'day', ',', 'he', 'saw', 'tom', 'and', 'the', 'window', 'rise', 'together', '.', 'when', 'the', 'rock', 'was', 'done'],
				['to', 'the', 'park', 'and', 'took', 'turns', '.', 'she', 'loved', 'to', 'play', 'in', 'the', 'park', 'when', 'she', 'feels', 'happy', '.', 'she', 'put', 'it', 'in', 'the', 'lake', ',', 'and', 'tom'],
				['see', 'what', 'she', 'could', 'have', 'a', 'secret', 'ride', '.', 'the', 'little', 'girl', 'said', ',', '"', 'let\'s', 'make', 'sure', 'of', 'the', 'special', 'flowers', 'too', 'did', 'wear', 'more', 'toys', '.']
			],
			probe: [0.0878, 0.0393, 0.006, 0.0226, 0.4972, 0.4939, 0.1703, 0.5379, 0.6528, 0.4576, 0.5483, 0.5859, 0.0313, 0.4392, 0.4164],
			guess: ['was', 'lily', 'mom', '.', 'to', 'the', 'park', '.', 'they', 'saw', 'a', 'big', 'tree', 'ball', '.'],
			guessP: [0.187, 0.2409, 0.378, 0.2304, 0.4972, 0.4939, 0.1703, 0.5379, 0.6528, 0.4576, 0.5483, 0.5859, 0.212, 0.4392, 0.4164],
			probeLoss: 1.7247
		},
		{
			step: 3200,
			trainLoss: 2.6908,
			validLoss: 2.6985,
			greedy: [
				['little', 'girl', 'named', 'lily', '.', 'she', 'loved', 'to', 'play', 'with', 'her', 'toys', '.', 'one', 'day', ',', 'she', 'found', 'a', 'big', 'box', 'of', 'toys', 'in', 'a', 'big', 'box', '.'],
				['to', 'the', 'park', 'with', 'her', 'mom', '.', 'she', 'saw', 'a', 'big', 'tree', 'with', 'a', 'big', ',', 'red', 'ball', 'on', 'it', '.', 'lily', 'wanted', 'to', 'help', 'her', '.', 'she'],
				['help', 'the', 'girl', '.', 'she', 'asked', 'her', 'mom', ',', '"', 'can', 'i', 'play', 'with', 'the', 'toy', 'car', '?', '"', 'her', 'mom', 'said', ',', '"', 'yes', ',', 'but', 'be']
			],
			sampled: [
				['powerful', 'lion', 'named', 'mark', '.', 'mark', 'loved', 'to', 'fly', 'high', 'in', 'the', 'flowers', '.', 'one', 'day', ',', 'he', 'and', 'his', 'brothers', 'decided', 'to', 'spin', '.', 'they', 'found', 'a'],
				['to', 'the', 'park', '.', 'she', 'showed', 'the', 'swing', 'to', 'her', 'mom', 'and', 'dad', '.', 'lily', 'liked', 'lily', 'and', 'tom', '.', 'she', 'wanted', 'to', 'see', 'what', 'the', 'sun', 'was'],
				['find', 'a', 'bit', 'scared', '.', 'he', 'decided', 'to', 'put', 'the', 'cat', 'back', 'in', 'the', 'park', '.', 'he', 'dropped', 'the', 'big', 'ball', 'and', 'enjoyed', 'rest', 'by', 'all', 'the', 'home']
			],
			probe: [0.1328, 0.0491, 0.0677, 0.0152, 0.5299, 0.3519, 0.255, 0.5383, 0.7724, 0.5987, 0.7641, 0.6795, 0.0216, 0.6842, 0.2766],
			guess: ['and', 'lily', 'mom', '.', 'to', 'the', 'park', '.', 'they', 'saw', 'a', 'big', 'tree', 'ball', '.'],
			guessP: [0.1328, 0.1654, 0.3772, 0.3922, 0.5299, 0.3519, 0.255, 0.5383, 0.7724, 0.5987, 0.7641, 0.6795, 0.2325, 0.6842, 0.2766],
			probeLoss: 1.5005
		},
		{
			step: 6400,
			trainLoss: 2.5072,
			validLoss: 2.5702,
			greedy: [
				['little', 'girl', 'named', 'lily', '.', 'she', 'loved', 'to', 'play', 'with', 'her', 'toys', 'and', 'play', 'with', 'her', 'toys', '.', 'one', 'day', ',', 'she', 'found', 'a', 'big', 'box', 'in', 'her'],
				['to', 'the', 'park', 'to', 'play', 'with', 'her', 'toys', '.', 'she', 'saw', 'a', 'big', 'tree', 'with', 'a', 'big', 'tree', '.', 'she', 'wanted', 'to', 'climb', 'the', 'tree', ',', 'but', 'she'],
				['help', 'the', 'bird', '.', 'she', 'said', ',', '"', 'i', 'will', 'help', 'you', 'find', 'my', 'way', 'home', '.', '"', 'the', 'bird', 'was', 'very', 'happy', 'and', 'said', ',', '"', 'thank']
			],
			sampled: [
				['furry', 'lion', 'named', 'fred', '.', 'fred', 'had', 'a', 'long', 'nose', 'that', 'he', 'loved', 'to', 'talk', 'and', 'dream', '.', 'the', 'monkey', 'often', 'lay', 'on', 'an', 'of', 'beautiful', 'voice', 'and'],
				['to', 'the', 'park', 'to', 'play', 'outside', '.', 'she', 'saw', 'a', 'big', 'hill', 'with', 'a', 'flag', 'on', 'its', 'ground', '.', '"', 'hello', ',', 'little', 'fish', '!', '"', 'said', 'lily'],
				['find', 'a', 'smile', '.', 'the', 'little', 'girl', 'laughed', 'and', 'said', ',', '"', 'no', ',', 'that', 'is', 'not', 'fake', '.', 'the', 'new', 'kids', 'will', 'forgive', 'us', '.', '"', 'the']
			],
			probe: [0.1564, 0.0397, 0.076, 0.0378, 0.4172, 0.5018, 0.3259, 0.6618, 0.7025, 0.57, 0.713, 0.5914, 0.0651, 0.7196, 0.3701],
			guess: ['and', 'sue', 'mom', ',', 'to', 'the', 'park', '.', 'they', 'saw', 'a', 'big', 'tree', 'ball', '.'],
			guessP: [0.1564, 0.2268, 0.4427, 0.3991, 0.4172, 0.5018, 0.3259, 0.6618, 0.7025, 0.57, 0.713, 0.5914, 0.3369, 0.7196, 0.3701],
			probeLoss: 1.3243
		},
		{
			step: 12000,
			trainLoss: 2.4311,
			validLoss: 2.4721,
			greedy: [
				['little', 'girl', 'named', 'lily', '.', 'she', 'loved', 'to', 'play', 'with', 'her', 'toys', 'and', 'eat', 'yummy', 'food', '.', 'one', 'day', ',', 'she', 'found', 'a', 'big', 'box', 'of', 'toys', '.'],
				['to', 'the', 'park', 'with', 'her', 'mom', '.', 'she', 'saw', 'a', 'big', 'tree', 'with', 'a', 'big', 'tree', '.', 'lily', 'wanted', 'to', 'climb', 'the', 'tree', '.', 'she', 'asked', 'her', 'mom'],
				['help', 'the', 'little', 'girl', '.', 'she', 'asked', 'her', 'mom', ',', '"', 'can', 'i', 'help', 'you', 'find', 'my', 'way', 'home', '?', '"', 'her', 'mom', 'said', ',', '"', 'yes', ',']
			],
			sampled: [
				['year', 'old', 'man', 'who', 'loved', 'to', 'take', 'things', '.', 'one', 'day', ',', 'he', 'was', 'walking', 'up', 'as', 'he', 'was', 'walking', 'past', 'an', 'apple', 'first', '.', 'he', 'found', 'a'],
				['to', 'the', 'park', '.', 'she', 'met', 'a', 'boy', 'named', 'tim', '.', 'tim', 'had', 'a', 'favorite', 'idea', 'to', 'play', 'with', 'his', 'toy', 'ball', '.', 'he', 'took', 'the', 'ball', 'with'],
				['see', 'the', 'promise', 'and', 'said', ',', '"', 'please', ',', 'mom', '.', 'i', 'will', 'help', 'you', '.', '"', 'lucy', 'asked', '.', 'she', 'wanted', 'to', 'feed', 'put', 'on', 'his', 'head']
			],
			probe: [0.1342, 0.0655, 0.1074, 0.0079, 0.5332, 0.6785, 0.5547, 0.8094, 0.7285, 0.5126, 0.6894, 0.582, 0.0654, 0.6892, 0.4541],
			guess: ['was', 'lily', 'mom', ',', 'to', 'the', 'park', '.', 'they', 'saw', 'a', 'big', 'tree', 'ball', '.'],
			guessP: [0.1374, 0.192, 0.3073, 0.4087, 0.5332, 0.6785, 0.5547, 0.8094, 0.7285, 0.5126, 0.6894, 0.582, 0.2203, 0.6892, 0.4541],
			probeLoss: 1.2945
		}
	]
};

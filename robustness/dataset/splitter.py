import random
from itertools import chain

def split_indices(df, window_size, overlap, windows_per_chunk=100, 
				train_split=0.8, val_split=0.1, test_split=0.1,
				start_seed = 42
				) -> tuple[list[int], list[int], list[int]]:
	"""
	Split dataframe indices using chunk-based strategy.

	Args:
		df: Cleaned pandas DataFrame
		window_size: Size of sliding window
		overlap: Overlap fraction (0.0-0.99)
		windows_per_chunk: Number of windows per chunk
		train_split, val_split, test_split: Split ratios
		random_seed: Random seed for reproducibility

	Returns:
		tuple: (train_indices, val_indices, test_indices)
	"""
	random_seed = start_seed

	random.seed(random_seed)

	stride = max(1, int(window_size * (1 - overlap)))
	total_points = len(df)
	total_windows = (total_points - window_size) // stride + 1
	tot_chunks = total_windows // windows_per_chunk

	data_indices = df.index.tolist()
	test_points = test_split * total_points
	val_points = val_split * total_points

	# Test set
	test_indices = data_indices[-int(test_points):]
	train_val_indices = data_indices[:-int(test_points)]

	train_val_block_chunks = [
		train_val_indices[i:i + tot_chunks] 
		for i in range(0, len(train_val_indices), tot_chunks)
	]

	# Adjusted original validation set percentage
	sample_percentage = round(val_points/len(train_val_indices), 2)

	# List of all block chunks indices
	block_chunks_indices = list(range(0, len(train_val_block_chunks)))

	# Validation set
	val_block_chunks_indices = sorted(random.sample(
			block_chunks_indices,
			k = int(len(train_val_block_chunks) * sample_percentage)
		)
	)
	val_dataset_blocks_indices = [
		train_val_block_chunks[i] for i in val_block_chunks_indices
	]

	val_indices = list(chain.from_iterable(val_dataset_blocks_indices))

	# Train set
	train_block_chunks_indices = [
		x for x in block_chunks_indices 
		if x not in set(val_block_chunks_indices)
	]

	train_dataset_blocks_indices = [
		train_val_block_chunks[i] for i in train_block_chunks_indices
	]

	train_indices = list(
		chain.from_iterable(train_dataset_blocks_indices)
	)
	
	return train_indices, val_indices, test_indices
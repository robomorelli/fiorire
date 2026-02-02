import torch
import numpy as np

from torch.utils.data import Dataset

class SignalWindowDataset(Dataset):
	"""Memory-efficient dataset that creates windows on-the-fly"""

	def __init__(self, std_data, window_size=16, overlap=0.0, n_signals=19):
		"""
		Args:
			std_data: Standardized pandas DataFrame
			window_size: Number of time points per window (default: 16)
			overlap: Overlap fraction between windows, 0.0 to 0.99 (default: 0.0)
			n_signals: Number of signal channels (default: 19)
		"""

		self.data = std_data
		# TODO: handle case where window_size is greater than data.shape[0]
		self.window_size = window_size
		self.n_signals = n_signals

		# Calculate stride based on overlap
		# TODO: handle case where overlap is 0
		# TODO: handle case where overlap is greater than 1
		# TODO: handle case where overlap is negative
		self.stride = max(1, int(window_size * (1 - overlap)))

		# Calculate number of windows
		self.n_windows = (
			(self.data.shape[0] - window_size) // self.stride
			) + 1

	def __len__(self):
		return self.n_windows

	def __getitem__(self, idx):
		start = idx * self.stride
		end = start + self.window_size
		window = self.data[start:end, :].T
		window = window[np.newaxis, :, :]
		return torch.from_numpy(window)
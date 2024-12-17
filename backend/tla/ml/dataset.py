import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path):
        # Load the data
        data = np.load(data_path)
        print(f"Available keys in the .npz file: {data.files}")

        # Define class objects
        self.images = data['image']
        self.labels = data['label']

        # Debugging: Check shapes and values
        print(f"Shape of images: {self.images.shape}")
        print(f"Shape of labels: {self.labels.shape}")
        print(f"Unique labels: {np.unique(self.labels)}")  # Check unique values in labels

    def __len__(self):
        # Define number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Define feature and target
        x_temp = self.images[idx]
        y_temp = self.labels[idx]

        # Convert to torch tensors
        x = torch.tensor(x_temp, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y_temp, dtype=torch.int64)

        return x, y

import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, data, grid):
        self.targets = data[:, 1, ...]

        grids = grid.repeat(data.shape[0], 1, 1)
        self.inputs = torch.concat((data[:, 0, ...], grids), 1)  # Adding grid/mesh channel to input

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
    

class KSHerDataset(Dataset):
    def __init__(self, N, data, grid):
        self.targets = data[:, 1, ...]
        data[:, :, :, 1::2] *= N/10

        grids = grid.repeat(data.shape[0], 1, 1)
        self.inputs = torch.concat((data[:, 0, ...], grids), 1)  # Adding grid/mesh channel to input

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


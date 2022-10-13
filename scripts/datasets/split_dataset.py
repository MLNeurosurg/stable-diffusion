import torch
from torch.utils.data import Dataset


# https://discuss.pytorch.org/t/using-imagefolder-random-split-with-multiple-transforms/79899
class SplitDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.targets = [
            self.subset.dataset.targets[i] for i in self.subset.indices
        ]
        self.classes = self.subset.dataset.classes

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.subset[index][0])
        else:
            x = self.subset[index][0]
        y = self.subset[index][1]
        return x, y

    def __len__(self):
        return len(self.subset)
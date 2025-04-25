# Custom dataset wrapper
from torch.utils.data import Dataset

class TorchDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform_fn):
        self.dataset = hf_dataset
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform_fn(self.dataset[idx])
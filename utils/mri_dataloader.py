import torch
import numpy as np
 #write data loader for brain data
from torch.utils.data import Dataset, DataLoader
import os
import glob

class fastMRI_brain(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = np.load(self.file_list[idx])
        # cast img as type double
        img = img.astype(np.float32)
        img = torch.from_numpy(img)

        return img #img here should be a torch tensor of shape (2, img_size, img_size)
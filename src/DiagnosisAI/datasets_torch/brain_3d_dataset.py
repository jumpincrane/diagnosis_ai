import torch as t
from pathlib import Path
from torchvision.transforms import Compose, ToTensor
import numpy as np
import pickle
import math
import nibabel as nib
from scipy.ndimage import zoom
import albumentations as A

class Brain3DDataset(t.utils.data.Dataset):
    def __init__(self, sub_folder_names: list):
        # self.root_path = Path("/mnt/e/mgr/datasets/brain/Brats2021_training_df")
        self.sub_folder_names = sub_folder_names
        self.image_size = (240, 240, 155)
        self.scales = [64 / size for size in self.image_size]

    def __getitem__(self, idx):
        case_folder_path = Path(self.sub_folder_names[idx])
        for filename in case_folder_path.iterdir():
            if 'flair' in filename.name:
                flair = nib.load(filename).get_fdata()
                flair = zoom(flair, self.scales).astype(np.float32)
            if 't1' in filename.name:
                t1 = nib.load(filename).get_fdata()
                t1 = zoom(t1, self.scales).astype(np.float32)
            if 't2' in filename.name:
                t2 = nib.load(filename).get_fdata()
                t2 = zoom(t2, self.scales).astype(np.float32)
            if 't1ce' in filename.name:
                t1ce = nib.load(filename).get_fdata()
                t1ce = zoom(t1ce, self.scales).astype(np.float32)
            if 'seg' in filename.name:
                seg = nib.load(filename).get_fdata()
                seg = zoom(seg, self.scales).astype(np.float32)

        # stack them into 4 channel input
        img = np.stack([flair, t1, t2, t1ce], axis=0)

        # extract segment mask, 1 channel with 4 classes
        seg = seg[np.newaxis, ...]
        seg = np.where(seg >= 1, 1, 0)

        return t.from_numpy(img), t.from_numpy(seg)

    def __len__(self):
        return len(self.sub_folder_names)
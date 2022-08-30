import torch as t
from pathlib import Path
from torchvision.transforms import Compose
import numpy as np
import pickle
import math
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.pytorch.transforms import ToTensor

class BrainSlicesDataset(t.utils.data.Dataset):
    def __init__(self, file_names: list, binary_mask: bool = True, t1ce_only: bool = False):
        self.root_path = Path("/mnt/e/mgr/datasets/brain/train_images_max_area")
        self.file_names = file_names
        self.image_size = (240, 240)
        self.binary_mask = binary_mask
        self.t1ce_only = t1ce_only
        self.padded_image_size = (
                                    math.ceil(self.image_size[0] / 32) * 32,
                                    math.ceil(self.image_size[1] / 32) * 32
                                 )
        self.transforms = A.Compose([A.PadIfNeeded(*self.padded_image_size),
                                     A.ToFloat(max_value=255),
                                     ToTensorV2(transpose_mask=True)])

    def __getitem__(self, idx):
        slice_pickle_path = self.root_path / self.file_names[idx]
        with open(slice_pickle_path, 'rb') as handle:
            slice_data = pickle.load(handle)
        # 4 core images
        img_flair = slice_data['flair'].astype(np.float32).squeeze()
        img_t1 = slice_data['t1'].astype(np.float32).squeeze()
        img_t1ce = slice_data['t1ce'].astype(np.float32).squeeze()
        img_t2 = slice_data['t2'].astype(np.float32).squeeze()

        # stack them into 4 channel input
        img = np.stack([img_flair, img_t1, img_t1ce, img_t2], axis=2)

        if self.t1ce_only:
            img = img_t1ce[..., np.newaxis]

        # extract segment mask, 1 channel with 4 classes
        label = slice_data['seg']
        label = label[:, :, np.newaxis]

        # convert to binary mask
        if self.binary_mask:
            label = np.where(label >= 1, 1, 0)
        else:
            # to onehot mask
            label = np.where(label == 0, 0, label)
            label = np.where(label == 1, 1, label)
            label = np.where(label == 2, 2, label)
            label = np.where(label == 4, 3, label)
            label = label.astype(np.int32)
            label = np.stack([label == i for i in range(label.max()+1)], axis=2)
            label = label.astype(np.int32)

        transformed = self.transforms(image=img, mask=label)

        return transformed['image'], transformed['mask'].type(t.float32)

    def __len__(self):
        return len(self.file_names)
from pathlib import Path
import pickle
import math
from typing import List

import numpy as np
import torch as t
import torchvision
import nibabel as nib
from scipy.ndimage import zoom
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image


class Brain3DDataset(t.utils.data.Dataset):
    def __init__(self, sub_folder_names: List[str], t1ce: bool = False, scale: int = 32):

        self.sub_folder_names = sub_folder_names
        self.image_size = (240, 240, 155)
        self.scales = [scale / size for size in self.image_size]
        self.t1ce = t1ce

    def __getitem__(self, idx):
        case_folder_path = Path(self.sub_folder_names[idx])

        for filename in case_folder_path.iterdir():
            if not self.t1ce:
                if 'flair' in filename.name:
                    flair = nib.load(filename).get_fdata()
                    flair = zoom(flair, self.scales).astype(np.float32)
                if 't1' in filename.name:
                    t1 = nib.load(filename).get_fdata()
                    t1 = zoom(t1, self.scales).astype(np.float32)
                if 't2' in filename.name:
                    t2 = nib.load(filename).get_fdata()
                    t2 = zoom(t2, self.scales).astype(np.float32)
            if 'seg' in filename.name:
                seg = nib.load(filename).get_fdata()
                seg = zoom(seg, self.scales).astype(np.float32)
            if 't1ce' in filename.name:
                t1ce = nib.load(filename).get_fdata()
                t1ce = zoom(t1ce, self.scales).astype(np.float32)

        # stack them into 4 channel input
        if not self.t1ce:
            img = np.stack([flair, t1, t2, t1ce], axis=0)
        else:
            img = np.stack([t1ce], axis=0)

        # extract segment mask, 1 channel with 4 classes
        seg = seg[np.newaxis, ...]
        seg = np.where(seg >= 1, 1, 0)

        return t.from_numpy(img), t.from_numpy(seg)

    def __len__(self):
        return len(self.sub_folder_names)


class BrainSlicesDataset(t.utils.data.Dataset):
    def __init__(self, file_names: List[str], binary_mask: bool = True, t1ce_only: bool = False):

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


class ChestDataset(t.utils.data.Dataset):
    def __init__(self, filenames: List[str], labels: List[int]):
        self.filenames = filenames
        self.labels = labels
        self.convert_to_tensor = torchvision.transforms.Compose([torchvision.transforms.Resize([224, 224]),
                                                    torchvision.transforms.ToTensor()])

    def __getitem__(self, idx): # indeksowanie
        filename = self.filenames[idx]
        label = self.labels[idx]

        img = Image.open(filename).convert('L') #L - gray image


        return self.convert_to_tensor(img), t.tensor(label, dtype=t.int32)

    def __len__(self): # nadpisanie metody
        return len(self.filenames)
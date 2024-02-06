from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
import nibabel as nib
from scipy.ndimage import zoom
from PIL import Image

from diagnosisai.utils.data_processing import get_plane_where_tumor_is_thelargest

# TODO: docstrings, chestdataset options to check grayscales, other channels


class BrainDataset(Dataset):
    """
    The dataset will work correctly by providing a folder with subfolders for each case,
    and each case will contain specific scans and a tumor mask for them.

    :param str dataset_dir: path to directory where is dataset located,
    :param Optional[tuple[int]] scan_size: to what size it should scale the original image size,
    :param bool only_the_largest_slice: extract only planes where tumor is the largest,
    :param bool binary_seg: if binary_seg is passed all labels are concatenate to one class.
    """
    def __init__(self,
                 dataset_dir: str,
                 to_scale_size: Optional[tuple[int]] = None,
                 only_the_largest_slice: bool = False,
                 binary_seg: bool = True):

        self.dataset_dir = Path(dataset_dir)
        self.cases_dirs = [case_dir for case_dir in self.dataset_dir.iterdir()]
        self.to_scale_size = to_scale_size
        self.binary_seg = binary_seg

        self.classes = {0: 0, 1: 1, 2: 2, 3: 4}

        self.only_the_largest_slice = only_the_largest_slice

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:

        for filename in self.cases_dirs[idx].iterdir():
            if 'flair' in filename.name:
                flair = self.__load_nib(filename)
            if 't1' in filename.name:
                t1 = self.__load_nib(filename)
            if 't2' in filename.name:
                t2 = self.__load_nib(filename)
            if 'seg' in filename.name:
                label = self.__load_nib(filename)
            if 't1ce' in filename.name:
                t1ce = self.__load_nib(filename)

        img = np.stack([flair, t1, t2, t1ce], axis=0)
        img, label = torch.from_numpy(img), torch.from_numpy(label)

        if self.only_the_largest_slice:
            the_largest_tumor_plane = get_plane_where_tumor_is_thelargest(label)
            img = img[..., the_largest_tumor_plane]
            label = label[..., the_largest_tumor_plane]

        if self.binary_seg:
            label = torch.where(label >= 1, 1, 0)
            label = label[None, ...].float()
        else:
            for replace_val, orig_label in self.classes.items():
                label[label == orig_label] = replace_val
            label = label.long()

        return img.float(), label

    def __len__(self):
        return len(self.cases_dirs)

    def __load_nib(self, scan_path: str, force_not_scale: bool = False):
        scan = nib.load(scan_path).get_fdata()

        if self.to_scale_size and not force_not_scale:
            scales = np.array(self.to_scale_size) / scan.shape
            scan = zoom(scan, scales, order=0).astype(np.float32)

        return scan


class ChestDataset(Dataset):
    """
    Dataset folder should contains images splitted in subdirectiories, each subdirectory is a class.
    Files should be black and white x-ray scans of the lungs.

    :param str dataset_dir: The directory containing the dataset.
    :param tuple[int, int] resize_shape: A tuple specifying the desired shape for resizing images.
            Defaults to (224, 224).
    """

    def __init__(self, dataset_dir: str, resize_shape: tuple[int] = (224, 224)):

        self.dataset_dir = Path(dataset_dir)
        self._classes_dirs = [case_dir for case_dir in self.dataset_dir.iterdir()]
        self.classes_info = {class_name.name: label for label, class_name in enumerate(self._classes_dirs)}
        
        self._all_files = [(img_path, class_dir.name) for class_dir in self._classes_dirs
                           for img_path in class_dir.iterdir()]

        self.convert_to_tensor = Compose([Resize(resize_shape), ToTensor()])

    def __getitem__(self, idx: int):  # indeksowanie
        img_path, class_name = self._all_files[idx]

        label = torch.tensor(self.classes_info[class_name])

        img = Image.open(img_path).convert('L')  # L - gray image
        img = self.convert_to_tensor(img)

        return img, label

    def __len__(self):
        return len(self._all_files)

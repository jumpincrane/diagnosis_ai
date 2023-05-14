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
    :param tuple[int, int, int] scan_size: [Height, Width, Number of planes],
    :param int scale: to what size it should scale the original image size,
    :param bool only_the_largest_slice: extract only planes where tumor is the largest
    """

    def __init__(self,
                 dataset_dir: str,
                 scan_size: tuple[int, int, int] = (240, 240, 155),
                 scale: Optional[int] = None,
                 only_the_largest_slice: bool = False):

        self.dataset_dir = Path(dataset_dir)
        self.cases_dirs = [case_dir for case_dir in self.dataset_dir.iterdir()]

        self.scan_size = scan_size
        self.scales = [scale / size for size in self.scan_size] if scale else None
        self.only_the_largest_slice = only_the_largest_slice

    def __getitem__(self, idx: int):

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
        label = np.where(label >= 1, 1, 0)

        img, label = torch.from_numpy(img), torch.from_numpy(label)

        if self.only_the_largest_slice:
            the_largest_tumor_plane = get_plane_where_tumor_is_thelargest(label)
            print(the_largest_tumor_plane)
            img = img[..., the_largest_tumor_plane]
            label = label[..., the_largest_tumor_plane]

        return img, label[None, ...]

    def __len__(self):
        return len(self.cases_dirs)

    def __load_nib(self, scan_path: str):
        scan = nib.load(scan_path).get_fdata()
        if self.scales:
            scan = zoom(scan, self.scales).astype(np.float32)

        return scan


class ChestDataset(Dataset):
    """
    Dataset folder should contain images splitted in subdirectiories, each subdirectory is a class.
    Files should be black and white x-ray scans of the lungs.
    """

    def __init__(self, dataset_dir: str, resize_shape: tuple[int, int] = (224, 224)):

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

from pathlib import Path
import gc
import pickle

import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm


def get_plane_where_tumor_is_thelargest(brain_scan_label: torch.Tensor) -> int:
    """
    From given 2D planes stacked into 3D brain scan, get index of surface where tumor is the largest.
    """

    return brain_scan_label.count_nonzero(dim=(0, 1)).argmax().item()


def get_thelargest_tumor_planes_in_dir(segments_folders: str, save_folder: str):
    """
    For the given folder where there are 3D scans (in nii.gz format) that consist of superimposed planes.
    For each scan extract the plane where the tumour is largest and save these scans to folder.
    """

    path = Path(segments_folders)
    path_save = Path(save_folder)

    for dir in tqdm(path.iterdir()):
        path_dir = path_save / dir.name
        try:
            path_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Folder is already there")
        else:
            print("Folder was created")

        case = {}

        for file in dir.iterdir():
            sample = nib.load(file).get_fdata().copy()
            sample = torch.from_numpy(sample)

            if 'flair' in file.name:
                case['flair'] = sample
            elif 't1ce' in file.name:
                case['t1ce'] = sample
            elif 'seg' in file.name:
                case['seg'] = sample
            elif 't1' in file.name:
                case['t1'] = sample
            elif 't2' in file.name:
                case['t2'] = sample

            del sample
            gc.collect()

        largest_tumor_slice = get_plane_where_tumor_is_thelargest(case['seg'])
        for img in case:
            case[img] = case[img][:, :, largest_tumor_slice]

        pickle.dump(case, open(path_dir / f'{dir.name}_slices_dict.pickle', 'wb'))

        del case
        gc.collect()


# def crop_mask_to_segment(output, label):
#     idx_x = np.where(label == 1)[0]
#     idx_y = np.where(label == 1)[1]

#     masked_label = torch.Tensor(label[idx_x, idx_y]).type(torch.int32)
#     masked_output = torch.Tensor(output[idx_x, idx_y])

#     return masked_output, masked_label
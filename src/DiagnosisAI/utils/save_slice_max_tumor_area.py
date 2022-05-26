from genericpath import sameopenfile
import nipy
from nibabel.testing import data_path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
from PIL import Image
import pickle
import gc
import tqdm


def get_max_area_slice(file_seg):
    segments = file_seg
    tumor_info = {1: {"class_idx": 1},
                2: {"class_idx": 2},
                3: {"class_idx": 4}}

    arr = np.zeros((155, ))
    for i in range(155):
        slice = segments[:, :, i]
        slice_areas = []
        slice_areas = np.zeros(3)
        for j, tumor_seg in enumerate(tumor_info):
            segment_info = tumor_info[tumor_seg]

            seg_mask = (slice == segment_info['class_idx']).nonzero()
            seg_x, seg_y = seg_mask

            seg_area = seg_x.shape[0] * seg_y.shape[0]
            slice_areas[j] = seg_area
        arr[i] = slice_areas.sum()

    return arr.argmax()


path = pathlib.Path('datasets/brain/Brats2021_training_df')
path_save = pathlib.Path('datasets/brain/train_images/')

for dir in tqdm.tqdm(path.iterdir()):
    case = {}
    path_dir = path_save / dir.name
    try:
        path_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("Folder was created")

    for file in dir.iterdir():
        sample = nib.load(file).get_fdata().copy()
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

    max_area_idx = get_max_area_slice(case['seg'])
    for img in case:
        case[img] = case[img][:, :, max_area_idx]
    
    pickle.dump(case, open(path_dir / f'{dir.name}_slices_dict.pickle', 'wb'))

    del case
    gc.collect()


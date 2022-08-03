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
from DiagnosisAI.utils.utils_function import get_max_area_slice


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


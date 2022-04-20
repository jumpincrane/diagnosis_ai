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
import tqdm

path = pathlib.Path('./Brats2021_training_df/')
path_save = pathlib.Path('./train_images/')
brain_all = []
for dir in tqdm.tqdm(path.iterdir()):

    brain_sample = {}
    path_dir = path_save / dir.name
    try:
        path_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("Folder was created")

    for file in dir.iterdir():

        sample = nib.load(file).get_fdata()[:, :, 80:100].astype(np.float16)
        if 'flair' in file.name:
            brain_sample['flair'] = sample
        elif 't1ce' in file.name:
            brain_sample['t1ce'] = sample
        elif 'seg' in file.name:
            brain_sample['seg'] = sample
        elif 't1' in file.name:
            brain_sample['t1'] = sample
        elif 't2' in file.name:
            brain_sample['t2'] = sample

    pickle.dump(brain_sample, open(path_dir / f'{dir.name}_slices_dict.pickle', 'wb'))
        # 75 105
        # for i in range(75, 105):
            # sample = nib.load(file).get_fdata()[i]
            # img = Image.fromarray(sample)
            # img.save(path_dir / f"{file.name[:-7]}_{i}.tiff")

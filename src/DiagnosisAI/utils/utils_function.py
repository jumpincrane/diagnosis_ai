import numpy as np
import torch as t


def crop_mask_to_segment(output, label):
    idx_x = np.where(label == 1)[0]
    idx_y = np.where(label == 1)[1]

    masked_label = t.Tensor(label[idx_x, idx_y]).type(t.int32)
    masked_output = t.Tensor(output[idx_x, idx_y])

    return masked_output, masked_label

def transform_to_metrics(output, label):
    transformed_output = t.Tensor(output)
    transformed_label = t.Tensor(label).type(t.int32)

    return transformed_output, transformed_label

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
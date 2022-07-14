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
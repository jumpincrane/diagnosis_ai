import pickle
import gc
from pathlib import Path

import nibabel as nib
import tqdm
import torch
import numpy as np
from torchmetrics.functional import stat_scores


def calculate_type_errors(
        pred: torch.Tensor, target: torch.Tensor, mode: str = 'segment_binary', num_classes: int = None):
    """
    :param str mode: [segment_binary, segment_semantic, classif_binary, classif_multiclass],
    :param int num_classes: pass num_clasess only to classif_multiclass,

    :return tp, fp, tn, fn:.

    Note: If we want to have global(no parameters per class) confusion matrix parameters we have to add all values across parameter tp.sum(), fp.sum() etc.,
          also after summing up all confusion matrix parameters per batch for an epoch just calculate metrics.
    """

    def __get_conf_atribs__(conf_matrix, num_classes):
        tp_indicies = torch.full((num_classes, 1), 0)
        fp_indicies = torch.full((num_classes, 1), 1)
        tn_indicies = torch.full((num_classes, 1), 2)
        fn_indicies = torch.full((num_classes, 1), 3)

        tp = torch.take_along_dim(conf_matrix, tp_indicies, dim=1)
        fp = torch.take_along_dim(conf_matrix, fp_indicies, dim=1)
        tn = torch.take_along_dim(conf_matrix, tn_indicies, dim=1)
        fn = torch.take_along_dim(conf_matrix, fn_indicies, dim=1)

        return tp, fp, tn, fn

    if mode == "segment_binary":
        # calculate global metrics for all channels, classes (like sample per sample) [B, 1, ...] shape
        tp, fp, tn, fn, _ = stat_scores(pred, target.type(torch.int32))

    elif mode == "segment_semantic":
        # calculate confusion matrix parameters per class, [B, C, ...]
        num_classes = target.shape[1]
        batch_size = target.shape[0]

        for batch in range(batch_size):
            conf_matrix_class = stat_scores(pred[batch], target[batch].type(
                torch.int32), multiclass=True, mdmc_reduce='samplewise', reduce='micro')
            conf_matrix_class += conf_matrix_class

        conf_matrix_class = conf_matrix_class
        tp, fp, tn, fn = __get_conf_atribs__(conf_matrix_class, num_classes)

    elif mode == "classif_binary":
        tp, fp, tn, fn, _ = stat_scores(pred.type(torch.int32), target.type(torch.int32))

    elif mode == "classif_multiclass":
        # input shape, vectors [B, ], Number of classes define the highest number in targets of whole dataset f.e. target= [0, 3, 2, 1, 10], number of classes is 10+1=11
        conf_matrix_class = stat_scores(pred.type(torch.int32), target.type(torch.int32),
                                        multiclass=True, mdmc_reduce='samplewise',
                                        reduce='macro', num_classes=num_classes)

        tp, fp, tn, fn = __get_conf_atribs__(conf_matrix_class, num_classes)

    else:
        raise ValueError("Wrong mode passed")

    return tp, fp, tn, fn


def calc_metrics(tp, fp, tn, fn):
    """
    Calculate recall, precision, accuracy, f1-score from TP, FP, TN, FN errors.
    """

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return recall, precision, acc, f1_score


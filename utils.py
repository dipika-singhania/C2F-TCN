import os
import time
import torch
from torch import Tensor
import numpy as np
import torchvision
import glob
import re

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_labels_start_end_time(frame_wise_labels, bg_class):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, bg_class):
    norm = True
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def recog_file(filename, ground_truth_path, overlap, background_class_list):

    # read ground truth
    gt_file = ground_truth_path + re.sub('.*/', '/', filename)
    with open(gt_file, 'r') as f:
        gt_content = f.read().split('\n')[0:-1]
        f.close()
    # read recognized sequence
    with open(filename, 'r') as f:
        recog_content = f.read().split('\n')[0:-1]  # framelevel recognition is in 6-th line of file
        f.close()

    n_frame_correct = 0
    for i in range(len(recog_content)):
        if recog_content[i] == gt_content[i]:
            n_frame_correct += 1

    edit_score_value = edit_score(recog_content, gt_content, background_class_list)

    tp_arr = []
    fp_arr = []
    fn_arr = []
    for s in range(len(overlap)):
        tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s], background_class_list)
        tp_arr.append(tp1)
        fp_arr.append(fp1)
        fn_arr.append(fn1)
    return n_frame_correct, len(recog_content), tp_arr, fp_arr, fn_arr, edit_score_value


def calculate_mof(ground_truth_path_name, prediction_path, background_class):
    overlap = [.1, .25, .5]
    overlap_scores = np.zeros(3)
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    edit = 0
    n_frames = 0
    n_correct = 0

    filelist = glob.glob(prediction_path + '/*txt')

    print('Evaluate %d video files...' % len(filelist))
    if len(filelist) == 0:
        return 0, 0, overlap_scores
    # loop over all recognition files and evaluate the frame error
    for filename in filelist:
        correct, frames, tp_arr, fp_arr, fn_arr, edit_score_value = recog_file(filename, ground_truth_path_name,
                                                                               overlap, background_class)
        n_correct += correct
        n_frames += frames
        edit += edit_score_value

        for i in range(len(overlap)):
            tp[i] += tp_arr[i]
            fp[i] += fp_arr[i]
            fn[i] += fn_arr[i]

    if n_correct == 0 or n_frames == 0:
        acc = 0
    else:
        acc = float(n_correct) * 100.0 / n_frames

    print('frame accuracy: %0.4f' % acc)
    final_edit_score = ((1.0 * edit) / len(filelist))
    print('Edit score: %0.4f' % final_edit_score)

    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])

        f1 = 2.0 * (precision * recall) / (precision + recall)

        f1 = np.nan_to_num(f1) * 100
        print('F1@%0.2f: %.4f' % (overlap[s], f1))
        overlap_scores[s] = f1

    return final_edit_score, acc, overlap_scores


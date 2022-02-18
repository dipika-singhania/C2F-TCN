import torch
import pandas as pd
import ast
import numpy as np
import h5py
from torchvision import transforms
import os
from PIL import Image
from collections import defaultdict
from itertools import chain as chain
import random


def collate_fn_override(data):
    """
       data:
    """
    data = list(filter(lambda x: x is not None, data))
    data_arr, count, labels, video_len, start, video_id, labels_present_arr, chunk_size, chunk_id = zip(*data)
    return torch.stack(data_arr), torch.tensor(count), torch.stack(labels), torch.tensor(video_len),\
           torch.tensor(start), video_id, torch.stack(labels_present_arr), torch.tensor(chunk_size),\
           torch.tensor(chunk_id)


class AugmentDataset(torch.utils.data.Dataset):
    def __init__(self, args, fold, fold_file_name, chunk_size):

        self.fold = fold
        self.max_frames_per_video = args.max_frames_per_video
        self.feature_size = args.feature_size
        self.base_dir_name = args.features_file_name
        self.frames_format = "{}/{:06d}.jpg"
        self.ground_truth_files_dir = args.ground_truth_files_dir
        self.void_class = args.num_class
        self.num_class = args.num_class
        self.args = args
        self.chunk_size_arr = chunk_size
        self.data = self.make_data_set(fold_file_name)
        
        
    def make_data_set(self, fold_file_name):  # Longer Videos 10 -- max_chunk_size # Shorter Videos = min(chunk size) - max
        df=pd.read_csv(self.args.label_id_csv)
        label_id_to_label_name = {}
        label_name_to_label_id_dict = {}
        for i, ele in df.iterrows():
            label_id_to_label_name[ele.label_id] = ele.label_name
            label_name_to_label_id_dict[ele.label_name] = ele.label_id

        data = open(fold_file_name).read().split("\n")[:-1]
        data_arr = []
        num_video_not_found = 0
        
        for i, video_id in enumerate(data):
            video_id = video_id.split(".txt")[0]
            filename = os.path.join(self.ground_truth_files_dir, video_id + ".txt")

            with open(filename, 'r') as f:
                recog_content = f.read().split('\n')[0:-1]  # framelevel recognition is in 6-th line of file
                f.close()

            recog_content = [label_name_to_label_id_dict[e] for e in recog_content]
            
            total_frames = len(recog_content)
            
            if not os.path.exists(os.path.join(self.base_dir_name, video_id + ".npy")):
                print("Not found video with id", os.path.join(self.base_dir_name, video_id + ".npy"))
                num_video_not_found += 1
                continue

            len_video = len(recog_content)
            
            chunk_size_arr = self.chunk_size_arr
            for i, chunk_size in enumerate(chunk_size_arr):
                for st_frame in range(0, len_video, self.max_frames_per_video * chunk_size):
                    max_end = st_frame + (self.max_frames_per_video * chunk_size)
                    end_frame = max_end if max_end < len_video else len_video
                    ele_dict = {'st_frame': st_frame, 'end_frame': end_frame, 'chunk_id': i, 'chunk_size': chunk_size,
                                'video_id': video_id, 'tot_frames': (end_frame - st_frame) // chunk_size}
                    ele_dict["labels"] = np.array(recog_content, dtype=int)
                    data_arr.append(ele_dict)
        print("Number of datapoints logged in {} fold is {}".format(self.fold, len(data_arr)))
        return data_arr

    def getitem(self, index):  # Try to use this for debugging purpose
        ele_dict = self.data[index]
        count = 0
        st_frame = ele_dict['st_frame']
        end_frame = ele_dict['end_frame']
        aug_chunk_size = ele_dict['chunk_size']
        
        data_arr = torch.zeros((self.max_frames_per_video, self.feature_size))
        label_arr = torch.ones(self.max_frames_per_video, dtype=torch.long) * -100
            
        image_path = os.path.join(self.base_dir_name, ele_dict['video_id'] + ".npy")
        elements = np.load(image_path)
        # elements = np.loadtxt(image_path)
        count = 0
        labels_present_arr = torch.zeros(self.num_class, dtype=torch.float32)
        
        for i in range(st_frame, end_frame, aug_chunk_size):
            end = min(end_frame, i + aug_chunk_size)
            key = elements[:, i: end]
            values, counts = np.unique(ele_dict["labels"][i: end], return_counts=True)
            label_arr[count] = values[np.argmax(counts)]
            labels_present_arr[label_arr[count]] = 1
            data_arr[count, :] = torch.tensor(np.max(key, axis=-1), dtype=torch.float32)
            count += 1
        
        return data_arr, count, label_arr, elements.shape[1], st_frame, ele_dict['video_id'], labels_present_arr, \
               aug_chunk_size, ele_dict['chunk_id']

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.data)


import argparse
import os
import torch
import numpy as np
import random
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
import os
import torch.nn as nn

class PostProcess(nn.Module):
    def __init__(self, args, weights):
        super().__init__()
        df_labels = pd.read_csv(args.label_id_csv)

        self.labels_dict_id2name = {}
        self.labels_dict_name2id = {}
        for i, val in df_labels.iterrows():
            self.labels_dict_id2name[val.label_id] = val.label_name
            self.labels_dict_name2id[val.label_name] = val.label_id

        self.results_dict = dict()
        self.gd_path = args.ground_truth_files_dir
        self.results_json = None
        self.count = 0
        self.acc_dict = dict()
        self.weights  = weights


    def start(self):
        self.results_dict = dict()
        self.count = 0

    def get_acc_dict(self):
        return self.acc_dict
    
    def upsample_video_value(self, predictions, video_len, chunk_size):
        new_label_name_expanded = [] 
        prediction_swap = predictions.permute(1, 0)
        for i, ele in enumerate(prediction_swap):
            st = i * chunk_size
            end = st + chunk_size
            for j in range(st, end):
                new_label_name_expanded.append(ele)
        out_p = torch.stack(new_label_name_expanded).permute(1, 0)[:, :video_len]
        return out_p

    def accumulate_result(self, all_pred_value):
        sum_ac = 0
        for wt, pred_v in zip(self.weights, all_pred_value):
            sum_ac = sum_ac + (wt * pred_v)
        
        return torch.argmax(sum_ac/ sum(self.weights) , dim=0)
        
    def dump_to_directory(self, path):
        
        print("Number of cats =", self.count)
        if len(self.results_dict.items()) == 0:
            return
        prev_vid_id = None
        all_pred_value = None
        ne_dict = {}
        video_id = None
        for video_chunk_id, video_value in self.results_dict.items():
            video_id, chunk_id = video_chunk_id.split("@")[0], video_chunk_id.split("@")[1]
            upped_pred_logit = self.upsample_video_value(video_value[0][:, :video_value[1]],
                                                             video_value[4], video_value[3]).unsqueeze(0)
            if video_id == prev_vid_id:
                all_pred_value = torch.cat([all_pred_value, upped_pred_logit], dim=0)
            else:
                if all_pred_value is not None:
                    ne_dict[prev_vid_id] = self.accumulate_result(all_pred_value)
                    all_pred_value = None
                prev_vid_id = video_id
                all_pred_value = upped_pred_logit # With refinement softmax has to be added
        
        if all_pred_value is not None:
            ne_dict[video_id] = self.accumulate_result(all_pred_value)
        
        for video_id, video_value in ne_dict.items():
            pred_value = video_value.detach().cpu().numpy()
            label_name_arr = [self.labels_dict_id2name[i.item()] for i in pred_value]

            out_path = os.path.join(path, video_id + ".txt")
            with open(out_path, "w") as fp:
                fp.write("\n".join(label_name_arr))
                fp.write("\n")

    @torch.no_grad()
    def forward(self, outputs, video_names, framewise_labels, counts, chunk_size_arr, chunk_id_arr, vid_len_arr):
        for output, vn, framewise_label, count, chunk_size, chunk_id, vid_len in zip(outputs, video_names, framewise_labels,
                                                      counts, chunk_size_arr, chunk_id_arr, vid_len_arr):
#             output_video = torch.argmax(output, 0)
            
            key = '{}@{}'.format(vn, chunk_id)

            if key in self.results_dict:
                self.count += 1

                prev_tensor, prev_count, prev_gt_labels, chunk_size, vid_len = self.results_dict[key]
                output = torch.cat([prev_tensor, output], dim=1)
                framewise_label = torch.cat([prev_gt_labels, framewise_label])
                count = count + prev_count

            self.results_dict[key] = [output, count, framewise_label, chunk_size, vid_len]


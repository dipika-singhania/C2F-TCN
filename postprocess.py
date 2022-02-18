import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np

class PostProcess(nn.Module):
    def __init__(self, args):
        super().__init__()
        df_labels = pd.read_csv(args.label_id_csv)
        
        self.labels_dict_id2name = {}
        self.labels_dict_name2id = {}
        for i, val in df_labels.iterrows():
            self.labels_dict_id2name[val.label_id] = val.label_name
            self.labels_dict_name2id[val.label_name] = val.label_id

        self.ignore_label = args.num_class
        self.results_dict = dict()
        self.threshold = args.iou_threshold
        self.chunk_size = args.chunk_size
        self.gd_path = args.ground_truth_files_dir
        self.results_json = None
        self.count = 0

    def start(self):
        self.results_dict = dict()
        self.count = 0

    def dump_to_directory(self, path):
        print("Number of cats =", self.count)
        if len(self.results_dict.items()) == 0:
            return
        for video_id, video_value in self.results_dict.items():
            pred_value = video_value[0].detach().cpu().numpy()
            label_count = video_value[1].detach().cpu().numpy()
            label_send = video_value[2].detach().cpu().numpy()
            
            video_path = os.path.join(self.gd_path, video_id + ".txt")
            with open(video_path, 'r') as f:
                recog_content = f.read().split('\n')[0:-1]  # framelevel recognition is in 6-th line of file
                f.close()
                    
            recog_content = np.array([self.labels_dict_name2id[e] for e in recog_content])
            
            label_name_arr = [self.labels_dict_id2name[i.item()] for i in pred_value[:label_count.item()]]
            new_label_name_expanded = [] # np.empty(len(recog_content), dtype=np.object_)
            for i, ele in enumerate(label_name_arr):
                st = i * self.chunk_size
                end = st + self.chunk_size
                if end > len(recog_content):
                    end = len(recog_content)
                for j in range(st, end):
                    new_label_name_expanded.append(ele)
                if len(new_label_name_expanded) >= len(recog_content):
                    break
        
            out_path = os.path.join(path, video_id + ".txt")
            with open(out_path, "w") as fp:
                fp.write("\n".join(new_label_name_expanded))
                fp.write("\n")
            
    @torch.no_grad()
    def forward(self, outputs, video_names, framewise_labels, counts):
        """ Perform the computation
        Parameters:
            :param outputs: raw outputs of the model
            :param start_frame:
            :param video_names:
            :param clip_length:
        """
        for output, vn, framewise_label, count in zip(outputs, video_names, framewise_labels, counts):
            output_video = torch.argmax(output, 0)
            if vn in self.results_dict:
                self.count += 1
                
                prev_tensor, prev_count, prev_gt_labels = self.results_dict[vn]
                output_video = torch.cat([prev_tensor, output_video])
                framewise_label = torch.cat([prev_gt_labels, framewise_label])
                count = count + prev_count
            
            self.results_dict[vn] = [output_video, count, framewise_label]

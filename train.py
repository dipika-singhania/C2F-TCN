import importlib
import os
import warnings
warnings.filterwarnings('ignore')
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from utils import dotdict
from utils import calculate_mof
from postprocess import PostProcess
import torch.nn.functional as F
from dataset import AugmentDataset, collate_fn_override
seed = 42

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--dataset_name', type=str, default="breakfast", choices=['breakfast', '50salads', 'gtea'])
my_parser.add_argument('--split', type=int, required=True, help="Split number of the dataset")
my_parser.add_argument('--cudad', type=str, default='0', help="Cuda device number to run the program")
my_parser.add_argument('--base_dir', type=str, help="Base directory containing groundTruth, features, splits, results directory of dataset")
my_parser.add_argument('--model_path', type=str, default='model')
my_parser.add_argument('--wd', type=float, required=False, help="Provide weigth decay if you want to change from default")
my_parser.add_argument('--lr', type=float, required=False, help="Provide learning rate if you want to change from default")
my_parser.add_argument('--chunk_size', type=int, required=False, help="Provide chunk size to be used if you want to change from default")
my_parser.add_argument('--ensem_weights', type=str, required=False,
                        help='Default = \"1,1,1,1,0,0\", provide in similar comma-seperated 6 weights values if required to be changed')
my_parser.add_argument('--ft_file', type=str, required=False, help="Provide feature file dir path if default is not base_dir/features")
my_parser.add_argument('--ft_size', type=int, required=False, help="Default=2048 for I3D features, change if feature size changes")
my_parser.add_argument('--err_bar', type=int, required=False)
my_parser.add_argument('--num_workers', type=int, default=7, help="Number of workers to be used for data loading")
my_parser.add_argument('--out_dir', required=False, help="Directory where output(checkpoints, logs, results) is to be dumped")
args = my_parser.parse_args()


if args.err_bar:
    seed = args.err_bar #np.random.randint(0, 999999)

# Ensure deterministic behavior
def set_seed():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
set_seed()

# Device configuration
os.environ['CUDA_VISIBLE_DEVICES']=args.cudad
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = dotdict(
    epochs = 500,
    dataset = args.dataset_name,
    feature_size = 2048,
    gamma = 0.5,
    step_size = 500,
    split_number = args.split,
    model_path = args.model_path,
    base_dir = args.base_dir,
    aug=1,
    lps=0)

if args.dataset_name == "breakfast":
    config.chunk_size = 10
    config.max_frames_per_video = 1200
    config.learning_rate = 1e-4
    config.weight_decay = 3e-3
    config.batch_size = 100
    config.num_class = 48
    config.back_gd = ['SIL']
    config.ensem_weights = [1, 1, 1, 1, 0, 0]
elif args.dataset_name == "gtea":
    config.chunk_size = 4
    config.max_frames_per_video = 600
    config.learning_rate = 5e-4
    config.weight_decay = 3e-4
    config.batch_size = 11
    config.num_class = 11
    config.back_gd = ['background']
    config.ensem_weights = [1, 1, 1, 1, 0, 0]
else: # args.dataset_name == "50salads":
    config.chunk_size = 20
    config.max_frames_per_video = 960
    config.learning_rate = 3e-4
    config.weight_decay = 1e-3
    config.batch_size = 20
    config.num_class = 19
    config.back_gd = ['action_start', 'action_end']
    config.ensem_weights = [1, 1, 1, 1, 0, 0]

config.output_dir = config.base_dir + "results/trym{}_split{}_aug{}".format(args.model_path, config.split_number,
                                                                                  config.aug)
if args.wd is not None:
    config.weight_decay = args.wd
    config.output_dir=config.output_dir + "_wd{:.5f}".format(config.weight_decay)

if args.lr is not None:
    config.learning_rate = args.lr
    config.output_dir=config.output_dir + "_lr{:.6f}".format(config.learning_rate)

if args.chunk_size is not None:
    config.chunk_size = args.chunk_size
    config.output_dir=config.output_dir + "_chunk{}".format(config.chunk_size)

if args.ensem_weights is not None:
    config.output_dir=config.output_dir + "_wts{}".format(args.ensem_weights.replace(',','-'))
    config.ensem_weights = list(map(int, args.ensem_weights.split(",")))
    print("C2F Ensemble Weights being used is ", config.ensem_weights)


print("printing in output dir = ", config.output_dir)
config.project_name="{}-split{}".format(config.dataset, config.split_number)
config.train_split_file = config.base_dir + "splits/train.split{}.bundle".format(config.split_number)
config.test_split_file = config.base_dir + "splits/test.split{}.bundle".format(config.split_number)
config.features_file_name = config.base_dir + "/features/"

if args.ft_file is not None:
    config.features_file_name = os.path.join(config.base_dir, args.ft_file)
    config.output_dir = config.output_dir + "_ft_file{}".format(args.ft_file)

if args.ft_size is not None:
    config.feature_size = args.ft_size
    config.output_dir = config.output_dir + "_ft_size{}".format(args.ft_file)
 
config.ground_truth_files_dir = config.base_dir + "/groundTruth/"
config.label_id_csv = config.base_dir + "mapping.csv"

config.output_dir = config.output_dir + "/"

if args.out_dir is not None:
    config.output_dir = args.out_dir + "/"

def model_pipeline(config):
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)

    # make the model, data, and optimization problem
    model, train_loader, test_loader, criterion, optimizer, scheduler, postprocessor = make(config)

    # and use them to train the model
    train(model, train_loader, criterion, optimizer, scheduler, config, test_loader, postprocessor)

    # and test its final performance
    model.load_state_dict(load_avgbest_model(config))
    acc = test(model, test_loader, criterion, postprocessor, config, config.epochs, 'avg')

    model.load_state_dict(load_best_model(config))
    acc = test(model, test_loader, criterion, postprocessor, config, config.epochs, '')

    return model

def load_best_model(config):
    return torch.load(config.output_dir + '/best_' + config.dataset + '_unet.wt')

def load_avgbest_model(config):
    return torch.load(config.output_dir + '/avgbest_' + config.dataset + '_unet.wt')

def make(config):
    # Make the data
    train, test = get_data(config, train=True), get_data(config, train=False)
    train_loader = make_loader(train, batch_size=config.batch_size, train=True)
    test_loader = make_loader(test, batch_size=config.batch_size, train=False)

    # Make the model
    model = get_model(config).to(device)
    
    num_params = sum([p.numel() for p in model.parameters()])
    print("Number of parameters = ", num_params/1e6, " million")

    # Make the loss and optimizer
    criterion = get_criterion(config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    
    # postprocessor declaration
    postprocessor = PostProcess(config)
    postprocessor = postprocessor.to(device)
    
    return model, train_loader, test_loader, criterion, optimizer, scheduler, postprocessor


class CriterionClass(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)  # Frame wise cross entropy loss
        self.mse = nn.MSELoss(reduction='none')           # Migitating transistion loss 
    
    def forward(self, outp, labels, src_mask, labels_present):
        outp_wo_softmax = torch.log(outp + 1e-10)         # log is necessary because ensemble gives softmax output
        ce_loss = self.ce(outp_wo_softmax, labels)        
        
        mse_loss = 0.15 * torch.mean(torch.clamp(self.mse(outp_wo_softmax[:, :, 1:],
                                                          outp_wo_softmax.detach()[:, :, :-1]), 
                                                 min=0, max=16) * src_mask[:, :, 1:])

        loss = ce_loss + mse_loss 
        return {'full_loss':loss, 'ce_loss':ce_loss, 'mse_loss': mse_loss} 

def get_criterion(config):
    return CriterionClass(config)

def get_data(args, train=True):
    if train is True:
        fold='train'
        split_file_name = args.train_split_file
    else:
        fold='val'
        split_file_name = args.test_split_file

    dataset = AugmentDataset(args, fold=fold, fold_file_name=split_file_name, zoom_crop=(0.5, 2))
    return dataset


def make_loader(dataset, batch_size, train=True):
    def _init_fn(worker_id):
        np.random.seed(int(seed))
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=train,
                                         pin_memory=True, num_workers=args.num_workers, collate_fn=collate_fn_override,
                                         worker_init_fn=_init_fn)
    return loader


def get_model(config):
    my_module = importlib.import_module(config.model_path)
    set_seed()
    return my_module.C2F_TCN(config.feature_size, config.num_class)


def get_c2f_ensemble_output(outp, weights):
    
    ensemble_prob = F.softmax(outp[0], dim=1) * weights[0] / sum(weights)

    for i, outp_ele in enumerate(outp[1]):
        upped_logit = F.upsample(outp_ele, size=outp[0].shape[-1], mode='linear', align_corners=True)
        ensemble_prob = ensemble_prob + F.softmax(upped_logit, dim=1) * weights[i + 1] / sum(weights)
    
    return ensemble_prob

def train(model, loader, criterion, optimizer, scheduler, config, test_loader, postprocessor):

    best_acc = 0
    avg_best_acc = 0
    accs = []
    
    for epoch in range(config.epochs):
        model.train()
        for i, item in enumerate(loader):
            samples = item[0].to(device).permute(0, 2, 1)
            count = item[1].to(device)
            labels = item[2].to(device)
            src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
            src_mask = src_mask.to(device)
            
            src_msk_send = src_mask.to(torch.float32).to(device).unsqueeze(1)

            # Forward pass ➡
            outputs_list = model(samples)
            outputs_ensemble = get_c2f_ensemble_output(outputs_list, config.ensem_weights)
            
            loss_dict = criterion(outputs_ensemble, labels, src_msk_send, item[6].to(device))
            loss = loss_dict['full_loss']

            # Backward pass ⬅
            optimizer.zero_grad()
            loss.backward()

            # Step with optimizer
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Train loss after {epoch} epochs, {i} iterations is {loss_dict['full_loss']:.3f}")

        acc, avg_score = test(model, test_loader, criterion, postprocessor, config, epoch, '')
        if avg_score > avg_best_acc:
            avg_best_acc = avg_score
            torch.save(model.state_dict(), config.output_dir + '/avgbest_' + config.dataset + '_unet.wt')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), config.output_dir + '/best_' + config.dataset + '_unet.wt')

        torch.save(model.state_dict(), config.output_dir + '/last_' + config.dataset + '_unet.wt')
        accs.append(acc)
        accs.sort(reverse=True)
        scheduler.step()
        print(f'Validation best accuracies till now -> {" ".join(["%.2f"%item for item in accs[:3]])}')


def test(model, test_loader, criterion, postprocessors, args, epoch, dump_prefix):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        avg_loss = []
        for i, item in enumerate(test_loader):
            samples = item[0].to(device).permute(0, 2, 1)
            count = item[1].to(device)
            labels = item[2].to(device)
            src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None]
            src_mask = src_mask.to(device)
            
            src_msk_send = src_mask.to(torch.float32).to(device).unsqueeze(1)

            # Forward pass ➡
            outputs_list = model(samples)
            outputs_ensemble = get_c2f_ensemble_output(outputs_list, config.ensem_weights)
            
            loss_dict = criterion(outputs_ensemble, labels, src_msk_send, item[6].to(device))
            loss = loss_dict['full_loss']
            avg_loss.append(loss.item())
            
            pred = torch.argmax(outputs_ensemble, dim=1)
            correct += float(torch.sum((pred == labels) * src_mask).item())
            total += float(torch.sum(src_mask).item())
            postprocessors(outputs_ensemble, item[5], labels, count)
            
        # Add postprocessing and check the outcomes
        path = os.path.join(args.output_dir, dump_prefix + "predict_" + args.dataset)
        if not os.path.exists(path):
            os.mkdir(path)
        postprocessors.dump_to_directory(path)
        final_edit_score, map_v, overlap_scores = calculate_mof(args.ground_truth_files_dir, path, config.back_gd)
        postprocessors.start()
        acc = 100.0 * correct / total

        print(f"Validation loss = {np.mean(np.array(avg_loss)): .3f}, accuracy of the model after epoch {epoch} = {acc: .3f}%")
        with open(config.output_dir + "/results_file.txt", "a+") as fp:
            fp.write("{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}\n".format(overlap_scores[0], overlap_scores[1], 
                                                overlap_scores[2], final_edit_score, map_v))
        if epoch == config.epochs:
            with open(config.output_dir + "/" + dump_prefix + "final_results_file.txt", "a+") as fp:
                fp.write("{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}\n".format(overlap_scores[0], overlap_scores[1], 
                                                    overlap_scores[2], final_edit_score, map_v))
                

    avg_score = (map_v + final_edit_score) / 2
    return map_v, avg_score

import time
start_time = time.time()
model = model_pipeline(config)
end_time = time.time()

duration = (end_time - start_time) / 60
print(f"Total time taken = ", duration, "mins")

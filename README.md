# C2F-TCN: Coarse to Fine Multi-Resolution Temporal Convolutional Network for Temporal Action Segmentation

Official implementation of Coarse to Fine Multi-Resolution Temporal Convolutional Network for Temporal Action Segmentation [link](https://arxiv.org/pdf/2105.10859.pdf)

Code for full supervsion version of ‘C2F-TCN: A Framework for Semi- and Fully-Supervised Temporal Action Segmentation’ [link] (https://ieeexplore.ieee.org/abstract/document/10147035) published in TPAMI-2023.

Code for semi-supervised version of the same is available at [link](https://github.com/dipika-singhania/ICC-Semi-Supervised-TAS).



### Data download and directory structure:

The I3D features, ground-truth and test split files are similar used to [MSTCN++](https://github.com/yabufarha/ms-tcn). 
In the mstcn_data, download additional files, checkpoints and semi-supervised splits can be downloaded from [drive](https://drive.google.com/drive/folders/1ArYPctLZZKfjicEf5nl4LJrY9xxFc6wU?usp=sharing) . 
Specifically, this drive link contains all necessary data in required directory structure except breakfast I3D feature files which can be downloaded from MSTCN++ data directory.
It also contains the checkpoints files for supervised C2FTCN.

The data directory is arranged in following structure

- mstcn_data
   - mapping.csv
   - dataset_name
   - groundTruth
   - splits
   - results
        - supervised_C2FTCN
            - split1
              - check_pointfile
            - split2
            - 

### Run Scripts
The various scripts to run the supervised training, evaluation with test augmentation or with test augmentation is provided as example below.
Change the dataset_name,  to run on a different dataset.

#### Training C2FTCN for a particular split of a dataset
    ##### python train.py --dataset_name <gtea/50salads/breakfast> --cudad <cuda_device_number> --base_dir <data_directory_for_dataset> --split <split_number>
    Example:
    python train.py --dataset_name 50salads --cudad 1 --base_dir ../mstcn_data/50salads/ --split 5


#### Evaluate C2FTCN without test time augmentation, showing average results from all splits of dataset
    ##### python eval.py --dataset_name <gtea/50salads/breakfast> --cudad <cuda_device_number> --base_dir <data_directory_for_dataset> --compile_result
    Example:
    python eval.py --dataset_name 50salads --cudad 2 --base_dir ../mstcn_data/50salads/ --compile_result

#### Evaluate C2FTCN with test time augmentation, showing average results from all splits of dataset
    ##### python eval.py --dataset_name <gtea/50salads/breakfast> --cudad <cuda_device_number> --base_dir <data_directory_for_dataset>
    Example:
    python eval.py --dataset_name 50salads --cudad 2 --base_dir ../mstcn_data/50salads/



### Citation:

If you use the code, please cite

D. Singhania, R. Rahaman and A. Yao, "C2F-TCN: A Framework for Semi- and Fully-Supervised Temporal Action Segmentation," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2023.3284080.

Singhania, D., Rahaman, R., & Yao, A. (2022, June). Iterative contrast-classify for semi-supervised temporal action segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 2, pp. 2262-2270).


    

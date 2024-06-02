# Transferring Adult-like Phase Images for Robust Multi-view Isointense Infant Brain Segmentation

### Authors
Huabing Liu*, Jiawei Huang, Dengqiang Jia, Qian Wang, Jun Xu, and Dinggang Shen

### Citation
To be released

#### Introduction
This repo includes the source codes and pretrained models for our latest work on isointense infant brain segmentation. The two major components are 1) disentangled cycle-consistent adversarial network ([dcan](https://github.com/hb-liu/multi-view-iseg/tree/main/dcan)) for style transfer between isointense and adult-like phase images; 2) the segmentation network [coseg](https://github.com/hb-liu/multi-view-iseg/tree/main/coseg) that implements multi-view learning to incorporate adult-like phase images in isointense infant brain segmentation. If you find this repo useful, please give it a star ⭐ and consider citing our paper in your research. Thank you.

## 1. Create Environment:
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download))

- NVIDIA GPU + [Pytorch](https://pytorch.org)

- Python packages:

```shell
pip install -r requirements.txt
```

## 2. Preparation
#### Prepare code
Build up the workspace, so that everything can be correctly stored:
```shell
sh install.sh
```

#### Prapare dataset
For your own dataset, format each data as:
```shell
|--<name_of_the_data>
    |-- t1.nii.gz
    |-- t2.nii.gz
    |-- seg.nii.gz
```
for T1-weighted images, T2-weighted images, and segmentation (if exists), respectively.

Then put formatted data into correct folders:
- for isointense phase images, put them into \<pwd\>/dcan/data/raw/6m
- for adult-like phase images, put them into \<pwd\>/dcan/data/raw/12m

Suppose \<pwd\> is the directory of this repo

#### Prapare pretrained models
For test-only purpose of this repo, we have shared all the pretrained models:

| Method | Model Zoo |
| :----: | :-------: |
|  dcan  | [OneDrive](https://onedrive.live.com/?authkey=%21ANnlU3K4Yt4EWqs&id=E5EC3254E49F853%218439&cid=0E5EC3254E49F853) |
|  coseg | [OneDrive](https://onedrive.live.com/?authkey=%21AIjdjYs8MQ5wVX4&id=E5EC3254E49F853%218432&cid=0E5EC3254E49F853) |

Put downloaded *.pth into Results folders

## 3. Run [DCAN](https://github.com/hb-liu/multi-view-iseg/tree/main/dcan)
#### Preprocess
```shell
Run proc.ipynb
```
Modify the proc.ipynb:
- for process isointense phase images:
```shell
data_path = 'data/raw/6m'
out_path = 'data/processed/6m'
```
- for process adult-like phase images:
```shell
data_path = 'data/raw/12m'
out_path = 'data/processed/12m'
```

#### Train
```shell
Python3 train.py
```

#### Synthesize data for downstream segmentation task
```shell
Run syn.ipynb
```
Modify the syn.ipynb:
- for transferring isointense phase images to adult-like contrast
```shell
config.dataset.src_dir = 'data/processed/6m'
config.dataset.dst_dir = 'data/processed/12m'
```
- for transferring adult-like phase images to isointense contrast:
```shell
config.dataset.src_dir = 'data/processed/12m'
config.dataset.dst_dir = 'data/processed/6m'
```

## 4. Run [COSEG](https://github.com/hb-liu/multi-view-iseg/tree/main/coseg)
#### Preprocess
```shell
Run proc.ipynb
```
Modify the proc.ipynb
- for processing source isointense phase images:
```shell
data_path = '../dcan/data/processed/6m'
out_path = 'data/processed/6m'
```
- for processing synthetic isointense phase images:
```shell
data_path = '../dcan/data/syn/6m'
out_path = 'data/syn/6m'
```
- for processing source adult-like phase images:
```shell
data_path = '../dcan/data/processed/12m'
out_path = 'data/processed/12m'
```
- for processing synthetic adult-like phase images:
```shell
data_path = '../dcan/data/syn/12m'
out_path = 'data/syn/12m'
```

#### Train
```shell
sh train.sh
```

#### Test
```shell
python3 test.py
```

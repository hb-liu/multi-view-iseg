# Transferring Adult-like Phase Images for Robust Multi-view Isointense Infant Brain Segmentation

#### Authors
Huabing Liu*, Jiawei Huang, Dengqiang Jia, Qian Wang, Jun Xu, and Dinggang Shen

#### Citation
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

## 2. Prepare Code, Dataset, and Pretained Models:
* Build up the workspace, so that everything can be correctly stored:
```shell
sh install.sh
```

* For your own dataset, format each data as:
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

* For test-only purpose of this repo, we have shared all the pretrained models:

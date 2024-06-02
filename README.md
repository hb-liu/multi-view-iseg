# Transferring Adult-like Phase Images for Robust Multi-view Isointense Infant Brain Segmentation

#### Authors
Huabing Liu*, Jiawei Huang, Dengqiang Jia, Qian Wang, Jun Xu, and Dinggang Shen

#### Citation
To be released

#### Introduction
This repo includes the source codes and pretrained models for our latest work on isointense infant brain segmentation. The two major components are 1) disentangled cycle-consistent adversarial network (DCAN) for style transfer between isointense and adult-like phase images; 2) the segmentation network coseg that implmentes multi-view learning to incorporate adult-like phase images in isointense infant brain segmentation. If you find this repo useful, please give it a star ⭐ and consider citing our paper in your research. Thank you.

## 1. Create Environment:
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

```shell
  pip install -r requirements.txt
```

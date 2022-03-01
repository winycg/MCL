
# Mutual Contrastive Learning for Visual Representation Learning

This project provides source code for our Mutual Contrastive Learning for Visual Representation Learning (MCL).


## Installation

### Requirements

Ubuntu 18.04 LTS

Python 3.8 ([Anaconda](https://www.anaconda.com/) is recommended)

CUDA 11.1

PyTorch 1.7.0

NCCL for CUDA 11.1


## Supervised Learning on CIFAR-100 dataset
### Dataset
CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

unzip to the `./data` folder

### Training two baseline networks
```
python main_cifar.py --arch resnet32 --number-net 2
```
More commands for training various architectures can be found in `scripts/train_cifar_baseline.sh`

### Training two networks by MCL
```
python main_cifar.py --arch resnet32  --number-net 2 \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. 
```
More commands for training various architectures can be found in `scripts/train_cifar_mcl.sh`

###  Results of MCL on CIFAR-100
We perform all experiments on a single NVIDIA RTX 3090 GPU (24GB) with three runs.
| Network | Baseline | MCL(×2) | MCL(×4) |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|
| ResNet-32 | 70.91±0.14 | 72.96±0.28 | 74.04±0.07 |
| ResNet-56 | 73.15±0.23 | 74.48±0.23 | 75.74±0.16 |
| ResNet-110 | 75.29±0.16 | 77.12±0.20 | 78.82±0.14 |
| WRN-16-2 | 72.55±0.24 | 74.56±0.11 | 75.79±0.07 |
| WRN-40-2 | 76.89±0.29 | 77.51±0.42 | 78.84±0.22 |
| HCGNet-A1 | 77.42±0.16 | 78.62±0.26 | 79.50±0.15 |
| ShuffleNetV2 0.5× |67.39±0.35 | 69.55±0.22 | 70.92±0.28 |
| ShuffleNetV2 1× | 70.93±0.24 | 73.26±0.18 | 75.18±0.25 |

### Training multiple networks by MCL combined with Logit distillation
```
python main_cifar.py --arch WRN_16_2  --number-net 4 \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. \
    --logit-distill
```
More commands for training various architectures can be found in `scripts/train_cifar_mcl_logit.sh`

###  Results of MCL combined with logit distillation on CIFAR-100
We perform all experiments on a single NVIDIA RTX 3090 GPU (24GB) with three runs.
| Network | Baseline |  MCL(×4)+Logit KD |
|:---------------:|:-----------------:|:-----------------:|
| WRN-16-2 | 72.55±0.24 | 76.34±0.22 |
| WRN-40-2 | 76.89±0.29 | 80.02±0.45 |
| WRN-28-4 | 79.17±0.29 | 81.68±0.31 |
| ShuffleNetV2 1× | 70.93±0.24 | 77.02±0.32 |
| HCGNet-A2 | 79.00±0.41 | 82.47±0.20 |


## Supervised Learning on ImageNet dataset

### Dataset preparation

- Download the ImageNet dataset to YOUR_IMAGENET_PATH and move validation images to labeled subfolders
    - The [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) may be helpful.

- Create a datasets subfolder and a symlink to the ImageNet dataset

```
$ ln -s PATH_TO_YOUR_IMAGENET ./data/
```
Folder of ImageNet Dataset:
```
data/ImageNet
├── train
├── val
```

### Training two networks by MCL
```
python main_imagenet.py --arch resnet18  --number-net 2 \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. 
```
More commands for training various architectures can be found in `scripts/train_imagenet_mcl.sh`

###  Results of MCL on ImageNet
We perform all experiments on a single NVIDIA Tesla V100 GPU (32GB) with three runs.
| Network | Baseline | MCL(×2) | MCL(×4) |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|
| ResNet-18 | 69.76 | 70.32 | 70.77 |
| ResNet-34 | 73.30 | 74.13 | 74.34 |

### Training two networks by MCL combined with logit distillation
```
python main_imagenet.py --arch resnet18  --number-net 2 \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. 
```
More commands for training various architectures can be found in `scripts/train_imagenet_mcl.sh`

###  Results of MCL combined with logit distillation on ImageNet
We perform all experiments on a single NVIDIA Tesla V100 GPU (32GB) with three runs.
| Network | Baseline | MCL(×4)+Logit KD |
|:---------------:|:-----------------:|:-----------------:|
| ResNet-18 | 69.76 |  70.82 |

## Self-Supervised Learning on ImageNet dataset

### Apply MCL(×2) to MoCo
```
python main_moco_mcl.py \
  -a resnet18 \
  --lr 0.03 \
  --batch-size 256 \
  --number-net 2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --gpu-ids 0,1,2,3,4,5,6,7 
```
### Linear Classification
```
python main_lincls.py \
  -a resnet18 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --gpu-ids 0,1,2,3,4,5,6,7 
```

###  Results of applying MCL to MoCo on ImageNet
We perform all experiments on 8 NVIDIA RTX 3090 GPUs with three runs.
| Network | Baseline | MCL(×2) |
|:---------------:|:-----------------:|:-----------------:|
| ResNet-18 | 47.45±0.11 |  48.04±0.13 |
## Citation

```
@inproceedings{yang2022mcl,
  title={Mutual Contrastive Learning for Visual Representation Learning},
  author={Chuanguang Yang, Zhulin An, Linhang Cai, Yongjun Xu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```

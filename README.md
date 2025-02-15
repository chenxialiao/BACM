# An Open-set Few-shot Face Recognition Framework with Balanced Adaptive Cohesive Mixing
PyTorch implementation of the paper An Open-set Few-shot Face Recognition Framework with Balanced Adaptive Cohesive Mixing.

# Requirement
This repo was tested with Ubuntu 18.04.5 LTS, Python 3.10, Pytorch 2.0.0, and CUDA 11.8. You will need at least 64GB RAM and 24GB VRAM(i.e. Nvidia RTX-3090) for running full experiments in this repo.


# Training
We use the same backbone network and training strategies as 'S2M2_R'. Please refer to [S2M2](https://github.com/nupurkmr9/S2M2_fewshot) for the backbone training. However, it should be noted that we have added Gaussian likelihood loss and need to place the train.py and Gaussian_loss.py files in appropriate folders

# Extracted Features
You can directly download the extracted features from the link:<br>
https://drive.google.com/file/d/1kvdfKsyS6Atr9-T8wOIYexSPrdM9-nuy/view?usp=drive_link
***
After downloading the extracted features, put them in ./features/WideResNet28_10_S2M2_R/[dataset]/ respectively.

# Datasets
* **_mini_ Imagenet:**
Please follow [miniImageNet](https://github.com/yaoyao-liu/mini-imagenet-tools) to obtain the _mini_ ImageNet dataset and put it in ./filelists/miniImagenet/.<br>
* **CIFAR-FS:**
Please follow [CIFAR-FS](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_cifar_fs.sh) to obtain the CIFAR-FS dataset and put it in ./filelists/cifar/.<br>
* **CUB:**
Please follow [CUB](https://github.com/cyizhuo/CUB-200-2011-dataset) to obtain the CUB_200_2011 dataset and put it in ./filelists/CUB/.<br>
***
After download all datasets, please run the make_josn.py in each dataset folder to generate json files which include filepath-label pairs of each image.

# Reference
[Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://arxiv.org/pdf/1907.12087v3.pdf)<br>
[https://github.com/nupurkmr9/S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot)<br>
[Free Lunch for Few-Shot Learning: Distribution Calibration](https://openreview.net/forum?id=JWOiYxMG92s)<br>
[https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration)<br>
[L-GM_loss_pytorch](https://github.com/ChaofWang/L-GM_loss_pytorch)<br>
[CPEA](https://github.com/FushengHao/CPEA)

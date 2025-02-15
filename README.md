# An Open-set Few-shot Face Recognition Framework with Balanced Adaptive Cohesive Mixing
PyTorch implementation of the paper An Open-set Few-shot Face Recognition Framework with Balanced Adaptive Cohesive Mixing.

# Requirement
This repo was tested with Ubuntu 18.04.5 LTS, Python 3.10, Pytorch 2.0.0, and CUDA 11.8. You will need at least 64GB RAM and 24GB VRAM(i.e. Nvidia RTX-3090) for running full experiments in this repo.

# Download Pretrained Weights:
You can directly download the weight of pre-training encoder from the link:<br>
[VGGNet-19](https://drive.google.com/file/d/1tGoX7fR-8m8MufA7HQdWWQn-DgxEOYJK/view?usp=share_link)<br>
[ResNet-50](https://drive.google.com/file/d/1aniiywJB-1jJRuq-vdpxAnKPp38y1CF3/view?usp=share_link)
***




# Datasets
* **VGGFace 2:**
Please follow [VGGFace 2](https://github.com/yaoyao-liu/mini-imagenet-tools) to obtain the VGGFace 2 dataset and put it in ./dataset/VGGFace2/. After download this dataset, please run the make_josn.py to generate json files which include filepath-label pairs of each image which is used to create dataloader.<br>
* **IJB-C:**
Please follow [IJB-C](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_cifar_fs.sh) to obtain the IJB-C dataset and put it in ./dataset/IJB-C/.<br>
* **CASIA-WebFace:** CASIA-WebFace
Please follow [CASIA](https://drive.google.com/file/d/1BSdGyJn0mTWuZDA-_fo7eQvElH2qD2X9/view?usp=share_link) to obtain the CASIA-WebFace dataset and put it in ./dataset/CASIA-WebFace/.<br>
***
The images are already cropped using MTCNN by timesler. <br>
To use your own face dataset, you can simply change the data_config in config.py. <br>
The face dataset must have the structure ROOT/SUBJECT_NAME/image.jpg. <br>

After downloading, change the dataset_config and encoder_config in config.py accordingly.

# Training and Evaluating
After the above steps are done, simply run:
```python main.py --dataset='CASIA' --encoder='VGG19' --classifier_init='WI' --finetune_layers='BN'```
***
# Reference
[https://github.com/1ho0jin1/OSFI-by-FineTuning](https://github.com/1ho0jin1/OSFI-by-FineTuning)<br>
[https://github.com/nupurkmr9/S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot)<br>
[L-GM_loss_pytorch](https://github.com/ChaofWang/L-GM_loss_pytorch)<br>
[CPEA](https://github.com/FushengHao/CPEA)

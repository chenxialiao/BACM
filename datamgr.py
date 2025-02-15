# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import sys
import os
# os.chdir(r'/home/liao/TEST/OSFI-by-FineTuning/data')
import torch.utils.data
import torchvision.transforms as transforms
from dataset_1 import SimpleDataset, SetDataset, EpisodicBatchSampler, CustomSampler, CustomBatchSampler, FiltDataset, RandomSelectDataset, RandomSelectClassDataset
from abc import abstractmethod
import random
from torch.utils.data import Sampler

import torch
from PIL import ImageEnhance

transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)
#ImageJitter 类：用于对图像进行随机亮度、对比度、锐度和色彩增强。通过传入一个包含增强参数的字典初始化，然后在调用时对输入的图像进行随机增强。
class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out
#TransformLoader 类：提供图像预处理的转换方法集合，包括图像尺寸调整、颜色归一化、随机裁剪、水平翻转等。用户可以设置不同的参数，如图像大小、归一化参数和图像抖动参数，然后通过 parse_transform 方法解析转换类型并返回相应的转换函数。get_composed_transform 方法用于组合一系列转换操作形成一个完整的图像预处理流程。
class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Resize':
            # return method([int(self.image_size*1.15), int(self.image_size*1.15)])
            return method([112, 112])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize','CenterCrop', 'ToTensor']
            # transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform
class CustomSampler(Sampler):
    def __init__(self, data, batch_size):
        super(CustomSampler, self).__init__(self)
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        indices = []
        random_class = [cls for cls in range(self.data.num_classes)]
        random.shuffle(random_class)
        for n in random_class:
            index = torch.where(self.data.meta['image_labels'] == n)[0]
            idx = torch.randperm(index.nelement())
            index = index.view(-1)[idx].view(index.size())
            indices.append(index)
        indices = torch.cat(indices, dim=0)
        chunks = int(len(indices) / self.batch_size)
        indices = torch.split(indices, self.batch_size,dim=0)
        index = [idx for idx in range(chunks)]
        random.shuffle(index)
        indices = torch.concatenate([indices[i] for i in index],dim=0)
        return iter(indices)

    def __len__(self):
        return len(self.data)

# class CustomSampler(Sampler):
#     def __init__(self, data):
#         super(CustomSampler, self).__init__(self)
#         self.data = data
#
#     def __iter__(self):
#         indices = []
#         random_class = [cls for cls in range(self.data.num_classes)]
#         random.shuffle(random_class)
#         for n in random_class:
#             index = torch.where(self.data.meta['image_labels'] == n)[0]
#             idx = torch.randperm(index.nelement())
#             index = index.view(-1)[idx].view(index.size())
#             indices.append(index)
#         indices = torch.cat(indices, dim=0)
#         return iter(indices)
#
#     def __len__(self):
#         return len(self.data)

class CustomBatchSampler:
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            if (
                i < len(sampler_list) - 1
                and self.sampler.data.meta['image_labels'][idx]
                != self.sampler.data.meta['image_labels'][sampler_list[i + 1]]
            ):
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch = []
                else:
                    batch = []
            i += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size, num_classes):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.num_classes = num_classes
    def get_data_loader(self, data_file, aug, shuffle): #parameters that would change on train/val set
        # transform = self.trans_loader.get_composed_transform(aug)
        transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize(torch.FloatTensor([0.5, 0.5, 0.5]), torch.FloatTensor([0.5, 0.5, 0.5])),
    ])
        dataset = SimpleDataset(data_file,num_classes=self.num_classes, transform=transform)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = shuffle, num_workers = 8, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

    def get_order_data_loader(self, data_file, aug, num_workers=8,
                              pin_memory=True):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file,num_classes=self.num_classes, transform=transform)
        sampler = CustomSampler(dataset, self.batch_size)
        batch_sampler = CustomBatchSampler(sampler, batch_size=self.batch_size , drop_last=True)
        # data_loader_params = dict(batch_size = self.batch_size, shuffle = False, num_workers = num_workers, pin_memory = pin_memory)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                                                  pin_memory=pin_memory)
        return data_loader

    def get_base_filt_data_loader(self, random_select_dataset, shuffle = True, filt_idx = None, pseudo_label = None):
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(torch.FloatTensor([0.5, 0.5, 0.5]), torch.FloatTensor([0.5, 0.5, 0.5])),
        ])
        dataset = FiltDataset(random_select_dataset, filt_idx = filt_idx, pseudo_label = pseudo_label)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

    def get_random_select_data_loader(self, data_file, num_samples, shuffle = True, filt_idx = None, pseudo_label = None):
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(torch.FloatTensor([0.5, 0.5, 0.5]), torch.FloatTensor([0.5, 0.5, 0.5])),
        ])
        dataset = RandomSelectDataset(data_file, num_classes=self.num_classes, transform=transform, num_samples=num_samples)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader, dataset
    def get_random_select_class_data_loader(self, data_file, num_samples, shuffle = True, filt_idx = None, pseudo_label = None):
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(torch.FloatTensor([0.5, 0.5, 0.5]), torch.FloatTensor([0.5, 0.5, 0.5])),
        ])
        dataset = RandomSelectClassDataset(data_file, num_classes=self.num_classes, transform=transform, num_samples=num_samples)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader, dataset

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_eposide =100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset( data_file , self.batch_size, transform )
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 8, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader



import collections
import csv

import config
from datamgr import SimpleDataManager
from dataset3 import open_set_folds, face_dataset, ijbc_dataset, partition_dataset
from model import fetch_encoder, head
from finetune_final_daso import (linear_probing, weight_imprinting,
                            hallucinate_label_max_pred,
                            hallucinate_label_max_pred_with_nac,
                            fine_tune_with_base_samples_mix_up,get_prototype
                            )
from utils import save_dir_far_curve, save_dir_far_excel_entire_experiments

import os
import json
import pprint
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pickle

# for boolean parser argument
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == "True":
        return True
    elif v == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("'True' or 'False' expected")


def False_or_float(v):
    if v == "False":
        return False
    else:
        return float(v)


parser = argparse.ArgumentParser()
# random seed
parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")
# basic arguments
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--image_size", default=112, type=int)
parser.add_argument("--num_cls", default=9000, type=int)
parser.add_argument("--num_epochs", default=2, type=int, help="num_epochs for fine-tuning")
parser.add_argument("--k", default=16, type=int, help="Number of nearest neighbors for NAC")

# val or test mode
parser.add_argument("--mode", default='test', type=str, help="val or test")

# file path
parser.add_argument("--train_file", default="/home/liao/TEST/OSFI-by-FineTuning/data/vggface2_train.json", type=str,
                    help="train json file path")
parser.add_argument("--result_file", default="/home/liao/TEST/OSFI-by-FineTuning/results/CASIA_Res50/auc.csv",
                    type=str, help="train json file path")


# pseudo_label
parser.add_argument("--pseudo_entropy", default=0.75, type=float, help="threshold for data filtering (0.75)")
parser.add_argument("--beta", default=1, type=int, help="coefficient parameter for random selecting base samples (24)")

# temperature scale
parser.add_argument("--temperature_scale", default=True, type=str2bool, help="whether use temperature parameter")
parser.add_argument("--T", default=6, type=float, help="temperature parameter for KL divergence (6 or 12)")
parser.add_argument("--T_dist", default=6, type=float, help="temperature parameter for KL divergence (6 or 12)")
parser.add_argument("--T_proto", default=6, type=float, help="temperature parameter for KL divergence (6 or 12)")

# loss
parser.add_argument("--loss_type", default='KL', type=str, help="['CE','KL'] loss type for base_mix-up loss")

# nac
parser.add_argument("--nac", default=False, type=str2bool, help="whether calculate nac")
parser.add_argument("--alpha_nac", default=10, type=float, help="coefficient parameter for nac loss")
parser.add_argument("--filter_partition", default=0.05, type=float, help="partition samples for nac")


# mix-up

# base_center mix-up
parser.add_argument("--alpha_gallery_base", default=0.01, type=float, help="coefficient parameter for base_mix-up loss (0.01)")

# gallery_gallery mix-up
parser.add_argument("--gallery_gallery", default=False, type=str2bool, help="whether calculate gallery gallery mix-up loss")
parser.add_argument("--alpha_gallery_gallery", default=1, type=float, help="coefficient parameter for gallery_gallery_mix-up loss")
parser.add_argument("--lam_gallery_gallery", default=0.3, type=float, help="lam of mix-up for gallery samples")

# gallery_center mix-up
parser.add_argument("--gallery_center", default=False, type=str2bool, help="whether calculate gallery center mix-up loss")
parser.add_argument("--alpha_gallery_center", default=1, type=float, help="coefficient parameter for gallery_center_mix-up loss")


# lam
parser.add_argument("--lam_type", default='distance_weight', type=str,
                    help="[uniform, dynamic, distance_weight, "  # lam生成方式：均匀分布，动态更新的均匀分布, 距离加权， 动态更新的距离加权
                         "dynamic_distance_weight]")
parser.add_argument("--g_c_lam_type", default='distance_weight', type=str,
                    help="[uniform, distance_weight, dynamic_distance_weight]")
parser.add_argument("--g_g_lam_type", default='fixed', type=str,
                    help="[uniform, fixed]")

# distance_weight lam
parser.add_argument("--gamma", default=2, type=float, help="hyperparameter for base_mix distance_weight (2.1)")
parser.add_argument("--epsilon", default=1.5, type=float, help="hyperparameter for gallery_mix distance_weight (1.5)")
# dynamic_distance_weight lam
parser.add_argument("--gamma_decay", default=0.98, type=float, help="hyperparameter for gamma decay")
# dynamic lam
parser.add_argument("--lam_step", default=0.01, type=float, help="0.002 coefficient parameter for random selecting base samples")
parser.add_argument("--lam_segment", default=5, type=float, help="5 coefficient parameter for random selecting base samples")
parser.add_argument("--lam_start", default=0, type=float, help="0 coefficient parameter for random selecting base samples")
parser.add_argument("--lam_end", default=0.1, type=float, help="0.01 coefficient parameter for random selecting base samples")
# uniform lam
parser.add_argument("--mixup_low", default=0, type=float, help="low threshold of mix-up for base sample")
parser.add_argument("--mixup_high", default=0.1, type=float, help="high threshold of mix-up for base sample (0.2)")

# beta lam
parser.add_argument("--mixup_alpha", default=1, type=int, help="low threshold of mix-up for base sample")
parser.add_argument("--mixup_beta", default=2, type=int, help="high threshold of mix-up for base sample ")


# dataset arguments
parser.add_argument("--dataset", type=str, default='CASIA', help="['CASIA','IJBC']")
parser.add_argument("--num_gallery", type=int, default=3, help="number of gallery images per identity")
parser.add_argument("--num_probe", type=int, default=5, help="maximum number of probe images per identity")

# encoder arguments
parser.add_argument("--encoder", type=str, default='Res50', help="['VGG19','Res50']")
parser.add_argument("--head_type", type=str, default='cos', help="['arc', 'cos', 'mag']")

# main arguments: classifier init / finetune layers / matcher
parser.add_argument("--classifier_init", type=str, default='WI',
                    help="['Random','LP','WI','None']")  # Random Init. / Linear Probing / Weight Imprinting
parser.add_argument("--finetune_layers", type=str, default='BN',
                    help="['None','Full','Partial','PA','BN']")  # 'None' refers to no fine-tuning
parser.add_argument("--matcher", type=str, default='NAC',
                    help="['Cos', 'org','NAC','EVM']")  # unused argument: refer to the results

# misc. arguments: no need to change
parser.add_argument("--arc_s", default=32, type=float, help="scale for ArcFace")
parser.add_argument("--arc_m", default=0.4, type=float, help="margin for ArcFace")
parser.add_argument("--cos_m", default=0.4, type=float, help="margin for CosFace")
parser.add_argument("--train_output", type=str2bool, default=False,
                    help="if True, train output layer")

args = parser.parse_args()


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def extract_feature(data_loader, model, save_dir):
    if os.path.isfile(save_dir + '/features.plk'):
        data = load_pickle(save_dir + '/features.plk')
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    with torch.no_grad():

        output_dict = collections.defaultdict(list)
        for (inputs, labels) in tqdm(data_loader):
            # compute output
            inputs = inputs.to(args.device).cuda()
            labels = labels.type(torch.int64).cuda()
            outputs = model(inputs)
            outputs = outputs.cpu().data.numpy()

            for out, label in zip(outputs, labels):
                output_dict[label.item()].append(out)

        all_info = output_dict
        save_pickle(save_dir + '/features.plk', all_info)
        return all_info


def main(args):
    # check arguments
    assert args.classifier_init in ['Random', 'LP', 'WI', 'None'], 'classifier_init must be one of ["Random","LP","WI",' \
                                                                   '"None"]'
    assert args.finetune_layers in ['None', 'Full', 'Partial', 'PA', 'BN'], \
        "finetune_layers must be one of ['None','Full','Partial','PA','BN']"
    # fix random seed
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # set device
    args.device = torch.device(args.device_id)

    # result save directory
    os.makedirs(f'results/{args.dataset}_{args.encoder}', exist_ok=True)
    if args.finetune_layers == 'None':
        exp_name = 'Pretrained'
    else:
        exp_name = f'{args.classifier_init}_{args.finetune_layers}'
    save_dir = f'results/{args.dataset}_{args.encoder}/{exp_name}'
    os.makedirs(save_dir, exist_ok=True)
    print("results are saved at: ", save_dir)

    # save arguments
    argdict = args.__dict__.copy()
    argdict['device'] = argdict['device'].type + f":{argdict['device'].index}"
    with open(save_dir + '/args.txt', 'w') as fp:
        json.dump(argdict, fp, indent=2)

    """
    prepare G, K, U sets for evaluation
    increase batch_size for faster inference
    
    data set =>     
                                                        known_list                                                                      unknown_list
                    [0, num_gallery]                                    [num_gallery, len(known_list)]                    [0, len(unknown_list)] 
                G (image_path, 0 -> num_gallery)                   K (image_path, num_gallery->len(known_list))         U (image_path, len(known_list))
    """
    data_config = config.data_config[args.dataset]

    train_trf = transforms.Compose([
        transforms.RandomResizedCrop(size=112, scale=(0.7, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(torch.FloatTensor([0.5, 0.5, 0.5]), torch.FloatTensor([0.5, 0.5, 0.5])),
    ])
    eval_trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(torch.FloatTensor([0.5, 0.5, 0.5]), torch.FloatTensor([0.5, 0.5, 0.5])),
    ])

    if args.dataset == 'CASIA':

        folds = open_set_folds(data_config["image_directory"], data_config["known_list_path"],
                               data_config["unknown_list_path"], args.num_gallery)
        # folds.G = folds.G[:1000]
        # folds.val = folds.val[:1000]
        # folds.test = folds.test[:1000]

        dataset_gallery = face_dataset(folds.G, eval_trf, img_size=112)
        if args.mode == 'val':
            dataset_probe = face_dataset(folds.val, eval_trf, img_size=112)
        elif args.mode == 'test':
            dataset_probe = face_dataset(folds.test, eval_trf, img_size=112)

        data_loader_gallery = DataLoader(dataset_gallery, batch_size=128, shuffle=False, num_workers=4)
        data_loader_probe = DataLoader(dataset_probe, batch_size=128, shuffle=False, num_workers=4)
        trainset_gallery = face_dataset(folds.G, train_trf, 112)
        train_loader_gallery = DataLoader(trainset_gallery, batch_size=args.batch_size, shuffle=True, num_workers=4)
    if args.dataset == 'IJBC':
        Gallery, Known, Unknown, Probe,  Val, Test, num_cls = partition_dataset(data_config["ijbc_t_m"], data_config["ijbc_5pts"],
                                                               data_config["ijbc_gallery_1"],
                                                               data_config["ijbc_gallery_2"], data_config["ijbc_probe"],
                                                               data_config["processed_img_root"],
                                                               data_config["plk_file_root"], args.num_gallery)
        dataset_gallery = ijbc_dataset(Gallery, eval_trf, img_size=112)
        if args.mode == 'val':
            dataset_probe = ijbc_dataset(Val, eval_trf, img_size=112)
        elif args.mode == 'test':
            dataset_probe = ijbc_dataset(Test, eval_trf, img_size=112)
        data_loader_gallery = DataLoader(dataset_gallery, batch_size=256, shuffle=False, num_workers=4)
        data_loader_probe = DataLoader(dataset_probe, batch_size=256, shuffle=False, num_workers=4)
        trainset_gallery = ijbc_dataset(Gallery, train_trf, img_size=112)
        train_loader_gallery = DataLoader(trainset_gallery, batch_size=args.batch_size, shuffle=True, num_workers=4)
    args.num_cls = 9000
    '''
    prepare encoder
    '''
    encoder = fetch_encoder.fetch(args.device, config.encoder_config,
                                  args.encoder, args.finetune_layers, args.train_output)
    encoder = encoder.cuda()

    '''
    fine-tune
    '''
    if args.finetune_layers != "None":  # for 'None', no fine-tuning is done
        if args.head_type == "arc":
            classifier = head.arcface_head(args.device, 512, args.num_cls, s=args.arc_s, m=args.arc_m, use_amp=True)
        elif args.head_type == "cos":
            classifier = head.cosface_head(512, args.num_cls, s=args.arc_s, m=args.cos_m)
        elif args.head_type == "mag":
            classifier = head.magface_head(args.device, 512, args.num_cls, s=args.arc_s, use_amp=True)
        classifier.to(args.device)

        # classifier initialization
        if args.classifier_init == 'WI':
            prototypes = weight_imprinting(args, encoder, data_loader_gallery, args.num_cls, 512)
            classifier.weight = nn.Parameter(prototypes.T)
            # prototypes = get_prototype(args, encoder, data_loader_gallery, args.num_cls, 512)
        elif args.classifier_init == 'LP':
            linear_probing(args, data_loader_gallery, encoder, classifier)
        else:
            pass  # just use random weights for classifier

        # set optimizer & LR scheduler
        optimizer = optim.Adam([{"params": encoder.parameters(), "lr": args.lr},
                                {"params": classifier.parameters(), "lr": args.lr}],
                               weight_decay=1e-3)
        # 最小学习率设置
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

        # 创建gallery dataloader
        # trainset_gallery = face_dataset(folds.G, train_trf, 112)
        # train_loader_gallery = DataLoader(trainset_gallery, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # gallery 库的样本数
        gallery_len = len(trainset_gallery)
        # args.batch_size 这个是随机选样本的比例系数
        num_random_samples = gallery_len * args.beta
        '''
            随机抽取部分base sample和novel sample一起微调模型
        '''
        # 创建 base class 加载器, 该加载器遍历所有基类样本
        train_file = args.train_file
        vgg_train_datamgr = SimpleDataManager(args.image_size, batch_size=256, num_classes=0)
        # 随机抽取样本
        data_loader_base, random_select_dataset = vgg_train_datamgr.get_random_select_data_loader(train_file,
                                                                                                  num_random_samples,
                                                                                                  shuffle=False)

        train_loader_low_conf = None
        train_loader_gallery_gallery = None
        if args.nac:
            trainset_max_pred_pseudo_ys, trainset_max_pred_items, trainset_min_pred_pseudo_ys, trainset_min_pred_items = hallucinate_label_max_pred_with_nac(
                args, data_loader_base, encoder, classifier)
            encoder = encoder.cuda()
            train_loader_low_conf = SimpleDataManager(args.image_size, batch_size=args.batch_size, num_classes=0).get_base_filt_data_loader(random_select_dataset, shuffle=False,
                                                                               filt_idx=trainset_min_pred_items.type(
                                                                                   torch.int64).numpy(),
                                                                               pseudo_label=trainset_min_pred_pseudo_ys.numpy())
        else:
            # 伪标签是一个长为5287的数组，里面每个值代表属于某个类别的概率
            trainset_max_pred_pseudo_ys, trainset_max_pred_items = hallucinate_label_max_pred(args,
                                                                                              data_loader_base,
                                                                                              encoder, classifier)
            encoder = encoder.cuda()
        '''
            用筛选后的base sample和novel sample一起微调模型
        '''
        data_loader_base = SimpleDataManager(args.image_size, batch_size=args.batch_size, num_classes=0).get_base_filt_data_loader(random_select_dataset,
                                                                       shuffle=False,
                                                                       filt_idx=trainset_max_pred_items.type(
                                                                           torch.int64).numpy(),
                                                                       pseudo_label=trainset_max_pred_pseudo_ys.numpy())

        if args.gallery_gallery:
            train_loader_gallery_gallery = DataLoader(trainset_gallery, batch_size=63, shuffle=False, num_workers=4)
        fine_tune_with_base_samples_mix_up(args, train_loader_gallery, data_loader_base, encoder, classifier,
                                           optimizer,
                                           scheduler,
                                           train_loader_low_conf,
                                           train_loader_gallery_gallery, prototypes)

    '''
    evaluate encoder
    '''

    flip = transforms.RandomHorizontalFlip(p=1)
    Gfeat = torch.FloatTensor([]).to(args.device)
    Glabel = torch.LongTensor([])
    data_loader_gallery = DataLoader(dataset_gallery, batch_size=128, shuffle=False, num_workers=8)

    for img, label in tqdm(data_loader_gallery):
        img = img.to(args.device)
        with torch.no_grad():
            feat = 0.5 * (encoder(img) + encoder(flip(img)))
        Gfeat = torch.cat((Gfeat, feat), dim=0)
        Glabel = torch.cat((Glabel, label), dim=0)

    Pfeat = torch.FloatTensor([]).to(args.device)
    Plabel = torch.LongTensor([])

    for img, label in tqdm(data_loader_probe):
        label = torch.tensor([int(i) for i in label])
        img = img.to(args.device)
        with torch.no_grad():
            feat = 0.5 * (encoder(img) + encoder(flip(img)))
        Pfeat = torch.cat((Pfeat, feat), dim=0)
        Plabel = torch.cat((Plabel, label), dim=0)

    Gfeat = Gfeat.cpu()
    Glabel = Glabel.cpu()
    Pfeat = Pfeat.cpu()

    args_dict = vars(args)


    params_list_key = ['dataset', 'encoder', 'classifier_init', 'finetune_layers', 'matcher', 'lr', 'mode', 'num_gallery', 'beta',
                       'pseudo_entropy', 'loss_type', 'num_epochs', 'alpha_gallery_base','T_dist','T_proto']

    params_list_key.append('lam_type')
    if args.lam_type == 'distance_weight':
        params_list_key.append('gamma')
    elif args.lam_type == 'uniform':
        params_list_key.append('mixup_low')
        params_list_key.append('mixup_high')
    elif args.lam_type == 'dynamic':
        params_list_key.append('lam_step')
        params_list_key.append('lam_segment')
        params_list_key.append('lam_start')
        params_list_key.append('lam_end')
    elif args.lam_type == 'beta':
        params_list_key.append('mixup_alpha')
        params_list_key.append('mixup_beta')
    else:
        params_list_key.append('gamma')
        params_list_key.append('gamma_decay')

    if args.temperature_scale:
        params_list_key.append('T')

    if args.gallery_center:
        params_list_key.append('alpha_gallery_center')
        if args.g_c_lam_type==args.lam_type:
            if args.g_c_lam_type == 'distance_weight' or args.g_c_lam_type == 'dynamic_distance_weight':
                params_list_key.append('epsilon')
            else:
                params_list_key.append('mixup_low')
                params_list_key.append('mixup_high')
        if args.g_c_lam_type == 'uniform':
            params_list_key.append('mixup_low')
            params_list_key.append('mixup_high')
    if args.gallery_gallery:
        params_list_key.append('alpha_gallery_gallery')
        if args.g_g_lam_type == 'fixed':
            params_list_key.append('lam_gallery_gallery')
        else:
            params_list_key.append('mixup_low')
            params_list_key.append('mixup_high')

    if args.nac:
        params_list_key.append('alpha_nac')

    params_list_value = [args_dict[key] for key in params_list_key]
    # save results
    save_dir_far_curve(Gfeat, Glabel, Pfeat, Plabel, save_dir)
    if not os.path.exists(args.result_file):
        header = [None, 0.1, 1.0, 10.0, 100.0, None, 0.1, 1.0, 10.0, 100.0, None, 0.1, 1.0, 10.0,
                  100.0, None, 0.1, 1.0, 10.0, 100.0, 'COS_AUC', 'NAC_AUC'] + params_list_key
        with open(args.result_file, 'w', newline='') as f:
            csv.writer(f).writerow(header)
    save_dir_far_excel_entire_experiments(args, Gfeat, Glabel, Pfeat, Plabel, params_list_value)


if __name__ == '__main__':

    # with open('seed_result.json', 'r', encoding='utf-8') as file:
    #     # 使用json.load()函数读取文件内容并转换为Python对象（通常是字典或列表）
    #     seed_result = json.load(file)
    # seeds_list = seed_result['seed_list']
    # for seed in seeds_list:
    #     for dataset in ['CASIA','IJBC']:
    #         for encoder in ['Res50', 'VGG19']:
    #             for mode in ['test']:
    #                 args.seed = seed
    #                 args.mode = mode
    #                 args.dataset = dataset
    #                 args.encoder = encoder
    #                 pprint.pprint(vars(args))
    #                 main(args)

    pprint.pprint(vars(args))
    args.device = torch.device(args.device_id)
    main(args)

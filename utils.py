import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import csv

def cosine(x,w):
    # x, w shape: [B, d], where B = batch size, d = feature dim.
    x_norm = F.normalize(x,dim=1)
    w_norm = F.normalize(w,dim=1)
    cos_sim = torch.mm(x_norm, w_norm.T).clamp(-1, 1)
    return cos_sim


def NAC(sim,k=16,s=1):
    """
    Neighborhood Aware Cosine (NAC) matcher
    args:
        sim: cosine similarity
        k: k for kNN
        s: scale (=1/T)  # In the paper scale is not used (no difference)
    returns:
        conf, pred : confidence and predicted class of shape [B,]
    """
    logit, pred = sim.topk(k,dim=1) # logit, label: [B,k]
    conf = (logit*s).softmax(1)     # norm. scale. >> use largest as conf.
    return conf[:,0],pred[:,0]      # return Top-1 confidence & prediction


def compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher="cos", nac_s=1, nac_k=16):
    num_cls = Plabel[-1].item()
    temp = torch.zeros(num_cls, Gfeat.size(1))
    for i in range(num_cls):
        mask = Glabel.eq(i)
        temp[i] = Gfeat[mask].mean(dim=0)  # make 1x512 vector
    Gfeat = temp.clone()

    num_cls = Plabel[-1].item()
    Umask = Plabel.eq(num_cls)
    Klabel = Plabel[~Umask]
    Kfeat = Pfeat[~Umask]
    Ufeat = Pfeat[Umask]

    # compute cosine similarity
    Kcos = cosine(Kfeat, Gfeat)
    Ucos = cosine(Ufeat, Gfeat)

    # get prediction & confidence
    if matcher == "cos":
        Kconf, Kidx = Kcos.max(1)
        Uconf, _ = Ucos.max(1)
    elif matcher == "NAC":
        Kconf, Kidx = NAC(Kcos, k=nac_k, s=nac_s)
        Uconf, _ = NAC(Ucos, k=nac_k, s=nac_s)

    corr_mask = Kidx.eq(Klabel)
    dir_far_tensor = torch.zeros(1000, 3)  # intervals: 1000
    for i, th in enumerate(torch.linspace(Uconf.min(), Uconf.max(), 1000)):
        mask = (corr_mask) & (Kconf > th)
        dir = torch.sum(mask).item() / Kcos.size(0)
        far = torch.sum(Uconf > th).item() / Ucos.size(0)
        dir_far_tensor[i] = torch.FloatTensor([th, dir, far])  # [threshold, DIR, FAR] for each row
    return dir_far_tensor



def dir_at_far(dir_far_tensor,far):
    # deal with exceptions: there can be multiple thresholds that meets the given FAR (e.g., FAR=1.000)
    # if so, we must choose maximum DIR value among those cases
    abs_diff = torch.abs(dir_far_tensor[:,2]-far)
    minval = abs_diff.min()
    mask = abs_diff.eq(minval)
    dir_far = dir_far_tensor[mask]
    dir = dir_far[:,1].max().item()
    return dir


# area under DIR@FAR curve
def AUC(dir_far_tensor):
    auc = 0
    eps = 1e-5
    for i in range(dir_far_tensor.size(0)-1):
        if dir_far_tensor[i,1].ge(eps) and dir_far_tensor[i,2].ge(eps)\
                and dir_far_tensor[i+1,1].ge(eps) and dir_far_tensor[i+1,2].ge(eps):
            height = (dir_far_tensor[i,1] + dir_far_tensor[i+1,1])/2
            width = torch.abs(dir_far_tensor[i,2] - dir_far_tensor[i+1,2])
            auc += (height*width).item()
    return auc


def save_dir_far_curve(Gfeat, Glabel, Pfeat, Plabel, save_dir):
    cos_tensor = compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher='cos')
    nac_tensor = compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher='NAC')
    cos_auc = AUC(cos_tensor)
    nac_auc = AUC(nac_tensor)
    fig,ax = plt.subplots(1,1)
    ax.plot(cos_tensor[:,2], cos_tensor[:,1])
    ax.plot(nac_tensor[:,2], nac_tensor[:,1])
    ax.set_xscale('log')
    ax.set_xlabel('FAR')
    ax.set_ylabel('DIR')
    ax.legend(['cos-AUC: {:.3f}'.format(cos_auc),
               'NAC-AUC: {:.3f}'.format(nac_auc)])
    ax.grid()
    fig.savefig(save_dir+'/DIR_FAR_curve.png', bbox_inches='tight')
    return cos_auc, nac_auc


def save_dir_far_excel(Gfeat, Glabel, Pfeat, Plabel, save_dir):
    cos_tensor = compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher='cos')
    nac_tensor = compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher='NAC')
    cos_list, nac_list = [], []
    for far in [0.001, 0.01, 0.1, 1.0]:
        cos_list.append('{:.2f}%'.format(dir_at_far(cos_tensor, far) * 100))
        nac_list.append('{:.2f}%'.format(dir_at_far(nac_tensor, far) * 100))
    cos_list = np.array(cos_list).reshape(1,4)
    nac_list = np.array(nac_list).reshape(1,4)
    dir_far = np.concatenate((cos_list, nac_list),axis=0)

    # save as excel
    columns = ['{:.1f}'.format(far) for far in [0.1, 1, 10, 100]]
    index = ['cos', 'NAC']
    df = pd.DataFrame(dir_far, index=index, columns=columns)
    df.to_excel(save_dir+'/DIR_FAR.xlsx')


def save_dir_far_excel_to_one_file(args, Gfeat, Glabel, Pfeat, Plabel,  params_list):
    cos_tensor = compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher='cos')
    nac_tensor = compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher='NAC')
    cos_list, nac_list = [], []
    for far in [0.001, 0.01, 0.1, 1.0]:
        cos_list.append('{:.2f}%'.format(dir_at_far(cos_tensor, far) * 100))
        nac_list.append('{:.2f}%'.format(dir_at_far(nac_tensor, far) * 100))
    cos_list = np.array(cos_list).reshape(1,4)
    nac_list = np.array(nac_list).reshape(1,4)
    cos_auc = AUC(cos_tensor)
    nac_auc = AUC(nac_tensor)

    dir_far = np.concatenate((['nac_result'], nac_list, ['cos_result'], cos_list, [cos_auc], [nac_auc], params_list), axis=0)
    with open(args.result_file, 'a', newline='') as f:
        csv.writer(f).writerow(dir_far)
def save_dir_far_excel_entire_experiments(args, Gfeat, Glabel, Pfeat, Plabel, params_list):
    cos_tensor = compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher='cos')
    nac_tensor = compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher='NAC')
    cos_list, nac_list, cos_improve_list, nac_improve_list = [], [], [], []
    # ori_results = {
    #     'CASIA-VGG19-WI-None-Cos':  [27.49,54.09,71.16,81.75],
    #     'CASIA-VGG19-WI-None-NAC':  [25.86,56.09,72.04,81.75],
    #     'CASIA-VGG19-Random-Full-NAC':[22.00,42.91,59.04,71.66],
    #     'CASIA-VGG19-LP-Full-Cos':    [28.13,55.58,71.01,80.59],
    #     'CASIA-VGG19-WI-Full-Cos':    [27.56,57.81,72.74,81.63],
    #     'CASIA-VGG19-WI-Partial-Cos': [27.36,57.89,72.89,81.84],
    #     'CASIA-VGG19-WI-PA-Cos':      [28.39,58.50,73.71,82.71],
    #     'CASIA-VGG19-WI-BN-Cos':      [29.32,57.16,73.32,82.91],
    #     'CASIA-VGG19-WI-BN-NAC':      [27.61,58.37,73.77,82.91],
    #     'CASIA-Res50-WI-None-Cos':  [27.09,58.82,75.16,84.46],
    #     'CASIA-Res50-WI-None-NAC':  [25.86,60.80,76.14,84.46],
    #     'CASIA-Res50-Random-Full-NAC':[26.84,45.79,61.25,72.85],
    #     'CASIA-Res50-LP-Full-Cos':    [28.99,61.22,75.35,83.50],
    #     'CASIA-Res50-WI-Full-Cos':    [26.49,64.70,78.19,85.43],
    #     'CASIA-Res50-WI-Partial-Cos': [26.59,65.62,78.72,85.87],
    #     'CASIA-Res50-WI-PA-Cos':      [28.09,64.78,78.49,85.79],
    #     'CASIA-Res50-WI-BN-Cos':      [28.57, 66.85, 80.38, 87.56],
    #     'CASIA-Res50-WI-BN-NAC':      [26.92,68.58,81.08,87.56]
    # }

    # with open('seed_result.json', 'r', encoding='utf-8') as file:
    #     # 使用json.load()函数读取文件内容并转换为Python对象（通常是字典或列表）
    #     seed_result = json.load(file)

    for far in [0.001, 0.01, 0.1, 1.0]:
        cos_list.append(dir_at_far(cos_tensor, far) * 100)
        nac_list.append(dir_at_far(nac_tensor, far) * 100)
    cos_list = np.array(cos_list)
    nac_list = np.array(nac_list)
    print(nac_list)
    # result_config = args.dataset+'-'+args.encoder+'-'+args.classifier_init+'-'+args.finetune_layers+'-'+args.matcher
    result_list = nac_list if args.matcher=='NAC'else cos_list


    cos_auc = AUC(cos_tensor)
    nac_auc = AUC(nac_tensor)
    # ori_result = ori_results[result_config] if result_config in ori_results else [0,0,0,0]
    # 按身份1：1划分未知身份
    # ori_result = ["48.13608229%","61.14879251%", "71.8536377%","79.83003259%"]
    # 按身份1：1划分未知身份 用锚点对齐

    if args.mode == 'test':
        if args.dataset=='IJBC':

            if args.encoder=='Res50':
                ori_result = ["39.35","59.83","72.44","81.13"]
            else:
                ori_result = ["38.04","55.52","67.52","77.25"]
        else:
            if args.encoder=='Res50':
                ori_result = ["25.75","68.73","81.04","87.54"]
            else:
                ori_result = ["26.55","57.88","73.51%","82.91"]
    else:
        if args.dataset=='IJBC':
            if args.encoder=='Res50':
                ori_result = ["42.30","60.77","72.37","81.10"]
            else:
                ori_result = ["42.50","54.02","67.59","77.32"]
        else:
            if args.encoder=='Res50':
                ori_result = ["29.99","68.47","81.12","87.58"]
            else:
                ori_result = ["29.78","58.84","73.86%","82.92"]
    improve_list = result_list-[float(i.replace("%","")) for i in ori_result]
    result_list = ['{:.2f}%'.format(far) for far in result_list]
    improve_list = ['{:.2f}%'.format(far) for far in improve_list]
    print(ori_result)
    print(improve_list)
    dir_far = np.concatenate((['OSFI_result'], ori_result, ['our_result'], result_list, ['improve'], improve_list, ['cos_result'], cos_list, [cos_auc],[nac_auc], params_list), axis=0)
    with open(args.result_file, 'a', newline='') as f:
        csv.writer(f).writerow(dir_far)












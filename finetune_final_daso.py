import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
from datamgr import SimpleDataManager
import torchvision.transforms as transforms
from model.head import cosface_head
from utils import cosine, NAC

'''
打伪标签，根据置信度筛选，做base-gallery混合
调整了应用温度参数的时机，之前是在混合之前对伪标签进行温度缩放，由于λ很小（调参后最优值在[0, 0.01]之间）
所以应用该方法获得的增益效果微乎其微。为了让模型更好的利用温度缩放，我将其用在特征混合之后.改进lam计算方式（根据离类中心的距离赋予不同样本不同的权重）
'''


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


# 我们可能会使用多个参数组来设置不同的学习率。在这里，我们取第一个参数组（索引为0），然后从中获取学习率"lr"。这样就返回了优化器的当前学习率。

def linear_probing(args, galleryloader, encoder, classifier, verbose=True, target_acc=95):
    CEloss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)
    print("start classifier training!")
    for epoch in range(args.num_epochs):
        train_corr, train_tot = 0, 0
        for img, label in tqdm(galleryloader):
            img, label = img.to(args.device), label.type(torch.int64).to(args.device)
            with torch.no_grad():
                feat = encoder(img)
            logit = classifier(feat)
            # logit, sim = classifier(feat, label)
            loss = CEloss(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            corr = torch.argmax(torch.softmax(logit, dim=0), dim=1).eq(label).sum().item()
            train_corr += corr
            train_tot += label.size(0)
        train_acc = train_corr / train_tot * 100
        scheduler.step()
        if verbose:
            print("epoch:{}, loss:{:.2f}, acc:{:.2f}%,  lr:{:.2e}".format(epoch, loss.item(),
                                                                          train_acc, get_lr(optimizer)))
        if train_acc > target_acc:
            if verbose:
                print("acc:{:.2f}%, target accuracy met. training finished".format(train_acc))
            break



def weight_imprinting(args, encoder, G_loader, num_cls, feat_dim):
    flip = transforms.RandomHorizontalFlip(p=1)
    encoder.eval()
    prototypes = torch.zeros(num_cls, feat_dim).to(args.device)
    with torch.no_grad():
        for batch, (img, label) in enumerate(G_loader):
            img, label = img.to(args.device), label.to(args.device)
            feat = 0.5 * (encoder(img) + encoder(flip(img)))
            for i in range(label.size(0)):
                prototypes[label[i]] += feat[i]
    prototypes = F.normalize(prototypes, dim=1)

    return prototypes

def get_prototype(args, encoder, G_loader, num_cls, feat_dim):
    flip = transforms.RandomHorizontalFlip(p=1)
    encoder.eval()
    prototypes = torch.zeros(num_cls, feat_dim).to(args.device)
    with torch.no_grad():
        for batch, (img, label) in enumerate(G_loader):
            img, label = img.to(args.device), label.to(args.device)
            feat = encoder(img)
            for i in range(label.size(0)):
                prototypes[label[i]] += feat[i]
    prototypes = prototypes/args.num_gallery
    return prototypes

def fine_tune(args, trainloader, encoder, classifier,
              optimizer, scheduler, verbose=True):
    CEloss = nn.CrossEntropyLoss()
    for epoch in range(args.num_epochs):
        train_corr, train_tot = 0, 0
        for img, label in trainloader:
            img, label = img.to(args.device), label.to(args.device)
            with torch.cuda.amp.autocast():
                feat = encoder(img)
                if args.head_type == "mag":  # if using MagFace head
                    logit, sim, loss_g = classifier(feat, label)
                else:
                    logit, sim = classifier(feat, label)
                loss = CEloss(logit, label)
                if args.head_type == "mag":
                    loss += loss_g
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            corr = torch.argmax(sim, dim=1).eq(label).sum().item()
            train_corr += corr
            train_tot += label.size(0)
        train_acc = train_corr / train_tot * 100
        scheduler.step()
        if verbose:
            print("epoch:{}, loss:{:.2f}, acc:{:.2f}%,  lr:{:.2e}".format(epoch, loss.item(),
                                                                          train_acc, get_lr(optimizer)))


'''
筛选条件不同：

第一段代码（hallucinate_label）在筛选伪标签时，使用的是最大预测值（max_pred）大于一个阈值 args.pseudo_entropy 的样本。
第二段代码（hallucinate_label_）计算每个样本伪标签分布的熵（base_entropy），并选择熵值低于 args.pseudo_entropy 的样本。
伪标签质量评估标准不同：

第一段代码基于softmax后的最大概率值作为衡量伪标签“确定性”的依据，认为最大概率超过一定阈值的样本更适合用于伪标签生成。
第二段代码则是基于信息熵理论，选取熵较低的样本，意味着这些样本的伪标签分布更集中，即更加确定。
输出排序不同：

在第一段代码中，针对筛选出的样本，还进行了按照最大预测值排序的操作，然后对应地更新了伪标签和索引的顺序。
第二段代码则没有这个排序步骤。
'''


# 最大预测大于 args.pseudo_entropy 的样本
def hallucinate_label_max_pred(args, vggface2, encoder, classifier):
    encoder = encoder.eval()
    # Use the trained model (feature extractor + the newly learned LinearRegression model) to Pseudo label the base set
    max_items = torch.Tensor([])  # 存储每个基准集样本的索引
    max_pseudo_ys = torch.Tensor([])  # 存储每个基准集样本的伪软标签（实际上是logits）
    max_preds = torch.Tensor([])  # 存储每个基准集样本的最大类别预测值
    max_pseudo_ys_set = torch.Tensor([])

    print("start hallucinating label!")
    batch = 0
    for img, label, item in tqdm(vggface2):  # 遍历VGGFace2数据集中的每个样本
        # 将输入数据移动到指定设备并转换为所需数据类型
        img, label, item = img.to(args.device), label.type(torch.int64).to(args.device), item.to(args.device)
        # 获取样本的特征表示
        with torch.no_grad():
            feat = encoder(img)
            # 使用分类器计算样本特征对应的logits和相似度
            logit, sim = classifier(feat, label)
            # 伪标签,softmax函数处理后可以转换为概率分布，每个类别的概率是该类别对应的logit经过softmax函数转换后的值。
            pseudo_ys = torch.softmax(logit, dim=1)
            # 筛选最大预测大于 args.pseudo_entropy 的样本
            pred, idx = torch.max(pseudo_ys, dim=1)
            max_index = torch.where(pred > args.pseudo_entropy)
            max_pred = pred[max_index]
            max_pred_pseudo_label = pseudo_ys[max_index]
            max_pred_item = item[max_index]
            # 将GPU内存中的张量复制到CPU内存并解除梯度跟踪
            max_pred_pseudo_label = max_pred_pseudo_label.detach().cpu()
            max_pred_item = max_pred_item.detach().cpu()

            item = item.detach().cpu()
            pseudo_ys = pseudo_ys.detach().cpu()
            max_pred = max_pred.detach().cpu()

            # 将筛选出的高置信度样本的伪标签、索引和最大预测值追加到相应的存储变量中
            max_pseudo_ys_set = torch.concatenate((max_pseudo_ys_set, max_pred_pseudo_label), dim=0)
            max_items = torch.concatenate((max_items, max_pred_item), dim=0)
            max_preds = torch.concatenate((max_preds, max_pred), dim=0)
            if batch % 100 == 0:
                max_pseudo_ys = torch.concatenate((max_pseudo_ys, max_pseudo_ys_set), dim=0)
                max_pseudo_ys_set = torch.Tensor([])
            torch.cuda.empty_cache()  # 清空GPU缓存，释放内存
            batch += 1
    max_pseudo_ys = torch.concatenate((max_pseudo_ys, max_pseudo_ys_set), dim=0)
    # descending=True 降序# 对最大预测值进行降序排序
    trainset_max_preds, index = torch.sort(max_preds, descending=True)
    # 根据排序后的索引重新组织最大预测值、伪标签和样本索引
    trainset_max_pseudo_ys = max_pseudo_ys[index]
    trainset_max_items = max_items[index]

    #返回处理后的高置信度样本的伪标签、索引，以及未正确筛选和更新的低置信度样本信息
    return trainset_max_pseudo_ys, trainset_max_items




# 取最大预测区间， 做验证实验
def hallucinate_label_max_pred_range(args, vggface2, encoder, classifier):
    encoder = encoder.eval()
    # Use the trained model (feature extractor + the newly learned LinearRegression model) to Pseudo label the base set
    items = torch.Tensor([])  # storing index for each base set example
    pseudo_ys_set = torch.Tensor([])  # storing pseudo soft label (logits actually) for each base set example
    pseudo_ys_set_ = torch.Tensor([])  # storing pseudo soft label (logits actually) for each base set example
    base_pred_set = torch.Tensor([])

    batch = 0
    print("start hallucinating label!")
    for img, label, item in tqdm(vggface2):
        img, label, item = img.to(args.device), label.type(torch.int64).to(args.device), item.to(args.device)
        with torch.no_grad():
            feat = encoder(img)
        logit, sim = classifier(feat, label)
        # 伪标签
        pseudo_ys = torch.softmax(logit, dim=1)
        base_pred, idx = torch.max(pseudo_ys, dim=1)
        # 计算伪标签的熵
        # base_entropy = -torch.sum(pseudo_ys * torch.log2(pseudo_ys), dim=1)
        item = item.detach().cpu()
        pseudo_ys = pseudo_ys.detach().cpu()
        base_pred = base_pred.detach().cpu()
        items = torch.concatenate((items, item), dim=0)
        base_pred_set = torch.concatenate((base_pred_set, base_pred), dim=0)
        pseudo_ys_set_ = torch.concatenate((pseudo_ys_set_, pseudo_ys), dim=0)
        if batch % 100 == 0:
            pseudo_ys_set = torch.concatenate((pseudo_ys_set, pseudo_ys_set_), dim=0)
            pseudo_ys_set_ = torch.Tensor([])

        torch.cuda.empty_cache()
        batch += 1
    pseudo_ys_set = torch.concatenate((pseudo_ys_set, pseudo_ys_set_), dim=0)
    # 从小到大排序
    base_entropy_sort, index = torch.sort(base_pred_set, descending=True)
    pseudo_ys_set = pseudo_ys_set[index]
    items = items[index]

    total_len = len(base_pred_set)
    filt_low = int(total_len * args.filter_partition_low)
    filt_high = int(total_len * args.filter_partition_high)
    low_entropy_ys_set = pseudo_ys_set[filt_low:filt_high]
    low_entropy_item_set = items[filt_low:filt_high]
    return low_entropy_ys_set, low_entropy_item_set  # , high_entropy_ys_set, high_entropy_item_set


# 筛选熵值熵低于 pseudo_entropy 的样本
def hallucinate_label_filt_entropy(args, vggface2, encoder, classifier):
    encoder = encoder.eval()
    # Use the trained model (feature extractor + the newly learned LinearRegression model) to Pseudo label the base set
    trainset_items = torch.Tensor([])  # storing index for each base set example
    trainset_pseudo_ys = torch.Tensor([])  # storing pseudo soft label (logits actually) for each base set example
    trainset_entropy = torch.Tensor([])  # storing entropy for each base set example
    print("start hallucinating label!")
    for img, label, item in tqdm(vggface2):
        img, label, item = img.to(args.device), label.type(torch.int64).to(args.device), item.to(args.device)
        with torch.no_grad():
            feat = encoder(img)
        logit, sim = classifier(feat, label)
        # 伪标签
        pseudo_ys = torch.softmax(logit, dim=1)
        # 计算伪标签的熵
        base_entropy = -torch.sum(pseudo_ys * torch.log(pseudo_ys), dim=1)
        # 筛选熵低于 pseudo_entropy 的样本
        index = torch.where(base_entropy < args.pseudo_entropy)
        low_entropy_pseudo_label = pseudo_ys[index]
        low_entropy_item = item[index]
        base_entropy = base_entropy[index]
        low_entropy_pseudo_label = low_entropy_pseudo_label.detach().cpu()
        low_entropy_item = low_entropy_item.detach().cpu()

        item = item.detach().cpu()
        pseudo_ys = pseudo_ys.detach().cpu()
        base_entropy = base_entropy.detach().cpu()
        trainset_pseudo_ys = torch.concatenate((trainset_pseudo_ys, low_entropy_pseudo_label), dim=0)
        trainset_items = torch.concatenate((trainset_items, low_entropy_item), dim=0)
        trainset_entropy = torch.concatenate((trainset_entropy, base_entropy), dim=0)
        torch.cuda.empty_cache()

    trainset_entropy, index = torch.sort(trainset_entropy)
    trainset_pseudo_ys = trainset_pseudo_ys[index]
    trainset_items = trainset_items[index]
    return trainset_pseudo_ys, trainset_items


# 最大预测, 筛选预测值最大和最小的两部分, 预测值大的用于微调, 预测值小的用于计算NAC损失
def hallucinate_label_max_pred_with_nac(args, vggface2, encoder, classifier):
    encoder = encoder.eval()
    # Use the trained model (feature extractor + the newly learned LinearRegression model) to Pseudo label the base set
    items = torch.Tensor([])  # storing index for each base set example
    pseudo_ys_set = torch.Tensor([])  # storing pseudo soft label (logits actually) for each base set example
    pseudo_ys_set_ = torch.Tensor([])  # storing pseudo soft label (logits actually) for each base set example
    base_pred_set = torch.Tensor([])

    batch = 0
    print("start hallucinating label!")
    for img, label, item in tqdm(vggface2):
        img, label, item = img.to(args.device), label.type(torch.int64).to(args.device), item.to(args.device)
        with torch.no_grad():
            feat = encoder(img)
        logit, sim = classifier(feat, label)
        # 伪标签
        pseudo_ys = torch.softmax(logit, dim=1)
        base_pred, idx = torch.max(pseudo_ys, dim=1)

        # 计算伪标签的熵
        # base_entropy = -torch.sum(pseudo_ys * torch.log2(pseudo_ys), dim=1)
        item = item.detach().cpu()
        pseudo_ys = pseudo_ys.detach().cpu()
        base_pred = base_pred.detach().cpu()
        items = torch.concatenate((items, item), dim=0)
        base_pred_set = torch.concatenate((base_pred_set, base_pred), dim=0)
        pseudo_ys_set_ = torch.concatenate((pseudo_ys_set_, pseudo_ys), dim=0)
        if batch % 100 == 0:
            pseudo_ys_set = torch.concatenate((pseudo_ys_set, pseudo_ys_set_), dim=0)
            pseudo_ys_set_ = torch.Tensor([])

        torch.cuda.empty_cache()
        batch += 1
    pseudo_ys_set = torch.concatenate((pseudo_ys_set, pseudo_ys_set_), dim=0)
    # 从小到大排序
    base_entropy_sort, index = torch.sort(base_pred_set, descending=True)
    pseudo_ys_set = pseudo_ys_set[index]
    items = items[index]

    total_len = len(base_pred_set)
    filt_len = int(total_len * args.filter_partition)
    # filt_high = int(total_len * args.filter_partition_high)
    high_pred_ys_set = pseudo_ys_set[:filt_len]
    high_pred_item_set = items[:filt_len]
    low_pred_ys_set = pseudo_ys_set[-filt_len:]
    low_pred_item_set = items[-filt_len:]
    return high_pred_ys_set, high_pred_item_set, low_pred_ys_set, low_pred_item_set


class DistillKL(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        # 存储超参数T，通常被称为“温度”，在知识蒸馏中用于调整学生网络和教师网络输出的软化程度
        self.T = T

    def forward(self, y_s, y_t):
        # 对学生网络的输出y_s应用log_softmax函数，并除以温度T，使得分布更平滑利于蒸馏
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        # 计算KL散度损失，即学生网络预测分布p_s与教师网络预测分布p_t之间的KL散度
        # 使用'reduction'参数设置为'sum'，表示将损失在每个样本上的KL散度求和
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss


# 定义一个函数，它根据当前训练进度（epoch, batch_i）来生成lam
def get_dynamic_mixup_bounds(total_epoch, epoch, total_batch, batch):
    epoch_low = epoch * (1 / total_epoch)
    # epoch_high = (epoch+1) * (1 / total_epoch)
    batch_low = epoch_low + batch * (1 / total_batch) * (1 / total_epoch)
    batch_high = epoch_low + (batch + 1) * (1 / total_batch) * (1 / total_epoch)
    lam = np.random.uniform(batch_low, batch_high)
    return lam


def dynamic_lam_generator(start, end, step, segment):
    slide_step = (end - (start + step)) / (segment - 1)
    lam_list = [[i * slide_step, step + i * slide_step] for i in range(segment)]
    return lam_list


def fine_tune_with_base_samples_mix_up(args, trainloader_g, trainloader_b, encoder, classifier, optimizer, scheduler,
                                       trainloader_low_conf, trainloader_g_g, prototypes):
    CEloss = nn.CrossEntropyLoss()
    # 计算 gallery和base sample的batch数
    train_iter_g = len(trainloader_g)
    train_iter_b = len(trainloader_b)

    print('base_batchs = ' + str(train_iter_b))
    print('gallery_batchs = ' + str(train_iter_g))
    if args.lam_type == 'dynamic':
        lam_list = dynamic_lam_generator(args.lam_start, args.lam_end, args.lam_step, args.lam_segment)
    similarity_fn = nn.CosineSimilarity(dim=2)
    train_iter = max(train_iter_g, train_iter_b)
    for epoch in range(args.num_epochs):
        total_loss_gallery = 0
        total_loss_base_mix = 0
        total_loss_gallery_gallery_mix = 0
        total_loss_gallery_center_mix = 0
        total_loss_nac = 0

        loss_gallery_gallery_mix = 0
        loss_gallery_center_mix = 0
        loss_nac = 0

        prototypes = weight_imprinting(args, encoder, trainloader_g, args.num_cls, 512)
        classifier.load_state_dict({'weight': nn.Parameter(prototypes.T)})

        daso_pseudo = None
        train_corr, train_tot = 0, 0
        # 获取gallery 和 base sample 的迭代器， 每调用一次 next() 函数就会返回一个batch的样本
        trainloader_g_iter = iter(trainloader_g)
        trainloader_b_iter = iter(trainloader_b)
        if args.nac: trainloader_low_conf_iter = iter(trainloader_low_conf)
        if args.gallery_gallery: trainloader_g_g_iter = iter(trainloader_g_g)



        debaised_pred = []
        for batch_i in tqdm(range(train_iter - 1)):
            # 计算gallery 库的损失
            try:
                img_g, label_g = next(trainloader_g_iter)
                img_g, label_g = img_g.to(args.device), label_g.to(args.device).type(torch.int64)
                with torch.cuda.amp.autocast():
                    feat_g = encoder(img_g).type(torch.float32)
                if args.head_type == "mag":  # if using MagFace head
                    logit_g, sim_g, loss_g = classifier(feat_g, label_g)
                else:
                    logit_g, sim_g = classifier(feat_g, label_g)
                loss_gallery = CEloss(logit_g, label_g)
                if args.head_type == "mag":
                    loss_gallery += loss_g

                if args.gallery_center:
                    with torch.cuda.amp.autocast():
                        prototype = classifier.weight.T[label_g]
                        if args.lam_type == 'distance_weight':
                            lam = torch.div(1, torch.pow(1 + torch.linalg.norm(feat_g - prototype, dim=-1),
                                                         args.epsilon)).view(-1, 1)
                        elif args.lam_type == 'dynamic':
                            lam = np.random.uniform(lam_list[epoch // args.lam_segment][0],
                                                    lam_list[epoch // args.lam_segment][1])
                        elif args.lam_type == 'uniform':
                            lam = np.random.uniform(args.mixup_low, args.mixup_high)
                        else:
                            lam = torch.div(1, torch.pow(1 + torch.linalg.norm(feat_g - prototype, dim=-1),
                                                         args.epsilon * args.gamma_decay ** epoch)).view(-1, 1)
                        feat_g_mix = feat_g * lam + prototype * (1 - lam)
                    if args.head_type == "mag":  # if using MagFace head
                        logit_g_mix, sim_g_mix, loss_g_mix = classifier(feat_g_mix, label_g)
                    else:
                        logit_g_mix, sim_g_mix = classifier(feat_g_mix, label_g)
                    loss_gallery_center_mix = CEloss(logit_g_mix, label_g)
                    if args.head_type == "mag":
                        loss_gallery_center_mix += loss_g_mix
                    loss_gallery_center_mix *= args.alpha_gallery_center

                corr_g = torch.argmax(sim_g, dim=1).eq(label_g).sum().item()
                train_corr += corr_g
                train_tot += label_g.size(0)
                total_loss_gallery += loss_gallery
                total_loss_gallery_center_mix += loss_gallery_center_mix

            except StopIteration:
                loss_gallery = 0
                loss_gallery_center_mix = 0

            # 计算base class的损失
            try:
                img_b, label_b, _ = next(trainloader_b_iter)
                img_b, label_b = img_b.to(args.device).type(torch.float32), label_b.to(args.device)
                with torch.cuda.amp.autocast():
                    feat_b = encoder(img_b).type(torch.float32)

                # 在伪标签的情况下进行mix-up混合
                pred_count, pred_value = torch.max(label_b, dim=1)
                one_hot = torch.zeros_like(label_b).to(args.device)
                one_hot.scatter_(1, pred_value.view(-1, 1), 1.0)
                prototype = classifier.weight.T[pred_value]
                if args.lam_type == 'distance_weight':
                    lam = torch.div(1, torch.pow(1 + torch.linalg.norm(feat_b - prototype, dim=-1),
                                                 args.gamma)).view(-1, 1)
                elif args.lam_type == 'dynamic':
                    lam = np.random.uniform(lam_list[epoch // args.lam_segment][0],
                                            lam_list[epoch // args.lam_segment][1])
                elif args.lam_type == 'uniform':
                    lam = np.random.uniform(args.mixup_low, args.mixup_high)
                elif args.lam_type == 'beta':
                    lam = np.random.beta(args.mixup_alpha, args.mixup_beta)
                else:
                    lam = torch.div(1, torch.pow(1 + torch.linalg.norm(feat_b - prototype, dim=-1),
                                                 args.gamma * args.gamma_decay ** epoch)).view(-1, 1)
                feat_mix = feat_b * lam + prototype * (1 - lam)
                label_mix = label_b * lam + one_hot * (1 - lam)
                if args.temperature_scale:
                    label_mix = torch.softmax(label_mix / args.T, dim=1)

                if args.head_type == "mag":  # if using MagFace head
                    logit_b, sim_b, loss_b = classifier(feat_mix, label_mix)
                else:
                    logit_b, sim_b = classifier(feat_mix, label_mix)
                    with torch.no_grad():
                        prototypes = classifier.weight.T
                        sim_logit_b = similarity_fn(feat_mix.reshape(-1,1,512), prototypes.repeat(feat_mix.size(0), 1, 1)) / args.T_proto
                        q_hat = sim_logit_b.softmax(dim=1)
                if args.loss_type == 'CE':
                    loss_base = CEloss(logit_b, label_mix)
                elif args.loss_type == 'KL':
                    if args.temperature_scale:
                        y_s = F.log_softmax(logit_b / args.T, dim=1)
                        p_hat = torch.exp(y_s)
                        confidence, pred_class = torch.max(p_hat.detach(), dim=1)
                        if daso_pseudo == None:
                            daso_pseudo = p_hat.mean(dim=0).data
                            p = daso_pseudo
                        else:
                            m = daso_pseudo ** (1. / args.T_dist)
                            m = m / m.sum()
                            m = m / m.max()
                            pred_to_dist = m[pred_class].view(-1, 1)
                            # entropy = -torch.sum(p_hat * torch.log(p_hat + 1e-9), dim=-1)
                            # p = (1 - pred_to_dist) * p_hat + pred_to_dist * q_hat * entropy.unsqueeze(-1)
                            p = (1. - pred_to_dist) * p_hat + pred_to_dist * q_hat
                            daso_pseudo = daso_pseudo + p.sum(dim=0).data
                            daso_pseudo = daso_pseudo / daso_pseudo.sum(dim=0)

                            debaised_pred = debaised_pred + torch.max(p,dim=1)[1].detach().cpu().numpy().tolist()
                        loss_base = F.kl_div(torch.log(p), label_mix, reduction='sum') * (args.T ** 2) / y_s.shape[0]
                    else:
                        y_s = F.log_softmax(logit_b , dim=1)
                        loss_base = F.kl_div(y_s, label_mix, reduction='sum') / y_s.shape[0]
                else:
                    print('error loss type (CE or KL)!  the following result is calculated by CE')
                    loss_base = CEloss(logit_b, label_mix)
                if args.head_type == "mag":
                    loss_base += loss_b
                loss_base *= args.alpha_gallery_base
                total_loss_base_mix += loss_base
            except StopIteration:
                loss_base = 0

            # 计算gallery 与 gallery 混合的损失
            if args.gallery_gallery:
                try:
                    img_g, label_g = next(trainloader_g_g_iter)
                    img_g, label_g = img_g.to(args.device), label_g.to(args.device).type(torch.int64)
                    with torch.cuda.amp.autocast():
                        feat_g = encoder(img_g).type(torch.float32)
                        length = len(feat_g) // 3 * 3
                        feat_g = feat_g[:length]
                        label_g = label_g[:length]
                        if args.g_g_lam_type == 'uniform':
                            args.lam_gallery_gallery = np.random.uniform(args.mixup_low, args.mixup_high)
                        f1 = feat_g[::3] * args.lam_gallery_gallery + feat_g[1::3] * (1 - args.lam_gallery_gallery)
                        f2 = feat_g[::3] * args.lam_gallery_gallery + feat_g[2::3] * (1 - args.lam_gallery_gallery)
                        f3 = feat_g[1::3] * args.lam_gallery_gallery + feat_g[2::3] * (1 - args.lam_gallery_gallery)
                        l = label_g[::3]
                        feat_g = torch.concatenate([f1, f2, f3])
                        label_g = torch.concatenate([l, l, l])
                    if args.head_type == "mag":  # if using MagFace head
                        logit_g, sim_g, loss_g = classifier(feat_g, label_g)
                    else:
                        logit_g, sim_g = classifier(feat_g, label_g)
                    loss_gallery_gallery_mix = CEloss(logit_g, label_g)
                    if args.head_type == "mag":
                        loss_gallery_gallery_mix += loss_g
                    loss_gallery_gallery_mix *= args.alpha_gallery_gallery
                    total_loss_gallery_gallery_mix += loss_gallery_gallery_mix
                except StopIteration:
                    loss_gallery_gallery_mix = 0

            if args.nac:
                try:
                    img_b, label_b, _ = next(trainloader_low_conf_iter)
                    img_b, label_b = img_b.to(args.device).type(torch.float32), label_b.to(args.device)
                    with torch.cuda.amp.autocast():
                        feat_b = encoder(img_b).type(torch.float32)
                    cos_sim = cosine(feat_g, feat_b)
                    conf, idx = NAC(cos_sim, k=16, s=1)
                    Nac = torch.sum(conf) / cos_sim.size(0)
                    loss_nac = Nac * args.alpha_nac
                    total_loss_nac += loss_nac
                except StopIteration:
                    loss_nac = 0

            loss = loss_gallery + loss_base + loss_gallery_center_mix + loss_gallery_gallery_mix + loss_nac

            if loss != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        train_acc = train_corr / train_tot * 100
        result = "epoch:{}, loss_gallery:{:.2f}, loss_base_mix:{:.2f}".format(
            epoch, total_loss_gallery / train_iter_g,
                   total_loss_base_mix / train_iter_b)
        if args.gallery_gallery:
            result += ", loss_g_g_mix: {:.2f}".format(total_loss_gallery_gallery_mix / len(trainloader_g_g))
        if args.gallery_center:
            result += ", loss_g_c_mix: {:.2f}".format(
                total_loss_gallery_center_mix / train_iter_g)
        if args.nac:
            result += ", loss_nac:{:.2f}".format(
                total_loss_nac / len(trainloader_low_conf))
        result += ", acc: {:.2f}%, lr: {:.2e}".format(train_acc, get_lr(optimizer))
        print(result)
        # # 计算每个类别的出现次数
        # from collections import Counter
        # category_counts = Counter(debaised_pred)
        # # 根据频率数从大到小排序
        # sorted_category_counts = category_counts.most_common()
        # # 创建DataFrame
        # df = pd.DataFrame(sorted_category_counts, columns=['类别', '数量'])
        # # 保存到CSV文件
        # df.to_csv('category_distribution_after.csv', index=False)
        scheduler.step()

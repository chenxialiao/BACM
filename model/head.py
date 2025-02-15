import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import Parameter


class softmax_head(nn.Module):
    def __init__(self, feat_dim, num_cls):
        super(softmax_head, self).__init__()
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))

    def forward(self, x, label):
        logit = torch.mm(x, self.weight)
        return logit, logit


class normface_head(nn.Module):
    def __init__(self, feat_dim, num_cls, s=32):
        super(normface_head, self).__init__()
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.s = s
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))
        nn.init.xavier_uniform_(self.weight)#快速收敛

    def forward(self, x, label):
        x_norm = F.normalize(x,dim=1)
        w_norm = F.normalize(self.weight,dim=0)
        cosine = torch.mm(x_norm, w_norm).clamp(-1, 1)
        logit = self.s * cosine
        return logit, cosine


#CosFace head
class cosface_head(nn.Module):
    def __init__(self, feat_dim, num_cls, s=32, m=0.35):
        super(cosface_head, self).__init__()
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))
        nn.init.xavier_uniform_(self.weight)
        #print(self.weight)

    def forward(self, x, label):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=0)
        cosine = torch.mm(x_norm, w_norm).clamp(-1, 1)
        # 根据label的形状创建one-hot编码标签
        # 如果label是一维向量，则使用scatter_方法创建one-hot编码；否则直接使用label
        if len(label.size()) == 1:
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1.0)
        else:
            one_hot = label
        #print("one_hot,scatter", one_hot)
        logit = self.s * (cosine - one_hot * self.m)
        #print("Logit:", logit)



        return logit, cosine



class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=torch.tensor([0.2, 0.3, 0.5,1])):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.min_alpha = min(self.alpha.values())

    def forward(self, pt, target):
        logpt = nn.functional.log_softmax(pt, dim=1) #计算softmax后在计算log
        pt = torch.exp(logpt) #对log_softmax去exp，把log取消就是概率
        # logpt = torch.log(pt)
        alpha=[self.alpha.get(t.item(),self.min_alpha) for t in target] # 去取真实索引类别对应的alpha
        alpha = torch.tensor(alpha).repeat(9000, 1).T.to(device=0)
        logpt = alpha * (1 - pt) ** self.gamma * logpt #focal loss计算公式
        loss = nn.functional.nll_loss(logpt, target,reduction='sum') # 最后选择对应位置的元素，ys与labelmix
        return loss

class FocalLoss_(nn.Module):
    def __init__(self, gamma=2, alpha=torch.tensor([0.2, 0.3, 0.5,1])):
        super(FocalLoss_, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.min_alpha = min(self.alpha.values())

    def forward(self, pt, target):
        # logpt = nn.functional.log_softmax(pt, dim=1) #计算softmax后在计算log
        # pt = torch.exp(logpt) #对log_softmax去exp，把log取消就是概率
        logpt = torch.log(pt)
        alpha=[self.alpha.get(t.item(),self.min_alpha) for t in target] # 去取真实索引类别对应的alpha
        alpha = torch.tensor(alpha).repeat(9000, 1).T.to(device=0)
        logpt = alpha * (1 - pt) ** self.gamma * logpt #focal loss计算公式
        loss = nn.functional.nll_loss(logpt, target,reduction='sum') # 最后选择对应位置的元素，ys与labelmix
        return loss

class BCLoss(nn.Module):
    def __init__(self,  alpha, beta, gamma=2):
        super(BCLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.min_alpha = min(self.alpha.values())

    def forward(self, pt, target):
        # logpt = nn.functional.log_softmax(input, dim=1) #计算softmax后在计算log
        # pt = torch.exp(logpt) #对log_softmax去exp，把log取消就是概率
        logpt = torch.log(pt)
        alpha=[self.alpha.get(t.item(),self.min_alpha) for t in target] # 去取真实索引类别对应的alpha
        alpha = torch.tensor(alpha).repeat(9000, 1).T.to(device=0)
        E_n = (1-self.beta)/(1-self.beta**alpha)
        logpt = E_n * (1 - pt) ** self.gamma * logpt #focal loss计算公式
        loss = nn.functional.nll_loss(logpt, target,reduction='sum') # 最后选择对应位置的元素，ys与labelmix
        return loss
# #类内方差
# class cosface_head(nn.Module):
#     def __init__(self, feat_dim, num_cls, s=32, m=0.35):
#         super(cosface_head, self).__init__()
#         self.feat_dim = feat_dim
#         self.num_cls = num_cls
#         self.s = s
#         self.m = m
#         self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))
#         nn.init.xavier_uniform_(self.weight)
#         #print(self.weight)
#
#     def forward(self, x, label):
#         x_norm = F.normalize(x, dim=1)
#         w_norm = F.normalize(self.weight, dim=0)
#         cosine = torch.mm(x_norm, w_norm).clamp(-1, 1)
#         # print("Cosine:", cosine)
#         one_hot = torch.zeros_like(cosine)
#         one_hot.scatter_(1, label.view(-1, 1), 1.0)
#         # print("one_hot,scatter", one_hot)
#         logit = self.s * (cosine - one_hot * self.m)
#         # print("Logit:", logit)
#         cosine_filtered = cosine * one_hot
#
#         # print(cosine_filtered)
#
#         # Convert cosine to angle
#         angle_rad = torch.acos(cosine_filtered)
#         # print(angle_rad)
#         angle_deg = torch.rad2deg(angle_rad)
#         # print(angle_deg)
#         # Calculate average angle for each class
#         angle_filtered = angle_deg[angle_deg != 90]
#         angles_avg = []
#         all_angles = []
#         for i in range(self.num_cls):
#             cls_angles = angle_filtered[label == i]
#             # print("Class", i + 1, "angles:", cls_angles)
#             angle_avg = torch.mean(cls_angles)
#             angles_avg.append(angle_avg.item())
#             # print("Class", i + 1, "average angle:", angle_avg)
#             all_angles.append(cls_angles)
#
#         # Calculate average angle across all classes
#         all_avg_angle = torch.mean(torch.cat(all_angles))
#         # print("Average angle across all classes:", all_avg_angle)
#         # 计算所有类别的平均角度
#         avg_angle_all_classes = all_avg_angle.item()
#
#         # 输出平均角度
#         print("Average angle across all classes: {:.4f}".format(avg_angle_all_classes))
#         return logit, cosine

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class AdaFace(nn.Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.weight = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings,  label):
        norms = torch.norm(embbedings, 2, -1, keepdim=True)
        embbedings = embbedings / norms

        kernel_norm = l2_norm(self.weight,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m, cosine



class ElasticArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50,std=0.0125,plus=False):
        super(ElasticArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.weight, std=0.01)
        self.std=std
        self.plus=plus
    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.weight, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device) # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = cos_theta[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta, cos_theta


class ElasticCosFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35,std=0.0125, plus=False):
        super(ElasticCosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.weight, std=0.01)
        self.std=std
        self.plus=plus

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.weight, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device)  # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = cos_theta[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret, cos_theta



# ArcFace head
class arcface_head(nn.Module):
    def __init__(self,device,feat_dim,num_cls,s,m,
                 easy_margin=True,use_amp=True):
        super(arcface_head,self).__init__()
        self.device = device
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.s = s
        self.m = torch.Tensor([m]).to(device)
        self.use_amp = use_amp
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, x, label):
        cos_m, sin_m = torch.cos(self.m),torch.sin(self.m)
        x_norm = F.normalize(x,dim=1)
        w_norm = F.normalize(self.weight,dim=0)
        cos_theta = torch.mm(x_norm, w_norm).clamp(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.use_amp:
            cos_theta_m = cos_theta_m.to(torch.float16)

        # easy_margin is to stabilize training:
        # i.e., if model is initialized badly s.t. theta + m > pi, then will not use arcface loss!
        if self.easy_margin:
            min_cos_theta = torch.cos(math.pi - self.m)
            cos_theta_m = torch.where(cos_theta > min_cos_theta, cos_theta_m, cos_theta)

        idx = torch.zeros_like(cos_theta)
        idx.scatter_(1, label.data.view(-1, 1), 1)
        logit = cos_theta_m * idx + cos_theta * (1-idx)
        logit *= self.s
        '''

首先，`idx` 是一个与 `cos_theta` 维度相同的张量，其初始化为全零。它用于创建一个独热编码（one-hot encoding）的标签张量，其中 `label` 是一个存储了样本类别标签的张量。通过使用 `scatter_` 函数，在 `idx` 张量中将每个样本的类别位置（通过 `label` 张量的值查找）设置为 1，其他位置保持为 0。这样就实现了独热编码的效果。

接下来，`logit` 是通过一定的计算得到的输出张量。这里的计算包括两部分：

1. `cos_theta_m * idx`：将 `cos_theta_m` 与 `idx` 相乘，实现了对具有正确类别标签的样本的加权处理。`cos_theta_m` 是输入经过余弦余量处理的特征向量（在前文中提到的 `cos_m`）。
2. `cos_theta * (1-idx)`：将 `cos_theta` 与 `(1-idx)` 相乘，实现了对具有错误类别标签的样本的加权处理。`(1-idx)` 将独热编码中的 1 与 0 进行了反转。

最后，`logit` 张量乘以 `self.s`，其中 `self.s` 是一个缩放因子（在前文中提到的 `--arc_s`），这样可以对输出进行缩放，强调类别特征的差异性。


        '''

        return logit, cos_theta



class magface_head(nn.Module):
    def __init__(self,device,feat_dim,num_cls,s,use_amp=True, easy_margin=True,
                 l_a=10, u_a=110, l_m=0.35, u_m=0.7, l_g=40):
        super(magface_head,self).__init__()
        self.feat_dim = feat_dim
        self.num_cls = num_cls
        self.device = device
        self.s = s
        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m
        self.l_g = l_g
        self.use_amp = use_amp
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_cls))
        nn.init.xavier_uniform_(self.weight)

    def calc_m(self, mag):
        return (self.u_m - self.l_m) / (self.u_a - self.l_a) * (mag - self.l_a) + self.l_m

    def forward(self, x, label=None):
        mag = torch.norm(x, dim=1, keepdim=True)
        mag = mag.clamp(self.l_a, self.u_a)
        m_a = self.calc_m(mag)
        cos_m, sin_m = torch.cos(m_a), torch.sin(m_a)
        g_a = 1 / mag + 1 / (self.u_a ** 2) * mag
        loss_g = self.l_g * g_a.mean()

        x_norm = F.normalize(x,dim=1)
        w_norm = F.normalize(self.weight,dim=0)
        cos_theta = torch.mm(x_norm, w_norm).clamp(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.use_amp:
            cos_theta_m = cos_theta_m.to(torch.float16)

        # easy_margin is to stabilize training:
        # i.e., if model is initialized badly s.t. theta + m > pi, then will not use arcface loss!
        if self.easy_margin:
            min_cos_theta = torch.cos(math.pi - m_a)
            cos_theta_m = torch.where(cos_theta > min_cos_theta, cos_theta_m, cos_theta)

        idx = torch.zeros_like(cos_theta)
        idx.scatter_(1, label.data.view(-1, 1), 1)
        logit = cos_theta_m * idx + cos_theta * (1-idx)
        logit *= self.s

        return logit, cos_theta, loss_g

import collections
from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.cuda import amp


class CM(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None




class CM_Hard(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None

def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().detach().numpy()

# def compute_similarity(inputs, centers, max_distance=None):
#     """
#     计算 0~1 范围内的相似度，与欧氏距离强相关。
    
#     参数:
#         inputs: [B, dim] 输入特征
#         centers: [B, dim] 类中心特征
#         max_distance: float, 特征空间中可能的最大距离。如果为 None，则自动计算。
    
#     返回:
#         similarity: [B] 相似度，范围在 0~1 之间
#     """
#     # 计算欧氏距离
#     dist = euclidean_distance(inputs, centers)  # [B, B]
#     similarity = 1 / (1 + dist)    # [query_num, gallery_num]
#     # 返回对角线元素
#     similarity = torch.from_numpy(similarity).float().to(inputs.device)  # 转换为 float 类型的张量
#     diagonal = torch.diag(similarity)

#     return diagonal

def compute_similarity(inputs, centers, max_distance=None):
    """
    计算 0~1 范围内的相似度，与欧氏距离强相关。
    
    参数:
        inputs: [B, dim] 输入特征
        centers: [B, dim] 类中心特征
        max_distance: float, 特征空间中可能的最大距离。如果为 None，则自动计算。
    
    返回:
        similarity: [B] 相似度，范围在 0~1 之间
    """
    # # 计算欧氏距离
    # distances = torch.norm(inputs - centers, p=2, dim=1)  # [B]
    
    # # 如果没有提供 max_distance，则自动计算
    # if max_distance is None:
    #     max_distance = distances.max().item()  # 使用当前 batch 的最大距离作为归一化因子
    
    # # 计算相似度
    # similarity = 1 - distances / max_distance
    # similarity = similarity.clamp(0, 1)  # 确保相似度在 0~1 范围内
    similarity = F.cosine_similarity(inputs, centers, dim=1)  # [B]
    similarity = (similarity + 1) / 2
    
    return similarity

class ClusterMemoryAMP(nn.Module, ABC):
    def __init__(self, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemoryAMP, self).__init__()
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.features = None

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)

        with torch.no_grad():
            # 获取类中心的特征
            centers = self.features[targets]  # 选择每个样本对应的类中心，形状为 [B, dim]
            # 计算余弦相似度
            similarity = compute_similarity(inputs, centers)  # 计算输入与类中心之间的余弦相似度

        return loss, similarity
    
    
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs) 
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1) 
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
    
class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0
    def forward(self, text_features, image_features, t_label, i_targets): 
        batch_size = text_features.shape[0] 
        batch_size_N = image_features.shape[0] 
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device) 

        logits = torch.div(torch.matmul(text_features, image_features.T),self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach() 
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 
        loss = - mean_log_prob_pos.mean()

        return loss
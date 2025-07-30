from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

class PartMaskLoss(nn.Module):
    def __init__(self, loss_type='cl', label_smoothing=0.1, device='cpu', background_label=None):
        super(PartMaskLoss, self).__init__()
        self.device = device
        self.loss_fun = KLabelSmoothedCrossEntropyLoss(epsilon=label_smoothing, ignore_index=background_label)  # 使用平滑交叉熵损失, 不需要背景得分
    
    def forward(self, pred_part_masks, target_mask):
        """
        input:
            pred_part_masks: [N, M, fH, fW]
            target_mask: [N, fH, fW]
        """
        loss = self.loss_fun(pred_part_masks, target_mask)
        return loss
    
class KLabelSmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=0.1, ignore_index=None, reduction='mean', class_weights=None):
        super(KLabelSmoothedCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.reduction = reduction
        if class_weights is not None:
            self.class_weights = class_weights.view(1, class_weights.shape[0], 1, 1)
        else:
            self.class_weights = None

    def forward(self, input, target):
        # 输入input的形状是[B, K+1, H, W], 实际的类别数K已经去除了背景类别
        # 目标target的形状是[B, H, W], 其值在0~K之间，0是背景类别
        B, C, H, W = input.size()
        
        # 进行softmax操作获得概率分布
        log_probs = F.log_softmax(input, dim=1)
        
        # 生成一个全是平滑标签值的张量
        smoothed_labels = torch.full((B, C, H, W), self.epsilon / (C - 1)).to(input.device)
        
        # 把目标标签值的位置设置为1 - epsilon
        target = target.unsqueeze(1).long()  # [B, 1, H, W] 增加1维以匹配input的维度
        smoothed_labels.scatter_(1, target, 1 - self.epsilon)
        
        # 应用ignore_index，生成mask忽略背景类别
        # mask = target != self.ignore_index  # 忽略背景值的掩码
        if self.ignore_index is not None:
            mask = target != self.ignore_index  # 忽略背景值的掩码
            smoothed_labels = smoothed_labels * mask.float()

        # 如果提供了类别权重，那么使用它来调整损失
        if self.class_weights is not None:
            class_weights = self.class_weights.to(input.device)
            smoothed_labels = smoothed_labels * class_weights
        
        # 计算标签平滑交叉熵损失
        loss = -log_probs * smoothed_labels
        
        # 根据reduction参数来决定输出的形式
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            scale = smoothed_labels.shape[0] * smoothed_labels.shape[2] * smoothed_labels.shape[3]
            loss = loss.sum() / scale
        elif self.reduction == 'none':
            loss = loss.sum(dim=1)  # 返回每个样本的平均损失

        return loss

if __name__ == '__main__':
    # 使用示例
    criteria = KLabelSmoothedCrossEntropyLoss(epsilon=0.1, ignore_index=0)
    logits = torch.randn(2, 3, 4, 5)  # 模拟输入
    target = torch.ones(2, 4, 5, dtype=torch.long)  # 模拟目标
    loss = criteria(logits, target)
    print(loss)


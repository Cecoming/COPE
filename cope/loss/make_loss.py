# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import torch

def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    # if cfg.MODEL.IF_LABELSMOOTH == 'on':
        # xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        # print("label smooth on, numclasses:", num_classes)

    # if sampler == 'softmax':
    #     def loss_func(score, feat, target):
    #         return F.cross_entropy(score, target)
    # else:
    def loss_func(score, feat, target, target_cam, visible=None):
            # if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
            #     if cfg.MODEL.IF_LABELSMOOTH == 'on':
                #     if isinstance(score, list):
                #         ID_LOSS = [xent(scor, target) for scor in score[0:]]
                #         ID_LOSS = sum(ID_LOSS)
                #     else:
                #         ID_LOSS = xent(score, target)

                #     if isinstance(feat, list):
                #         TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                #         TRI_LOSS = sum(TRI_LOSS) 
                #     else:   
                #         TRI_LOSS = triplet(feat, target)[0]
                    
                #     loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                #     if i2tscore != None:
                #         I2TLOSS = xent(i2tscore, target)
                #         loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                        
                #     return loss
                # else:
                    if visible is not None:
                        visible = visible.repeat(3)
                    if isinstance(score, list):
                        GID_LOSS = F.cross_entropy(score[0], target)
                    
                        PID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        PID_LOSS = sum(PID_LOSS)/len(score[1:])
                        ID_LOSS = GID_LOSS + PID_LOSS
                    else:
                        if visible is not None:
                            if visible.sum() == 0:
                                ID_LOSS = torch.tensor(0).cuda()
                            else:                            
                                ID_LOSS = F.cross_entropy(score[visible == 1], target[visible == 1])
                        else:
                            ID_LOSS = F.cross_entropy(score, target)
                    
                    # if isinstance(feat, list):
                    # #     # GTRI_LOSS = triplet(feat[0], target)[0]
                    # #     # PTRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                    # #     # PTRI_LOSS = sum(PTRI_LOSS) / len(feat[1:])
                    # #     # TRI_LOSS = GTRI_LOSS + PTRI_LOSS
                    #     PTRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                    #     TRI_LOSS = sum(PTRI_LOSS) / len(feat[1:])
                    # else:
                    #     if visible is not None:
                    #         if visible.sum() == 0:
                    #             TRI_LOSS = torch.tensor(0).cuda()
                    #         else:
                    #             feat_ = replace_invisible_features(feat, target, visible)
                    #             TRI_LOSS = triplet(feat_, target)[0]
                    #             TRI_LOSS = (TRI_LOSS * visible).sum() / visible.sum() + 1e-6
                    #     else:
                    #         TRI_LOSS = triplet(feat, target)[0]
                    #         TRI_LOSS = TRI_LOSS.sum() / TRI_LOSS.shape[0] + 1e-6

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS# + TRI_LOSS

                    return loss

            # else:
            #     print('expected METRIC_LOSS_TYPE should be triplet'
            #           'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    # else:
    #     print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
    #           'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


def replace_invisible_features(feat, target, visible):
    # 创建一个新的张量用于存储替换后的特征
    new_feat = feat
    
    # 遍历 visible 为 0 的位置
    for i in range(visible.size(0)):
        if visible[i] == 0:
            # 获取当前样本的 ID
            current_id = target[i]
            
            # 找到相同 ID 且 visible 为 1 的特征
            same_id_indices = (target == current_id) & (visible == 1)
            
            if same_id_indices.any():
                # 随机选择一个相同 ID 的特征
                replacement_feat = feat[same_id_indices][0]
            else:
                # 如果没有相同 ID 的 visible 特征，保留原特征
                replacement_feat = feat[i]
            
            # 替换特征
            new_feat[i] = replacement_feat
    
    return new_feat
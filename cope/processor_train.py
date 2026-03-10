import logging
import os
import torch
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics_PSS import R1_mAP_eval
from utils.CICO_PBF import CICO_PBF
from torch.cuda import amp
from .utils import *
from .loss_ import ClusterMemoryAMP
from .loss.make_loss import make_loss
import time

from utils.random_erasing import RandomErasing


def sample_patch_prob(part_score, kernel_size=16, stride=(16, 16), threshold=0.5):
    """
    Args:
        part_score: [B, H, W], foreground probability for each pixel
        kernel_size: size of the patch, default is 16
        stride: downsampling stride (height and width directions)
    Returns:
        Binarized patch average probability, shape [B, L],
        where L = (h_resolution * w_resolution), e.g., L = 128 for stride=(16,16), L = 210 for stride=(12,12)
    """
    B, H, W = part_score.shape
    part_score = part_score.unsqueeze(1)
    patches = F.unfold(part_score, kernel_size=kernel_size, stride=stride)
    patch_mean = patches.mean(dim=1)
    patch_binary = (patch_mean > threshold).float()
    return patch_binary

def attn_loss_fn(attn_list, fore_mask):
    """
    part_score: Predicted scores, expected to be in [0,1], shape can be [B, H, W] or [B, 1, H, W], etc.
    parT_score_target: Human parsing labels, 0 indicates background, greater than 0 indicates foreground
    """
    loss = 0
    for attn in attn_list:
        # 取背景区域的注意力值
        loss += (attn * (1 - fore_mask)).mean()
    return loss

def segmentation_loss(part_score, part_score_target, radius=10):
    """
    part_score: 预测的得分，要求在 [0,1]，形状可以为 [B, H, W] 或 [B, 1, H, W] 等
    part_score_target: 人体解析标签，0 表示背景，大于 0 表示前景
    """
    import torch.nn.functional as F

    part_score_target = part_score_target.argmax(dim=1)  # [B, N]
    binary_target = (part_score_target > 0).float()
    loss = F.binary_cross_entropy_with_logits(part_score, binary_target) * radius
    return loss

def train(cfg,
              model,
              train_loader,
              val_loader,
              cluster_loader,
              optimizer,
              scheduler,
              num_query,
              num_classes,
              pre_optimizer,
              pre_scheduler):
    
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    pre_checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs_pre = cfg.SOLVER.STAGE1.MAX_EPOCHS
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("COPE")
    logger.info('start training')
    
    model.to(device)
    
    loss_meter = AverageMeter()
    loss_1_meter = AverageMeter()
    acc_meter = AverageMeter()
    loss_mem_meter = AverageMeter() 
    loss_reid_meter = AverageMeter() 
    loss_2_meter = AverageMeter()  
    loss_3_meter = AverageMeter()  
    loss_4_meter = AverageMeter()  
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    logger.info(f'smoothed cross entropy loss on {num_classes} classes.')

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, K1=cfg.SOLVER.K1, K2=cfg.SOLVER.K2)
    scaler = amp.GradScaler()
    data_aug = CICO_PBF(M=cfg.SOLVER.OCCLUSION_NUM, image_shape = (cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1], 3), device=device).to(device)

    ################################# 1. Pre-Train the model
    if cfg.SOLVER.STAGE1.PRETRAINED_PATH == '':
        logger.info("Pre-training stage is begin.")
        for epoch in range(1, epochs_pre+1):
            loss_meter.reset()
            pre_scheduler.step(epoch)
            # train one iteration
            model.train()
            model.text_encoder.eval()

            for n_iter, (img, vid, target_cam, target_view, mask) in enumerate(train_loader):
                pre_optimizer.zero_grad()
                img = img.to(device)
                mask = mask.to(device)
                target = vid.to(device)
                target_cam = target_cam.to(device)
                
                if cfg.MODEL.SIE_CAMERA:
                    target_cam = target_cam.to(device)
                else: 
                    target_cam = None
                if cfg.MODEL.SIE_VIEW:
                    target_view = target_view.to(device)
                else: 
                    target_view = None
                
                with amp.autocast(enabled=True):
                    feat_src, logit_src, matrix, _, _ = model(img, cam_label=target_cam, view_label=target_view, get_matrix=True)
                    loss_seg = segmentation_loss(matrix, mask)
                    loss = loss_seg

                scaler.scale(loss).backward()
                scaler.step(pre_optimizer)
                scaler.update()

                loss_meter.update(loss.item(), img.shape[0])

                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, pre_scheduler._get_lr(epoch)[0]))

            logger.info("Epoch {} done.".format(epoch))
        if epoch % pre_checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'pre_{}.pth'.format(epoch)))
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'pre_{}.pth'.format(epoch)))
        logger.info("Pre-training stage is done.")
    else:
        model.load_param(cfg.SOLVER.STAGE1.PRETRAINED_PATH)
        logger.info("Pre-trained model loaded from {}".format(cfg.SOLVER.STAGE1.PRETRAINED_PATH))


    scaler = amp.GradScaler()
    ################################### 2. Train the model
    logger.info("Training stage is begin.")
    for epoch in range(1, epochs+1):
        loss_meter.reset()
        loss_1_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        loss_mem_meter.reset()  
        loss_reid_meter.reset()  
        loss_2_meter.reset()  
        loss_3_meter.reset()  
        loss_4_meter.reset()  

        # create memory bank
        image_features, gt_labels = extract_image_features(model, cluster_loader, use_amp=True)
        image_features = image_features.float()
        image_features = F.normalize(image_features, dim=1)
            
        num_classes = len(gt_labels.unique()) - 1 if -1 in gt_labels else len(gt_labels.unique())
        logger.info(f'Memory has {num_classes} classes.')
    
        # CAP memory
        memory = ClusterMemoryAMP(momentum=cfg.MODEL.MEMORY_MOMENTUM, use_hard=True).to(device)
        memory.features = compute_cluster_centroids(image_features, gt_labels).to(device)

        # train one iteration
        model.train()
        model.text_encoder.eval()

        RE = RandomErasing(probability=1.0, mode='pixel', max_count=1, device='cpu')

        for n_iter, (img, vid, target_cam, target_view, mask) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            mask = mask.to(device)

            target = vid.to(device)
            target_cam = target_cam.to(device)
            
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            
            with amp.autocast(enabled=True):
                ######################### source images
                feat_src, logit_src, matrix, feat_patch11_src, p_score_src = model(img, cam_label=target_cam, view_label=target_view, get_matrix=True)
                part_score = torch.sigmoid(matrix)  # [B, N]

                ######################### CICO + PBF
                # data augmentation
                img_cico, img_pbf, mask_cico = data_aug(img, pred_mask=part_score.detach())

                # extract feature
                feat_cico, logit_cico, feat_patch11_cico, p_score_cico = model(img_cico, cam_label=target_cam, view_label=target_view, get_matrix=False)
                feat_pbf, logit_pbf, feat_patch11_pbf, _ = model(img_pbf, cam_label=target_cam, view_label=target_view,  get_matrix=False)

                ######################### loss_memory
                loss_mem, sim_cico = memory(feat_cico, target)
                loss_mem_src, sim_src = memory(feat_src, target)
                loss_mem = (loss_mem + loss_mem_src)

                ######################### loss_ce
                loss_ce = loss_func(torch.cat([logit_cico, logit_src, logit_pbf], dim=0), torch.cat([feat_cico, feat_src, feat_pbf], dim=0), torch.cat([target, target, target], dim=0), None) * 0.5

                ######################### loss_occ
                mask_cico = sample_patch_prob(mask_cico, kernel_size=16, stride=cfg.MODEL.STRIDE_SIZE).float().unsqueeze(-1)
                mask_occ_sum = mask_cico.sum(dim=1, keepdim=True) + 1e-6
                occ_feature = (feat_patch11_cico * mask_cico).sum(dim=1) / mask_occ_sum.squeeze(-1)   # [B, D]
                groups = {i: [] for i in range(4)}
                for idx in range(occ_feature.size(0)):
                    groups[idx % 4].append(occ_feature[idx])

                group_tensors = {key: torch.stack(value) for key, value in groups.items()}
                loss_occ_align = 0.0
                for key, group in group_tensors.items():
                    if group.size(0) > 1: 
                        diff = group.unsqueeze(1) - group.unsqueeze(0)  # [n, n, D]
                        mse = (diff ** 2).mean(dim=-1) 
                        loss_occ_align += mse.sum() / 2 
                loss_occ_align /= len(group_tensors) 
                loss_occ_align *= cfg.MODEL.ID_LOSS_WEIGHT

                ######################### loss_align
                mask_fg = (part_score >= 0.5).float().unsqueeze(-1)   # [B, N, 1]
                mask_fg = sample_patch_prob(part_score, kernel_size=16, stride=cfg.MODEL.STRIDE_SIZE).float().unsqueeze(-1)
                mask_sum = mask_fg.sum(dim=1, keepdim=True) + 1e-6
                fg_feature_src = (feat_patch11_src * mask_fg).sum(dim=1) / mask_sum.squeeze(-1)   # [B, D]
                fg_feature_pbf = (feat_patch11_pbf * mask_fg).sum(dim=1) / mask_sum.squeeze(-1)   # [B, D]
                loss_fg_align = F.mse_loss(fg_feature_src, fg_feature_pbf)

                ######################### loss_sim
                loss_sim = (F.mse_loss(p_score_cico, sim_cico.detach()) + F.mse_loss(p_score_src, sim_src.detach()))

                ######################### loss_segmentation
                loss_seg = segmentation_loss(matrix, mask)
                loss = loss_mem + loss_ce + loss_seg + loss_sim + loss_fg_align + loss_occ_align


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f} , Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, scheduler.get_lr()[0]))
        

        scheduler.step()
        logger.info("Epoch {} done.".format(epoch))
        
        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            
        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    if cfg.MODEL.SIE_CAMERA:
                        camids = camid.to(device)
                    else: 
                        camids = None
                    if cfg.MODEL.SIE_VIEW:
                        target_view = target_view.to(device)
                    else: 
                        target_view = None
                    feat, p_score = model(img, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, vid, camid, p_score))

            distmat, pids, camids, qf, gf, cmc, mAP = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            
            torch.cuda.empty_cache()
    logger.info('Training done.')
    print(cfg.OUTPUT_DIR)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    
    device = "cuda"
    logger = logging.getLogger("COPE")
    logger.info("Enter inferencing")
    model.to(device)

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING, K1=cfg.SOLVER.K1, K2=cfg.SOLVER.K2)
    evaluator.reset()

    start_time = time.time()  # Start timing
    model.eval()
    for n_iter, (img, pid, camid, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camid.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat, p_score = model(img, cam_label=camids, view_label=target_view)

            evaluator.update((feat, pid, camid, p_score))

    distmat, pids, camids, qf, gf, cmc, mAP = evaluator.compute()

    logger.info("Validation P_socre Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    end_time = time.time()  # End timing
    total_time = end_time - start_time  # Calculate total runtime
    logger.info("Inference completed in {:.2f} seconds".format(total_time))  # Output runtime

    return cmc[0], cmc[4]


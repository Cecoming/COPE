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


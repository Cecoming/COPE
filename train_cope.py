from utils.logger import setup_logger
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
from solver.lr_scheduler import WarmupMultiStepLR
from cope.dataloader import make_dataloader
from cope.processor_train import train
from cope.optimizer import make_optimizer
from cope.model import make_model
from solver.make_optimizer import make_optimizer_1stage
from solver.scheduler_factory import create_scheduler
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="config/pcl-vit.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger = setup_logger("COPE", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    
    train_loader, val_loader, cluster_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    if cfg.MODEL.NAME == 'ViT-B-16':
        model = make_model(cfg, num_classes, camera_num=camera_num, view_num=view_num)
    else:
        raise ValueError("Model name not recognized: {}".format(cfg.MODEL.NAME))
    optimizer = make_optimizer(cfg, model)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, max_epoch=cfg.SOLVER.MAX_EPOCHS)
    
    optimizer_1stage = make_optimizer_1stage(cfg, model)
    scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS, lr_min = cfg.SOLVER.STAGE1.LR_MIN, \
                        warmup_lr_init = cfg.SOLVER.STAGE1.WARMUP_LR_INIT, warmup_t = cfg.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range = None)
    
    train(
        cfg,
        model,
        train_loader,
        val_loader,
        cluster_loader,
        optimizer,
        scheduler,
        num_query,
        num_classes,
        optimizer_1stage,
        scheduler_1stage,
    )

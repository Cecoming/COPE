import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from datasets.market1501 import Market1501
from datasets.msmt17 import MSMT17
from .dataset import ImageDataset, IterLoader, ImageDatasetMask
from .sampler import RandomIdentitySampler
from datasets.occ_duke import OCC_DukeMTMCreID
from datasets.occ_reid import Occluded_REID
from datasets.p_duke import P_DukeMTMC_REID
from datasets.p_reid import Partial_REID
import random
import numpy as np
from PIL import Image
from torchvision.transforms import RandomCrop  # 用于获取裁剪参数
from .mask_transform import AddBackgroundMask, PART_MAP
from .pifpaf_mask_transform import CombinePifPafIntoFourVerticalParts
from torch import nn

FACTORY = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'occ_reid': Occluded_REID,
    'p_duke_reid': P_DukeMTMC_REID,
    'p_reid': Partial_REID
}

def train_collate_fn(batch):
    imgs, pids, camids, viewids, masks = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, torch.stack(masks, dim=0)

def make_dataloader(cfg):
    """
    PCL dataloader. It returns 3 dataloaders: training loader, cluster loader and validation loader.
    - For training loader, PK sampling is applied to select K instances from P classes.
    - For cluster loader, a plain loader is returned with validation augmentation but on
      training samples.
    - For validation loader, a validation loader is returned on test samples.
    
    Args:
    - dataset: dataset object.
    - all_iters: if `all_iters=True`, number training iteration is decided by `num_samples//batchsize`
    """
    
    dataset = FACTORY[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    
    if cfg.DATASETS.NAMES == 'occ_reid' or cfg.DATASETS.NAMES == 'p_reid':
        train_transforms = PairCompose([
            ToTensorPair(),
            ResizePair(cfg.INPUT.SIZE_TRAIN, interpolation_img=3, interpolation_mask=Image.NEAREST),
            ColorJitterPair(brightness=0.3, contrast=0.3, saturation=0.2),
            MasktoSix(),
            RandomHorizontalFlipPair(p=cfg.INPUT.PROB),
            PadPair(cfg.INPUT.PADDING, fill=0),
            RandomCropPair(cfg.INPUT.SIZE_TRAIN),
            NormalizePair(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])
    else:
        train_transforms = PairCompose([
            ToTensorPair(),
            ResizePair(cfg.INPUT.SIZE_TRAIN, interpolation_img=3, interpolation_mask=Image.NEAREST),
            MasktoSix(),
            RandomHorizontalFlipPair(p=cfg.INPUT.PROB),
            PadPair(cfg.INPUT.PADDING, fill=0),
            RandomCropPair(cfg.INPUT.SIZE_TRAIN),
            NormalizePair(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])
    train_set = ImageDatasetMask(dataset.train, train_transforms)

    sampler = RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=sampler,
        num_workers=num_workers, collate_fn=train_collate_fn
    )
        
    # val loader
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    val_set = ImageDataset(dataset.query+dataset.gallery, val_transforms)
    num_queries = len(dataset.query)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers
    )
    
    # cluster loader
    cluster_set = ImageDataset(dataset.train, val_transforms)
    cluster_loader = DataLoader(
        cluster_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, cluster_loader, num_queries, num_classes, cam_num, view_num

class ColorJitterPair(object):
    """
    对 img 应用 ColorJitter，同时 mask 保持不变。
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def __call__(self, img, mask):
        img = self.color_jitter(img)
        return img, mask

# ----------------- 成对变换类 -----------------
class MasktoSix(object):
    def __init__(self):
        super().__init__()
        self.combine_transform = CombinePifPafIntoFourVerticalParts()
        self.AddBackgroundMask = AddBackgroundMask(background_computation_strategy="threshold",
                                                   softmax_weight=15,
                                                   mask_filtering_threshold=0.4)
    def __call__(self, img, mask):
        mask = self.combine_transform.apply_to_mask(mask)
        mask = self.AddBackgroundMask.apply_to_mask(mask)
        return img, mask
    
class PairCompose(object):
    """成对 Compose：依次对 (img, mask) 应用多个变换"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class ResizePair(object):
    """
    成对 Resize：
    - 对 img 使用 PIL 内置的 resize；
    - 对 mask（numpy 数组，shape: [H, W, N]）逐通道 resize（采用最近邻插值）
    """
    def __init__(self, size, interpolation_img=Image.BILINEAR, interpolation_mask=Image.NEAREST, mask_scale=4):
        self.size = size  # (height, width)
        self.interpolation_img = interpolation_img
        self.interpolation_mask = interpolation_mask
        self._size = (int(size[0]/mask_scale), int(size[1]/mask_scale))


    def __call__(self, img, mask):
        img = F.resize(img, self.size, interpolation=self.interpolation_img)
        # mask = nn.functional.interpolate(mask.unsqueeze(0), self.size, mode='nearest').squeeze(0)  # Best perf with nearest here and bilinear in parts engine
        mask = nn.functional.interpolate(mask.unsqueeze(0), self.size, mode='bilinear').squeeze(0)
        return img, mask

class RandomHorizontalFlipPair(object):
    """成对随机水平翻转"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = F.hflip(img)
            if isinstance(mask, np.ndarray):
                mask = np.fliplr(mask)  # numpy 水平翻转
            else:
                mask = mask.flip(-1)  # tensor 水平翻转
        return img, mask

class PadPair(object):
    """成对 Pad 操作
       对 img 使用 torchvision.transforms.functional.pad；
       对 mask 使用 np.pad，（对高度和宽度进行相同填充，对通道不填充）
    """
    def __init__(self, padding, fill=0):
        self.padding = padding  # padding 可以是 int
        self.fill = fill

    def __call__(self, img, mask):
        # 对 img 进行常规 pad 操作
        img = F.pad(img, self.padding, fill=self.fill)
        pad = self.padding if isinstance(self.padding, int) else None
        if pad is None:
            raise NotImplementedError("非 int 类型的 padding 暂未实现")
        
        if isinstance(mask, np.ndarray):
            bg = mask[0:1, :, :]
            parts = mask[1:, :, :]
            bg = np.pad(bg, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=1)
            parts = np.pad(parts, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
            mask = np.concatenate([bg, parts], axis=0)
        else:
            bg = mask[0:1, :, :]
            parts = mask[1:, :, :]
            bg = F.pad(bg, (pad, pad, pad, pad), fill=1)
            parts = F.pad(parts, (pad, pad, pad, pad), fill=0)
            mask = torch.cat([bg, parts], dim=0)
        return img, mask

class RandomCropPair(object):
    """成对随机裁剪"""
    def __init__(self, size):
        self.size = size  # (height, width)

    def __call__(self, img, mask):
        i, j, h, w = RandomCrop.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        if isinstance(mask, np.ndarray):
            mask = mask[i:i+h, j:j+w, :]
        else:
            mask = F.crop(mask, i, j, h, w)
        # mask = mask[i:i+h, j:j+w, :]
        return img, mask

class ToTensorPair(object):
    """将 img 转换为 tensor，同时将 mask（多通道 np.array，shape: [H, W, N]）
       转换为 tensor 并调整为 shape: [N, H, W]
    """
    def __call__(self, img, mask):
        img = F.to_tensor(img)
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask).float()  # shape: [H, W, N]
        mask_tensor = mask_tensor.permute(2, 0, 1)  # shape: [N, H, W]
        return img, mask_tensor

class NormalizePair(object):
    """对 img 归一化，mask 保持不变"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img, mask
# Cross-Identity Consistency Occlusion and Prompt Similarity Scoring (CICO_PSS)
import torch
import torch.nn as nn
import random
import math

class CICO_PBF(nn.Module):
    def __init__(self, M, image_shape=(256, 128, 3), min_area=1/5, max_area=1/2,
                     min_aspect=0.3, max_aspect=None, device='cuda', occlusion_prob=1.0):
        """
        M: Number of occlusions
        image_shape: (height, width, channels)
        min_area, max_area: Proportion range of occlusion area to the whole image area
        min_aspect, max_aspect: Aspect ratio range of occlusion (if max_aspect is None, take 1/min_aspect)
        device: Specify storage device
        """
        super(CICO_PBF, self).__init__()
        self.M = M
        self.image_h, self.image_w, self.image_c = image_shape
        self.min_area = min_area
        self.max_area = max_area
        if max_aspect is None:
            max_aspect = 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.device = torch.device(device)
        self.occlusion_info = []  # Position of each occlusion: (top, left, h, w)
        self.patch_list = nn.ParameterList()
        self.occlusion_prob = occlusion_prob
        
        area = self.image_h * self.image_w
        for _ in range(M):
            while 1:
                target_area = random.uniform(self.min_area, self.max_area) * area
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < self.image_w and h < self.image_h and w > 0 and h > 0:
                    top = random.randint(0, self.image_h - h)
                    left = random.randint(0, self.image_w - w)
                    self.occlusion_info.append((top, left, h, w))
                    patch = nn.Parameter(torch.randn(self.image_c, h, w, device=self.device))
                    self.patch_list.append(patch)
                    break
        self.frozen()
    
    
    def forward(self, x, pred_mask=None):
        """
        x: [B, C, H, W], requires H=256, W=128, C=3, and x is already on self.device
        Apply the same occlusion with probability self.occlusion_prob to images with the same index modulo 4,
        otherwise keep the original image unchanged.
        Returns the occluded image, shape: [B, C, H, W]
        """
        ############## CICO
        B, C, H, W = x.shape
        out = x.clone()
        o_mask = torch.zeros(B, H, W, device=x.device)
        # Grouping: group by the remainder of the image index divided by 4
        for r in range(2):
            idxs = [i for i in range(B) if i % 4 == r]
            if not idxs:
                continue
            if random.random() <= self.occlusion_prob:
                patch_idx = random.randint(0, self.M - 1)
                top, left, h, w = self.occlusion_info[patch_idx]
                for i in idxs:
                    out[i, :, top:top+h, left:left+w] = self.patch_list[patch_idx]
                    o_mask[i, top:top+h, left:left+w] = 1

        ############## PBF
        occ_only_img, visible = self.PBF(x, pred_mask)
        return out, occ_only_img, o_mask

    
    def PBF(self, image, occ_mask=None, threshold=0.5):
        """
        Parameters:
        image: [B, C, H, W] Original image
        occ_mask: [B, 5, H, W] Occlusion segmentation result, 0 for background, 1~4 for foreground
        Returns:
        occluded_image: Foreground retains the original image, background is filled with random colors.
        The range of random colors is determined based on the image, with each sample randomly determining a color
        """
        # pred_mask
        occluded_image = image.clone()
        bg_mask = (occ_mask < threshold)
        B, H, W = bg_mask.shape
        total_pixels = H * W
        # Calculate the background pixel ratio for each sample
        bg_ratios = bg_mask.view(B, -1).sum(dim=1).float() / total_pixels

        B, C, H, W = image.shape
        rand_colors = []
        for i in range(B):
            channel_colors = []
            for c in range(C):
                min_val = image[i, c].min().item()
                max_val = image[i, c].max().item()
                color_val = torch.empty(1, dtype=image.dtype, device=image.device).uniform_(min_val, max_val)
                channel_colors.append(color_val)
            # Form [C, 1, 1] random color
            rand_color = torch.stack(channel_colors).view(C, 1, 1)
            rand_colors.append(rand_color)
        # rand_colors: [B, C, 1, 1]
        rand_colors = torch.stack(rand_colors, dim=0)

        # For each sample, fill the background area with the random color of that sample
        visible_list = []
        for i in range(B):
            # if (random.random() <= self.occlusion_prob) or (bg_ratios[i] > 0.95):    # 一半的进行颜色填充
            if (bg_ratios[i] > 0.95):
                visible_list.append(0)
                continue                   
            visible_list.append(1)
            num_bg = bg_mask[i].sum()
            if num_bg > 0:
                occluded_image[i, :, bg_mask[i]] = rand_colors[i].view(C, 1).expand(C, int(num_bg))
        return occluded_image, torch.tensor(visible_list, device=image.device)

    def frozen(self):
        for patch in self.patch_list:
            patch.requires_grad = False
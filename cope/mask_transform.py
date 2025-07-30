import torch
from torch import nn
# from albumentations import DualTransform
import torch.nn.functional as F

# self.parts_map
PART_MAP = {'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4, 'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8, 'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12, 'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16, 'left_ankle_to_left_knee': 17, 'left_knee_to_left_hip': 18, 'right_ankle_to_right_knee': 19, 'right_knee_to_right_hip': 20, 'left_hip_to_right_hip': 21, 'left_shoulder_to_left_hip': 22, 'right_shoulder_to_right_hip': 23, 'left_shoulder_to_right_shoulder': 24, 'left_shoulder_to_left_elbow': 25, 'right_shoulder_to_right_elbow': 26, 'left_elbow_to_left_wrist': 27, 'right_elbow_to_right_wrist': 28, 'left_eye_to_right_eye': 29, 'nose_to_left_eye': 30, 'nose_to_right_eye': 31, 'left_eye_to_left_ear': 32, 'right_eye_to_right_ear': 33, 'left_ear_to_left_shoulder': 34, 'right_ear_to_right_shoulder': 35}

class MaskTransform():
    def __init__(self):
        super(MaskTransform, self).__init__()

    def apply(self, img, **params):
        return img

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError("Method apply_to_keypoint is not implemented in class " + self.__class__.__name__)


class MaskGroupingTransform(MaskTransform):

    def __init__(self, parts_grouping, combine_mode='max'):
        super().__init__()
        self.parts_grouping = parts_grouping
        self.parts_map = PART_MAP
        self.parts_names = list(parts_grouping.keys())
        self.parts_num = len(self.parts_names)
        self.combine_mode = combine_mode    # 'max'

    def apply_to_mask(self, masks, **params):
        parts_masks = []
        for i, part in enumerate(self.parts_names):
            if self.combine_mode == 'sum':
                parts_masks.append(masks[[self.parts_map[k] for k in self.parts_grouping[part]]].sum(dim=0).clamp(0, 1))
            else:
                parts_masks.append(masks[[self.parts_map[k] for k in self.parts_grouping[part]]].max(dim=0)[0].clamp(0, 1))
        return torch.stack(parts_masks)


class PermuteMasksDim(MaskTransform):
    def apply_to_mask(self, masks, **params):
        return masks.permute(2, 0, 1)


class ResizeMasks(MaskTransform):
    def __init__(self, height, width, mask_scale):
        super(ResizeMasks, self).__init__()
        self._size = (int(height/mask_scale), int(width/mask_scale))

    def apply_to_mask(self, masks, **params):
        return nn.functional.interpolate(masks.unsqueeze(0), self._size, mode='nearest').squeeze(0)  # Best perf with nearest here and bilinear in parts engine


class RemoveBackgroundMask(MaskTransform):
    def apply_to_mask(self, masks, **params):
        return masks[:, :, 1::]


class AddBackgroundMask(MaskTransform):
    def __init__(self, background_computation_strategy='sum', softmax_weight=15, mask_filtering_threshold=0.5):
        super().__init__()
        self.background_computation_strategy = background_computation_strategy
        self.softmax_weight = softmax_weight
        self.mask_filtering_threshold = mask_filtering_threshold

    def apply_to_mask(self, masks, **params):
        if self.background_computation_strategy == 'sum':
            background_mask = 1 - masks.sum(dim=0)
            background_mask = background_mask.clamp(0, 1)
            masks = torch.cat([background_mask.unsqueeze(0), masks])
        elif self.background_computation_strategy == 'threshold':
            background_mask = masks.max(dim=0)[0] < self.mask_filtering_threshold
            masks = torch.cat([background_mask.unsqueeze(0), masks])
        elif self.background_computation_strategy == 'diff_from_max':
            background_mask = 1 - masks.max(dim=0)[0]
            background_mask = background_mask.clamp(0, 1)
            masks = torch.cat([background_mask.unsqueeze(0), masks])
        else:
            raise ValueError('Background mask combine strategy {} not supported'.format(self.background_computation_strategy))
        if self.softmax_weight > 0: # 15
            masks = F.softmax(masks * self.softmax_weight, dim=0)
        else:
            masks = masks / masks.sum(dim=0)
        return masks


class IdentityMask(MaskTransform):
    parts_names = ['id']
    parts_num = 1
    def apply_to_mask(self, masks, **params):
        return torch.ones((1, masks.shape[1], masks.shape[2]))

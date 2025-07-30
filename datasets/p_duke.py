# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import glob
import re
import urllib
import zipfile

import os.path as osp

from utils.iotools import mkdir_if_missing
from .bases import BaseImageDataset


class P_DukeMTMC_REID(BaseImageDataset):
    """
    This is a person dataset with occlusion named P-DukeMTMC-REID used for occluded person re-identification, selected from [1].

    The dataset is divided into training set and test set.
    There are two folders in the dataset in each data partition: <occluded_body_images> and <whole_body_images>.

    The train set includes 12927 images from 665 ID, with 2647 images with occlusion and 10280 images without occlusion.
    The test set includes 11216 images from 634 ID, with 2163 images with occlusion and 9053 images without occlusion.
    """
    dataset_dir = 'P-DukeMTMC-reid'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(P_DukeMTMC_REID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'test/occluded_body_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'test/whole_body_images')
        self.mask_train_dir = osp.join(self.dataset_dir, 'masks/pifpaf_maskrcnn_filtering/train')
        self.mask_mask_query_dir = osp.join(self.dataset_dir, 'masks/pifpaf_maskrcnn_filtering/test/occluded_body_images')
        self.mask_mask_gallery_dir = osp.join(self.dataset_dir, 'masks/pifpaf_maskrcnn_filtering/whole_body_images')
        self.pid_begin = pid_begin
        self._check_before_run()

        train = self._process_train_dir(self.train_dir, relabel=True, mask_path=self.mask_train_dir)
        query = self._process_dir(self.query_dir, relabel=False, mask_path=self.mask_mask_query_dir, is_query=True)
        gallery = self._process_dir(self.gallery_dir, relabel=False, mask_path=self.mask_mask_gallery_dir, is_query=False)

        if verbose:
            print("=> DukeMTMC-reID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_train_dir(self, dir_path, relabel=True, mask_path=None):
        img_paths = glob.glob(osp.join(dir_path,'whole_body_images', '*', '*.jpg'))
        pattern = re.compile(r'([-\d]+)_(\d)')
        camid = 1
        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]

            if mask_path is not None:
                mask_fname = osp.splitext(img_path)[0] + ".npy"
                # mask_fname = img_path + ".confidence_fields.npy"
                mask_fname = mask_fname.split('/')[-1]
                mask_file = osp.join(mask_path + "/whole_body_images", mask_fname)
                dataset.append((img_path, self.pid_begin + pid, camid, 1, mask_file))
            else:
                dataset.append((img_path, self.pid_begin + pid, camid, 1))

        img_paths = glob.glob(osp.join(dir_path,'occluded_body_images', '*', '*.jpg'))
        pattern = re.compile(r'([-\d]+)_(\d)')
        camid = 0
        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]

            if mask_path is not None:
                mask_fname = osp.splitext(img_path)[0] + ".npy"
                # mask_fname = img_path + ".confidence_fields.npy"
                mask_fname = mask_fname.split('/')[-1]
                mask_file = osp.join(mask_path + "/occluded_body_images", mask_fname)
                dataset.append((img_path, self.pid_begin + pid, camid, 1, mask_file))
            else:
                dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

    def _process_dir(self, dir_path, relabel=False, mask_path=None, is_query=False):
        img_paths = glob.glob(osp.join(dir_path, '*', '*.jpg'))
        pattern = re.compile(r'([-\d]+)_(\d)')
        if is_query:
            camid = 0
        else:
            camid = 1
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if relabel: pid = pid2label[pid]
            if mask_path is not None:
                mask_fname = img_path + ".confidence_fields.npy"
                mask_fname = mask_fname.split('/')[-1]
                mask_file = osp.join(mask_path, mask_fname)
                dataset.append((img_path, self.pid_begin + pid, camid, 1, mask_file))
            else:
                dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

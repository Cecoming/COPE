# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class Partial_REID(BaseImageDataset):

    dataset_dir = 'market1501'
    val_dir = 'Partial_REID'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(Partial_REID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.val_dir = osp.join(root, self.val_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.val_dir, 'occluded_body_images')
        self.gallery_dir = osp.join(self.val_dir, 'whole_body_images')
        self.mask_train_dir = osp.join(self.dataset_dir, 'masks/pifpaf_maskrcnn_filtering/bounding_box_train')
        self.mask_mask_query_dir = osp.join(self.val_dir, 'masks/pifpaf_maskrcnn_filtering/occluded_body_images')
        self.mask_mask_gallery_dir = osp.join(self.val_dir, 'masks/pifpaf_maskrcnn_filtering/whole_body_images')
        self.pid_begin = pid_begin

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True, mask_path=self.mask_train_dir)
        query = self._process_valdir(self.query_dir, camera_id=1, relabel=False, mask_path=self.mask_mask_query_dir)
        gallery = self._process_valdir(self.gallery_dir, camera_id=2, relabel=False, mask_path=self.mask_mask_gallery_dir)

        if verbose:
            print("=> Market1501 loaded")
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

    def _process_dir(self, dir_path, relabel=False, mask_path=None):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            if mask_path is not None:
                mask_fname = osp.splitext(img_path)[0] + ".npy"
                # mask_fname = img_path + ".confidence_fields.npy"
                mask_fname = mask_fname.split('/')[-1]
                mask_file = osp.join(mask_path, mask_fname)
                dataset.append((img_path, self.pid_begin + pid, camid, 1, mask_file))
            else:
                dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset
    
    def _process_valdir(self, dir_path, camera_id=1, relabel=False, mask_path=None):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pid_container = set()
        for img_path in img_paths:
            jpg_name = img_path.split('/')[-1]
            pid = int(jpg_name.split('_')[0])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            jpg_name = img_path.split('/')[-1]
            pid = int(jpg_name.split('_')[0])
            camid = camera_id
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            if mask_path is not None:
                mask_fname = img_path + ".confidence_fields.npy"
                mask_fname = mask_fname.split('/')[-1]
                mask_file = osp.join(mask_path, mask_fname)
                data.append((img_path, self.pid_begin + pid, camid, 1, mask_file))
            else:
                data.append((img_path, self.pid_begin + pid, camid, 1))
        return data



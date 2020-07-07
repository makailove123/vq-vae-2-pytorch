# coding=utf-8

import sys
import os
from io import BytesIO
import lmdb
import random
import time
import math
from typing import Optional
import torch
from torch.utils.data.dataset import IterableDataset
import distributed as dist
from PIL import Image


class ImageLmdbDataset(IterableDataset):
    def __init__(
            self,
            img_lmdb_path,
            img_keys_path,
            transform,
            batch_size,
            dist_mode: bool = False,
            rank_seed: Optional[int] = None,
            with_key: bool = False
    ):
        self.img_lmdb_path = img_lmdb_path
        self.img_keys_path = img_keys_path
        self.transform = transform
        self.batch_size = batch_size
        if rank_seed is not None:
            self.rand = random.Random(rank_seed)
        else:
            self.rand = random.Random(time.time())

        self.img_keys_file_list = [
            os.path.join(self.img_keys_path, f) for f in
            os.listdir(self.img_keys_path) if not f.startswith('.')
        ]

        self.rand.shuffle(self.img_keys_file_list)
        self.rank = -1
        if dist_mode:
            rank_pic_size = int(math.ceil(len(self.img_keys_file_list) / dist.get_world_size()))
            self.img_keys_file_list = self.img_keys_file_list[rank_pic_size * dist.get_rank():
                                                              rank_pic_size * (dist.get_rank() + 1)]
            self.rank = dist.get_rank()
        self.num_examples = max((sum((1 for _ in open(f))) for f in self.img_keys_file_list[:10])) \
                            * len(self.img_keys_file_list)
        self.num_itertions = int(math.ceil(self.num_examples / batch_size))
        self.with_key = with_key

    def __len__(self):
        return self.num_examples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0
        lmdb_env = lmdb.open(self.img_lmdb_path)
        lmdb_txn = lmdb_env.begin()
        pic_size = int(math.ceil(len(self.img_keys_file_list) / num_workers))
        file_list = self.img_keys_file_list[pic_size * worker_id: pic_size * worker_id + pic_size]
        self.rand.shuffle(file_list)
        for f in file_list:
            for line in open(f):
                img_key = line.strip()
                key_enc = img_key.encode("utf-8")
                img_buf = lmdb_txn.get(key_enc)
                try:
                    img = Image.open(BytesIO(img_buf)).convert("RGB")
                except:
                    continue
                if self.with_key:
                    yield img_key, self.transform(img)
                else:
                    yield self.transform(img)
        lmdb_env.close()



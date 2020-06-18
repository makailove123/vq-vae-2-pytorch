import argparse
import sys
import os
from typing import Optional, List
import random
import time
import math
import logging
import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import hashlib
import urllib

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torchsummary import summary

from torchvision import datasets, transforms, utils
from torchvision.datasets.folder import default_loader

from tqdm import tqdm

from scheduler import CycleScheduler
import distributed as dist

from vqvae import VQVAE
from vqvae_config import VqvaeConfig

logger = logging.getLogger(__name__)

PREFIX_CHECKPOINT_DIR = "checkpoint"
MODEL_SAVE_NAME = "pytorch_model.bin"
OPTIMIZER_SAVE_NAME = "optimizer.pt"
SCHEDULER_SAVE_NAME = "scheduler.pt"
CONFIG_SAVE_NAME = "config.json"

try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex


def parse_checkpoint_step(ckpt_path: str) -> Optional[int]:
    try:
        step = int(ckpt_path.split("-")[1])
    except (ValueError, TypeError):
        return None
    return step


def find_recent_checkpoint(model_path: str):
    if not os.path.isdir(model_path):
        return None
    ret = None
    max_step = 0
    for f in os.listdir(model_path):
        if not f.startswith("checkpoint-"):
            continue
        step = parse_checkpoint_step(f)
        if step is None:
            continue
        if step > max_step:
            max_step = step
            ret = os.path.join(model_path, f)
    return max_step, ret


class Trainer:
    def __init__(
            self, args, dataloader, model_config, model, optimizer, scheduler, device, trained_steps
    ):
        self.args = args
        self.dataloader = dataloader
        self.model_config = model_config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.global_step = trained_steps

    def sorted_checkpoints(self, output_dir, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def rotate_checkpoints(self, output_dir, use_mtime=False) -> None:
        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self.sorted_checkpoints(output_dir, use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return
        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            print("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint), file=sys.stderr,
                  flush=True)
            shutil.rmtree(checkpoint)

    def save_checkpoint(self):
        output_dir = os.path.join(
            self.args.output_path, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
        )
        os.makedirs(output_dir, exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model.state_dict(), os.path.join(output_dir, MODEL_SAVE_NAME))
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_SAVE_NAME))
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_SAVE_NAME))
        config_fp = open(os.path.join(output_dir, CONFIG_SAVE_NAME), "w")
        config_fp.write(self.model_config.to_json())
        config_fp.close()
        print("Saving optimizer and scheduler states to {}".format(output_dir), file=sys.stderr, flush=True)
        self.rotate_checkpoints(self.args.output_path, self.args.save_total_limit)

    def train_epoch(self, epoch):
        if dist.is_primary():
            loader = tqdm(self.dataloader)
        else:
            loader = self.dataloader

        criterion = nn.MSELoss()

        latent_loss_weight = 0.25
        sample_size = 25

        mse_sum = 0
        mse_n = 0

        for i, img in enumerate(loader):
            self.model.zero_grad()
            img = img.to(self.device)

            outputs = self.model(img)
            out, latent_loss = outputs[:2]
            recon_loss = criterion(out, img)
            latent_loss = latent_loss.mean()
            loss = recon_loss + latent_loss_weight * latent_loss
            if self.args.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backword()
            else:
                loss.backward()

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            part_mse_sum = recon_loss.item() * img.shape[0]
            part_mse_n = img.shape[0]
            comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
            comm = dist.all_gather(comm)

            for part in comm:
                mse_sum += part["mse_sum"]
                mse_n += part["mse_n"]

            self.global_step += 1

            if dist.is_primary() and self.global_step % self.args.logging_steps == 0:
                print(
                    "global_step", self.global_step,
                    "mse", "{:.4g}".format(recon_loss.item()),
                    "latent", "{:.4g}".format(latent_loss.item()),
                    "avg_mse", "{:.4g}".format(mse_sum / mse_n),
                    "lr", "{:.4g}".format(self.optimizer.param_groups[0]["lr"]),
                    file=sys.stderr, flush=True
                )

            if dist.is_primary() and self.global_step % self.args.save_steps == 0:
                self.save_checkpoint()

            if dist.is_primary() and self.global_step % self.args.eval_steps == 0:
                self.model.eval()
                sample = img[:sample_size]
                with torch.no_grad():
                    out = self.model(sample)[0]
                utils.save_image(
                    torch.cat([sample, out], 0),
                    f"{self.args.eval_path}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )
                self.model.train()

    def train(self):
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
        self.save_checkpoint()


def build_transform(size):
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


class ImageDataset(IterableDataset):
    def __init__(
            self,
            img_root_path,
            img_keys_path,
            transform,
            batch_size,
            dist_mode: bool = False,
            rank_seed: Optional[int] = None,
            with_key: bool = False
    ):
        self.img_root_path = img_root_path
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
        self.rand.shuffle(self.img_keys_file_list)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0
        pic_size = int(math.ceil(len(self.img_keys_file_list) / num_workers))
        for f in self.img_keys_file_list[pic_size * worker_id: pic_size * worker_id + pic_size]:
            for line in open(f):
                img_key = line.strip()
                img_path = get_key_path(self.img_root_path, img_key)
                try:
                    img = default_loader(img_path)
                except:
                    continue
                if self.with_key:
                    yield img_key, self.transform(img)
                else:
                    yield self.transform(img)


class IterDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, num_workers=0, collate_fn=None, pin_memory=False):
        super(IterDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory
        )

    def __len__(self):
        return int(math.ceil(len(self.dataset) / self.batch_size))


def main(args):
    model_config_json = open(args.config_path).read()
    print("ModelConfig:", model_config_json, file=sys.stderr, flush=True)
    model_config = VqvaeConfig.from_json(model_config_json)

    args.distributed = dist.get_world_size() > 1
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        if args.distributed:
            device = torch.device("cuda", dist.get_local_rank())
            torch.cuda.set_device(device)
        else:
            device = torch.device("cuda")

    transform = build_transform(args.size)

    dataset = ImageDataset(args.img_root_path, args.img_keys_path, transform, args.batch_size, args.distributed,
                           int(time.time()))
    local_batch_size = args.batch_size
    if args.distributed:
        local_batch_size = local_batch_size // dist.get_world_size()
    print("local_batch_size={}".format(local_batch_size), file=sys.stderr, flush=True)
    loader = IterDataLoader(
        dataset, batch_size=local_batch_size, num_workers=1, pin_memory=True
    )

    model = VQVAE(model_config).to(device)

    trained_steps, recent_ckpt = find_recent_checkpoint(args.output_path)
    if recent_ckpt is not None:
        model.load_state_dict(
            torch.load(os.path.join(recent_ckpt, MODEL_SAVE_NAME), map_location=device))
        print("load ckpt {}".format(recent_ckpt), file=sys.stderr, flush=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        '''
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epochs,
            momentum=None,
            warmup_proportion=0.05,
        )
        '''
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=args.lr / 10,
            max_lr=args.lr,
            cycle_momentum=False
        )
    if recent_ckpt is not None:
        if os.path.isfile(os.path.join(recent_ckpt, OPTIMIZER_SAVE_NAME)):
            optimizer.load_state_dict(
                torch.load(os.path.join(recent_ckpt, OPTIMIZER_SAVE_NAME), map_location=device)
            )
        if os.path.isfile(os.path.join(recent_ckpt, SCHEDULER_SAVE_NAME)):
            scheduler.load_state_dict(torch.load(os.path.join(recent_ckpt, SCHEDULER_SAVE_NAME)))

    if args.fp16:
        if not is_apex_available():
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    trainer = Trainer(args, loader, model_config, model, optimizer, scheduler, device, trained_steps)
    trainer.train()


def get_key_path(root_dir: str, key: str) -> str:
    md5 = hashlib.md5(key.encode("utf-8")).hexdigest()
    filename = urllib.parse.quote_plus(key)
    return os.path.join(root_dir, md5[:3], md5[3:6], filename)


def load_img_path_list(root_dir: str, list_file: str) -> List[str]:
    ret = []
    linecnt = 0
    for line in open(list_file):
        linecnt += 1
        if linecnt % 1000000 == 0:
            print("load {} imgs".format(linecnt), file=sys.stderr, flush=True)
        img_path = get_key_path(root_dir, line.strip())
        ret.append(img_path)
        if len(ret) > 1000000:
            break
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    port = (
            2 ** 15
            + 2 ** 14
            + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--fp16_opt_level", type=str, default="01")
    parser.add_argument("--img_keys_path", type=str, default=None)
    parser.add_argument("--img_root_path", type=str, default=None)

    args = parser.parse_args()

    assert os.path.isdir(args.img_keys_path)
    assert os.path.isdir(args.img_root_path)

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.eval_path, exist_ok=True)

    print(args, file=sys.stderr, flush=True)

    proc_num = 1
    if args.device == "cuda":
        proc_num = torch.cuda.device_count()
    print("proc_num={}".format(proc_num), file=sys.stderr, flush=True)
    dist.launch(main, proc_num, 1, 0, args.dist_url, args=(args,))

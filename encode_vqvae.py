# coding=utf-8

import sys
import os
import argparse
from torch import multiprocessing as mp
from torch.multiprocessing import Process
import math
import lmdb
try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass

import torch
from torchvision.datasets.folder import default_loader

from train_vqvae import build_transform, get_key_path
from vqvae import VQVAE
from vqvae_config import VqvaeConfig
from train_vqvae import ImageDataset, IterDataLoader


def encode_proc(
        model_path,
        model_config_path,
        img_root_path,
        img_key_path_list,
        img_size,
        device,
        output_path
):
    model_config_json = open(model_config_path).read()
    print("ModelConfig:", model_config_json, file=sys.stderr, flush=True)
    model_config = VqvaeConfig.from_json(model_config_json)
    model = VQVAE(model_config).to(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()

    transforms = build_transform(img_size)

    output_fp = open(output_path, "w")
    linecnt = 0
    for f in img_key_path_list:
        for line in open(f):
            linecnt += 1
            if linecnt % 100000 == 0:
                print("{} {} done".format(f, linecnt), file=sys.stderr, flush=True)
            img_key = line.strip()
            img_path = get_key_path(img_root_path, line.strip())
            try:
                img = default_loader(img_path)
            except:
                continue
            img = transforms(img)[None].to(device)
            id_t = model(img)[2].detach().cpu().flatten(1)
            print("{}\t{}".format(img_key, ",".join((str(x) for x in id_t[0].tolist()))), file=output_fp, flush=True)
    output_fp.close()


def collate_fn(batch):
    key_list = []
    img_list = []
    for key, img in batch:
        key_list.append(key)
        img_list.append(img[None])
    img_batch = torch.cat(img_list, dim=0)
    return key_list, img_batch


def encode(args):
    model_config_json = open(args.config_path).read()
    print("ModelConfig:", model_config_json, file=sys.stderr, flush=True)
    model_config = VqvaeConfig.from_json(model_config_json)
    device = torch.device(args.device)
    n_gpu = torch.cuda.device_count() if args.device == "cuda" else 0
    model = VQVAE(model_config).to(device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device)
    )
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    trans = build_transform(args.img_size)
    dataset = ImageDataset(
        args.img_root_path, args.img_key_path, trans, args.batch_size, with_key=True
    )
    dataloader = IterDataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    lmdb_env = lmdb.open(args.output_path, map_size=int(1e12))
    lmdb_txn = lmdb_env.begin(write=True)
    commit_cache = 0
    batch_cnt = 0
    for key_list, img_batch in dataloader:
        id_t_batch = model(img_batch)[2].detach().cpu().flatten(1)
        for key, id_t in zip(key_list, id_t_batch):
            lmdb_txn.put(key.encode("utf-8"), id_t.to(torch.int16).numpy().tobytes())
        commit_cache += id_t_batch.shape[0]
        if commit_cache > 1000:
            lmdb_txn.commit()
            lmdb_txn = lmdb_env.begin(write=True)
            commit_cache = 0
        batch_cnt += 1
        if batch_cnt % 100 == 0:
            print("{} batch done".format(batch_cnt), file=sys.stderr, flush=True)
    lmdb_txn.commit()
    lmdb_env.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img_root_path", type=str, default=None)
    parser.add_argument("--img_key_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    if not os.path.exists(args.img_root_path) or not os.path.isdir(args.img_root_path):
        raise ValueError("img_root_path must be a directory")
    if not os.path.exists(args.img_key_path) or not os.path.isdir(args.img_key_path):
        raise ValueError("img_key_path must be a directory")
    os.makedirs(args.output_path, exist_ok=True)
    if os.path.exists(args.output_path) and not os.path.isdir(args.output_path):
        raise ValueError("output_path must be a directory")
    '''
    img_key_path_list = [
        os.path.join(args.img_key_path, f) for f in os.listdir(args.img_key_path) if not f.startswith(".")
    ]
    device_list = []
    if args.device == "cpu":
        device_list = [torch.device("cpu")]
    else:
        device_list = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]

    proc_pic = int(math.ceil(len(img_key_path_list) / len(device_list)))
    proc_list = []
    for device_i in range(len(device_list)):
        device = device_list[device_i]
        input_list = img_key_path_list[device_i * proc_pic: (device_i + 1) * proc_pic]
        output_path = os.path.join(args.output_path, "part-{}".format(device_i))
        proc_list.append(Process(
            target=encode_proc,
            args=(args.model_path, args.config_path, args.img_root_path,
                  input_list, args.img_size, device, output_path)
        ))

    for proc in proc_list:
        proc.start()
    for proc in proc_list:
        proc.join()
    '''
    encode(args)


if __name__ == "__main__":
    main()
    exit(0)
    #config_path = "output_full_v2/ckpt/checkpoint-206000/config.json"
    #model_path = "output_full_v2/ckpt/checkpoint-206000/pytorch_model.bin"
    #config_path = "output_base/ckpt/checkpoint-217000/config.json"
    #model_path = "output_base/ckpt/checkpoint-217000/pytorch_model.bin"
    config_path = "output_full_base_cycle/ckpt/checkpoint-60000/config.json"
    model_path = "output_full_base_cycle/ckpt/checkpoint-60000/pytorch_model.bin"
    model = VQVAE(VqvaeConfig.from_json(
        open(config_path).read())).to("cpu")
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model.eval()
    img_path = get_key_path("/mnt2/makai/imgs", "img/ochvq0GGSkRVPRjCvf1edchz50MM1585628833055_1.jpg")
    trans = build_transform(224)
    img = default_loader(img_path)
    img = trans(img)[None]
    print(img)
    id_t, id_b = model(img)[2:4]
    print(id_t)
    print(id_b)
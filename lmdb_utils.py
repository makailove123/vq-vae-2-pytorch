# coding=utf-8

import sys
import os
import lmdb
import numpy as np
from train_vqvae import get_key_path


def build_lmdb_from_encode_result(args):
    input_dir = args[0]
    output_dir = args[1]

    lmdb_env = lmdb.open(output_dir, map_size=1099511627776)
    txn = lmdb_env.begin(write=True)

    linecnt = 0
    for f in os.listdir(input_dir):
        if f.startswith("."):
            continue
        for line in open(os.path.join(input_dir, f)):
            linecnt += 1
            if linecnt % 10000 == 0:
                print("{} done".format(linecnt), file=sys.stderr, flush=True)
            arr = line.rstrip("\r\n").split("\t")
            key = arr[0]
            val = np.asarray([int(x) for x in arr[1].split(",")], dtype=np.int16)
            txn.put(key.encode("utf-8"), val.tobytes())
    txn.commit()
    lmdb_env.close()


def build_img_lmdb(args):
    img_root_path = args[0]
    img_key_list_file = args[1]
    output_dir = args[2]

    lmdb_env = lmdb.open(output_dir, map_size=1099511627776)
    txn = lmdb_env.begin(write=True)

    linecnt = 0
    commit_cache = 0
    for line in open(img_key_list_file):
        linecnt += 1
        if linecnt % 10000 == 0:
            print("{} done".format(linecnt), file=sys.stderr, flush=True)
        img_key = line.strip()
        key_enc = img_key.encode("utf-8")
        img_path = get_key_path(img_root_path, img_key)
        try:
            img = open(img_path, "rb").read()
        except:
            continue
        txn.put(key_enc, img)
        commit_cache += 1
        if commit_cache > 10000:
            txn.commit()
            txn = lmdb_env.begin(write=True)
            commit_cache = 0
    txn.commit()
    lmdb_env.close()


if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2:])
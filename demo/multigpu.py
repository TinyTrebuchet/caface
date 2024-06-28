import os
import sys
import math
import glob
import argparse
import pyrootutils
import os.path as osp

import numpy as np
import torch

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
sys.path.append(os.path.dirname(root))
sys.path.append(os.path.join(os.path.dirname(root), 'caface'))

from model_loader import load_caface
from dataset import prepare_imagelist_dataloader
from inference import infer_features, fuse_feature

torch.distributed.init_process_group(backend="nccl")

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
gpu = f'cuda:{rank}'
print(f"[+] Rank {rank} : Worker initialized with {gpu}!")

parser = argparse.ArgumentParser(description='')
parser.add_argument("--ckpt_path", type=str, default='../pretrained_models/CAFace_AdaFaceWebFace4M.ckpt')
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--fusion_method", type=str,
                    default='cluster_and_aggregate',
                    choices=['cluster_and_aggregate', 'average'])
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

in_dir = args.input
out_dir = args.output
pids = set(os.listdir(in_dir))
done_pids = set(os.listdir(out_dir)) if osp.exists(out_dir) else set()
rem_pids = sorted(list(pids - done_pids))
work = math.ceil(len(rem_pids) / world_size)
local_pids = rem_pids[work*rank : work*(rank+1)]

# load caface
aggregator, model, hyper_param = load_caface(args.ckpt_path, device=gpu)

def infer(seq_dir):
    probe_image_list = sorted(glob.glob(osp.join(seq_dir, '*.png'), recursive=True))
    if len(probe_image_list) == 0:
        return np.empty((1, 512), dtype=np.float32)

    dataloader = prepare_imagelist_dataloader(probe_image_list, batch_size=args.batch_size, num_workers=0)

    # infer singe image features
    probe_features, probe_intermediates = infer_features(dataloader, model, aggregator, hyper_param, device=gpu)
    # fuse features
    probe_fused_feature, _ = fuse_feature(probe_features, aggregator, probe_intermediates,
                                                        method=args.fusion_method, device=gpu)
    
    return (np.expand_dims(probe_fused_feature, axis=0))

for i,pid in enumerate(sorted(local_pids)):
    print(f"Rank {rank} | Processing subject {pid} ({i}/{len(local_pids)})")
    for phase in sorted(os.listdir(osp.join(in_dir, pid))):
        for view in sorted(os.listdir(osp.join(in_dir, pid, phase))):
            seq_dir = osp.join(in_dir, pid, phase, view)
            save_dir = osp.join(out_dir, pid, phase, view)
            os.makedirs(save_dir, exist_ok=True)
            feat = infer(seq_dir)
            np.save(osp.join(save_dir, 'face.npy'), feat)

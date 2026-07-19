#!/usr/bin/env python3
"""v11 prep — domain-BALANCED retrain. Unlike v10 (from base, 20ep -> gold regressed), v11 continues
from v5 gently and upsamples the flywheel so it has signal without a wholesale distribution shift.
Reuses v10's flywheel train/eval split (data/v10/*) so the flywheel eval is directly comparable."""
import os, json
ROOT = "/home/ece/BigDisk/Prathosh/ASR"
UP = int(os.environ.get("UPSAMPLE", "3"))                 # flywheel replay factor
base = [l for l in open(f"{ROOT}/data/train_ft5.jsonl").read().rstrip("\n").split("\n") if l]
fly = [l for l in open(f"{ROOT}/data/v10/flywheel_train.jsonl").read().rstrip("\n").split("\n") if l]
with open(f"{ROOT}/data/v11_train.jsonl", "w") as f:
    f.write("\n".join(base) + "\n")
    for _ in range(UP):
        f.write("\n".join(fly) + "\n")
tot = len(base) + UP * len(fly)
print(f"v11 train: {len(base)} v5-data + {UP}x{len(fly)} flywheel = {tot} rows "
      f"(flywheel share {100*UP*len(fly)/tot:.0f}%)  | init=v5, low-LR continuation")

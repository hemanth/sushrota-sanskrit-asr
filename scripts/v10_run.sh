#!/bin/bash
# v10: finetune v5's recipe + flywheel (pass-clean + override) from base. Eval v5 vs v10 on the
# standard 3-domain suite AND the held-out flywheel sessions (the in-the-wild distribution). GPU1.
set -e
cd /home/ece/BigDisk/Prathosh/ASR
echo "[v10] prep"
envs/nemo_ai4b/bin/python scripts/v10_prep.py
echo "[v10] train rows: $(wc -l < data/v10_train.jsonl)"
CUDA_VISIBLE_DEVICES=1 envs/nemo_ai4b/bin/python scripts/finetune_ctc_init.py \
  --train data/v10_train.jsonl --outdir exp/ft_ctc_v10 --epochs 20 --save-eps 15,20 --bs 16 --lr 1e-4
echo "[v10] eval"
SN=(gold chant prose flywheel)
SM=(data/epgp/eval_gold.jsonl data/utts_fa/manifest_fa_eval_sk10.jsonl \
    data/utts_fa/manifest_prose_eval_vtn.jsonl data/v10/flywheel_eval.jsonl)
for i in "${!SN[@]}"; do
  CUDA_VISIBLE_DEVICES=1 envs/nemo_ai4b/bin/python scripts/eval_model.py \
    --model exp/ft_ctc_v5/ft_ctc_ep20.nemo --manifest "${SM[$i]}" --tag "v5|${SN[$i]}" 2>>logs/v10.err
done
for ep in 15 20; do
  for i in "${!SN[@]}"; do
    CUDA_VISIBLE_DEVICES=1 envs/nemo_ai4b/bin/python scripts/eval_model.py \
      --model exp/ft_ctc_v10/ft_ctc_ep${ep}.nemo --manifest "${SM[$i]}" --tag "v10-ep${ep}|${SN[$i]}" 2>>logs/v10.err
  done
done
echo "V10 ALL DONE"

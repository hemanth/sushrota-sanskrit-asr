#!/bin/bash
# v11: domain-balanced. CONTINUE from v5 (not base) at LOW LR for FEW epochs, flywheel upsampled 3x.
# Save ep3/6/9 to read the gold-vs-flywheel tradeoff curve. Eval v5 vs each on the 4-domain suite. GPU1.
set -e
cd /home/ece/BigDisk/Prathosh/ASR
echo "[v11] prep"
UPSAMPLE=3 envs/nemo_ai4b/bin/python scripts/v11_prep.py
echo "[v11] train rows: $(wc -l < data/v11_train.jsonl)"
CUDA_VISIBLE_DEVICES=1 envs/nemo_ai4b/bin/python scripts/finetune_ctc_init.py \
  --init exp/ft_ctc_v5/ft_ctc_ep20.nemo \
  --train data/v11_train.jsonl --outdir exp/ft_ctc_v11 --epochs 9 --save-eps 3,6,9 --bs 16 --lr 3e-5
echo "[v11] eval"
SN=(gold chant prose flywheel)
SM=(data/epgp/eval_gold.jsonl data/utts_fa/manifest_fa_eval_sk10.jsonl \
    data/utts_fa/manifest_prose_eval_vtn.jsonl data/v10/flywheel_eval.jsonl)
for i in "${!SN[@]}"; do
  CUDA_VISIBLE_DEVICES=1 envs/nemo_ai4b/bin/python scripts/eval_model.py \
    --model exp/ft_ctc_v5/ft_ctc_ep20.nemo --manifest "${SM[$i]}" --tag "v5|${SN[$i]}" 2>>logs/v11.err
done
for ep in 3 6 9; do
  for i in "${!SN[@]}"; do
    CUDA_VISIBLE_DEVICES=1 envs/nemo_ai4b/bin/python scripts/eval_model.py \
      --model exp/ft_ctc_v11/ft_ctc_ep${ep}.nemo --manifest "${SM[$i]}" --tag "v11-ep${ep}|${SN[$i]}" 2>>logs/v11.err
  done
done
echo "V11 ALL DONE"

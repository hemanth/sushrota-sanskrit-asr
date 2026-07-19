#!/usr/bin/env python3
"""Point app_sushrota at the SHARED core-ASR pointer (exp/ft_ctc_current.nemo) so it always tracks
the same model as Vāgbodhinī. Idempotent."""
import ast, sys
F = "/home/ece/BigDisk/Prathosh/ASR/scripts/app_sushrota.py"
src = open(F).read()
OLD = 'M = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location="cpu")'
NEW = ('MODEL_PATH = os.environ.get("MODEL_PATH", f"{ROOT}/exp/ft_ctc_current.nemo")  # shared core ASR (same pointer as Vāgbodhinī)\n'
       'M = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL_PATH, map_location="cpu")')
if "exp/ft_ctc_current.nemo" in src:
    print("[skip] already on shared pointer"); sys.exit(0)
if OLD not in src:
    print("[FAIL] model-load anchor not found"); sys.exit(1)
src = src.replace(OLD, NEW, 1)
ast.parse(src)
open(F, "w").write(src)
print("[done] app_sushrota now reads exp/ft_ctc_current.nemo")

#!/usr/bin/env python3
"""Add corrections-only data collection to app_sushrota /feedback. New kind='corrections' carries a
JSON list [{seg, raw, text}]; for each seg whose human-corrected text differs from the ASR raw, we
pair it with that segment's stored audio (FLY/audio/{sess}_{seg}.wav) and save a gold training pair
to data/sushrota_flywheel/ (same schema as the Vāgbodhinī flywheel, so harvest reads both).
Additive + idempotent — existing /feedback behaviour is unchanged."""
import ast, sys
F = "/home/ece/BigDisk/Prathosh/ASR/scripts/app_sushrota.py"
src = open(F).read()
if "sushrota_flywheel" in src:
    print("[skip] already patched"); sys.exit(0)

# 1) add the flywheel dir + a save helper right after the FLY definition
ANCHOR1 = 'FLY = f"{ROOT}/data/flywheel"; os.makedirs(f"{FLY}/audio", exist_ok=True)'
ADD1 = ANCHOR1 + '''
SFLY = f"{ROOT}/data/sushrota_flywheel"; os.makedirs(f"{SFLY}/audio", exist_ok=True)  # corrections-only training pairs
SS_VER = "sushrota-2026-07-19"
import shutil as _shutil, soundfile as _sf
def _save_correction(sess, seg, corrected):
    ap = f"{FLY}/audio/{sess}_{seg}.wav"                     # the segment audio logSegment already stored
    if not os.path.exists(ap): return False
    cid = uuid.uuid4().hex[:12]
    dst = f"{SFLY}/audio/{cid}.wav"
    try:
        _shutil.copyfile(ap, dst)
        try: dur = round(_sf.info(dst).duration, 2)
        except Exception: dur = 0.0
        with open(f"{SFLY}/log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": cid, "t": int(time.time()), "tier": "corrected",
                                 "text": corrected, "dur": dur, "session": sess, "ver": SS_VER,
                                 "source": "sushrota"}, ensure_ascii=False) + "\\n")
        return True
    except Exception:
        return False'''

# 2) handle kind='corrections' inside /feedback (before the events.jsonl write)
ANCHOR2 = '''    rec = {"kind": kind, "session": sess, "ts": ts, "suggest": suggest}
    if kind == "segment":'''
ADD2 = '''    rec = {"kind": kind, "session": sess, "ts": ts, "suggest": suggest}
    if kind == "corrections":                               # corrections-only: pair (audio, corrected)
        n = 0
        try: items = json.loads(corrections) if corrections else []
        except Exception: items = []
        for it in items:
            raw_i = norm(it.get("raw", "")); cor_i = norm(it.get("text", ""))
            if cor_i and cor_i != raw_i and _save_correction(sess, str(it.get("seg", "")), cor_i): n += 1
        rec["corrections"] = len(items); rec["saved"] = n
        with open(f"{FLY}/events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        return {"stored": True, "saved": n}
    if kind == "segment":'''

# 3) add the `corrections` form field to the signature
ANCHOR3 = '                   corrected_text: str = Form(""), suggest: str = Form(""),'
ADD3 = '                   corrected_text: str = Form(""), corrections: str = Form(""), suggest: str = Form(""),'

for name, a, b in [("helper", ANCHOR1, ADD1), ("handler", ANCHOR2, ADD2), ("signature", ANCHOR3, ADD3)]:
    if a not in src: print(f"[FAIL] anchor not found: {name}"); sys.exit(1)
    src = src.replace(a, b, 1)
ast.parse(src)
open(F, "w").write(src)
print("[done] app_sushrota: corrections-only collection -> data/sushrota_flywheel/")

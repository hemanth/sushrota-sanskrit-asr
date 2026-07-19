#!/usr/bin/env python3
"""v10 prep: finetune v5 on its own data + the flywheel's TRUSTWORTHY tiers (pass n_red==0 +
override). Split flywheel by SESSION (no speaker leakage) into train/eval so we can measure whether
the in-the-wild data helps on the in-the-wild distribution, not just the clean studio eval."""
import os, json, random
ROOT = "/home/ece/BigDisk/Prathosh/ASR"
FLYW = f"{ROOT}/data/practice_flywheel"
OUT = f"{ROOT}/data/v10"; os.makedirs(OUT, exist_ok=True)
EVAL_FRAC = 0.15
rng = random.Random(42)

rows = []
for l in open(f"{FLYW}/log.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    tier = r.get("tier")
    if tier == "pass" and (r.get("n_red") or 0) == 0:      # clean pass (no confident errors)
        pass
    elif tier == "override":                                # human-affirmed / corrected label
        pass
    else:
        continue
    p = f"{FLYW}/audio/{r['id']}.wav"
    t = (r.get("text") or "").strip()
    dur = r.get("dur") or 0
    if not (t and os.path.exists(p) and 0.4 <= dur <= 30):
        continue
    rows.append({"audio_filepath": p, "text": t, "duration": round(dur, 2), "lang": "sa",
                 "session": r.get("session", ""), "tier": tier})

# split by session -> no speaker leakage between train/eval
sessions = sorted({r["session"] for r in rows if r["session"]})
rng.shuffle(sessions)
n_eval = max(1, int(EVAL_FRAC * len(sessions)))
eval_sess = set(sessions[:n_eval])
tr = [r for r in rows if r["session"] not in eval_sess]
ev = [r for r in rows if r["session"] in eval_sess]

def w(path, data):
    with open(path, "w") as f:
        for r in data:
            f.write(json.dumps({k: r[k] for k in ("audio_filepath", "text", "duration", "lang")},
                               ensure_ascii=False) + "\n")

w(f"{OUT}/flywheel_train.jsonl", tr)
w(f"{OUT}/flywheel_eval.jsonl", ev)

# combine with v5's exact training data -> v10 train set (finetune from base, like v9)
base = open(f"{ROOT}/data/train_ft5.jsonl").read().rstrip("\n").split("\n")
with open(f"{ROOT}/data/v10_train.jsonl", "w") as f:
    f.write("\n".join(base) + "\n")
    for r in tr:
        f.write(json.dumps({"audio_filepath": r["audio_filepath"], "text": r["text"],
                            "duration": r["duration"], "lang": "sa"}, ensure_ascii=False) + "\n")

print(f"flywheel usable (pass-clean+override): {len(rows)} clips / {len(sessions)} sessions")
print(f"  train: {len(tr)} clips  ({round(sum(r['duration'] for r in tr)/60,1)} min)")
print(f"  eval : {len(ev)} clips / {len(eval_sess)} held-out sessions")
print(f"v10 train set: {len(base)} (v5 data) + {len(tr)} (flywheel) = {len(base)+len(tr)} rows")

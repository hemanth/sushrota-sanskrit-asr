#!/usr/bin/env python3
"""Validation prep: carve a BIGGER held-out (~300 clips, by session) from sessions that v12-ep9
NEVER trained on, so both v12-ep9 (existing) and a fresh v13b (trained on everything except this
held-out) are judged on genuinely unseen data. Confirms whether v13's SN-WER win is real at scale."""
import os, json, random, collections
ROOT = "/home/ece/BigDisk/Prathosh/ASR"
FLYW = f"{ROOT}/data/practice_flywheel"
OUT = f"{ROOT}/data/v13b"; os.makedirs(OUT, exist_ok=True)
rng = random.Random(7); TARGET = 300

# v12-ep9's training clip ids -> sessions (the held-out must avoid these so v12-ep9 is unseen)
v12_ids = {os.path.basename(json.loads(l)["audio_filepath"])[:-4]
           for l in open(f"{ROOT}/data/v12/flywheel_train.jsonl")}
orig_held = {os.path.basename(json.loads(l)["audio_filepath"])[:-4]
             for l in open(f"{ROOT}/data/v10/flywheel_eval.jsonl")}

id2sess = {}; clean = []
for l in open(f"{FLYW}/log.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    if r.get("session"): id2sess[r["id"]] = r["session"]
    tier = r.get("tier")
    if not ((tier == "pass" and (r.get("n_red") or 0) == 0) or tier == "override"): continue
    p = f"{FLYW}/audio/{r['id']}.wav"; t = (r.get("text") or "").strip(); dur = r.get("dur") or 0
    if t and os.path.exists(p) and 0.4 <= dur <= 30:
        clean.append({"id": r["id"], "audio_filepath": p, "text": t,
                      "duration": round(dur, 2), "session": r.get("session", "")})

v12_sess = {id2sess.get(i) for i in v12_ids}
sess_clips = collections.defaultdict(list)
for c in clean: sess_clips[c["session"]].append(c)
# held-out pool = sessions v12-ep9 never trained on (and non-empty session id)
pool = sorted(s for s in sess_clips if s and s not in v12_sess)
rng.shuffle(pool)
H_sess = set(); n = 0
for s in pool:
    if n >= TARGET: break
    H_sess.add(s); n += len(sess_clips[s])

H = [c for c in clean if c["session"] in H_sess]
train = [c for c in clean if c["session"] not in H_sess and c["id"] not in orig_held]

def w(path, data):
    with open(path, "w") as f:
        for r in data:
            f.write(json.dumps({k: r[k] for k in ("audio_filepath", "text", "duration")} | {"lang": "sa"},
                               ensure_ascii=False) + "\n")
w(f"{OUT}/flywheel_eval_big.jsonl", H)
w(f"{OUT}/flywheel_train.jsonl", train)

base = [l for l in open(f"{ROOT}/data/train_ft5.jsonl").read().rstrip("\n").split("\n") if l]
with open(f"{ROOT}/data/v13b_train.jsonl", "w") as f:
    f.write("\n".join(base) + "\n")
    for r in train:
        f.write(json.dumps({"audio_filepath": r["audio_filepath"], "text": r["text"],
                            "duration": r["duration"], "lang": "sa"}, ensure_ascii=False) + "\n")

# sanity: none of H is in v12's training (v12-ep9 must be unseen)
leak = sum(1 for c in H if c["id"] in v12_ids)
print(f"BIG held-out H: {len(H)} clips / {len(H_sess)} sessions  (v12-training leakage: {leak} — must be 0)")
print(f"v13b flywheel train: {len(train)} clips / {len({c['session'] for c in train})} sessions")
print(f"v13b train: {len(base)} v5-data + {len(train)} flywheel = {len(base)+len(train)} rows")

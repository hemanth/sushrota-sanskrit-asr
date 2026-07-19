#!/usr/bin/env python3
"""Harvest the Vāgbodhinī consented flywheel into a curated NeMo training manifest.

EVERY attempt is logged (nothing discarded); the tier records trustworthiness of the (audio,label):
  override : ASR-flagged but user affirmed correct  -> real failures, TRUSTED label (highest value)
  pass     : >=90% match                             -> label trusted, but easy (reinforcement)
  review   : 55..90% match                           -> ASR struggled; label AMBIGUOUS (needs review)
  low      : <55% match                              -> ASR-hard or wrong verse; ARCHIVE (analysis)
  unclear  : ~nothing recognised (noise/off-mic)     -> ARCHIVE
  unscored : ASR was down when recorded              -> ARCHIVE (audio+label still good; re-score later)

Training set = override + pass. `review` -> separate file for triage. low/unclear/unscored are
ARCHIVED (kept, counted) but never auto-trained.

Usage:
  python harvest_flywheel.py                 # stats + write train manifest (override+pass)
  python harvest_flywheel.py --include-review # also fold in reviewed 'review' rows
  python harvest_flywheel.py --clean-only    # only rows with n_red==0 (strictest)
"""
import os, json, sys, wave, contextlib
from collections import Counter, defaultdict
ROOT = "/home/ece/BigDisk/Prathosh/ASR"
FLYW = f"{ROOT}/data/practice_flywheel"
LOG = f"{FLYW}/log.jsonl"
INCLUDE_REVIEW = "--include-review" in sys.argv
CLEAN_ONLY = "--clean-only" in sys.argv

def wav_dur(p):
    try:
        with contextlib.closing(wave.open(p)) as w: return round(w.getnframes() / w.getframerate(), 2)
    except Exception: return 0.0

rows = []
if os.path.exists(LOG):
    for l in open(LOG):
        try: rows.append(json.loads(l))
        except Exception: pass

by_tier = Counter(r.get("tier", "?") for r in rows)
by_script = Counter(r.get("script", "?") for r in rows)
print(f"[flywheel] {len(rows)} records @ {LOG}")
print(f"  tiers  : {dict(by_tier)}")
print(f"  scripts: {dict(by_script)}")
tot_dur = defaultdict(float)
for r in rows: tot_dur[r.get("tier", "?")] += r.get("dur", 0) or 0
print(f"  hours  : " + "  ".join(f"{k}={v/3600:.2f}h" for k, v in tot_dur.items()))

train, review = [], []
for r in rows:
    p = f"{FLYW}/audio/{r['id']}.wav"
    if not os.path.exists(p): continue
    txt = (r.get("text") or "").strip()
    if not txt: continue
    dur = r.get("dur") or wav_dur(p)
    if not (0.4 <= dur <= 30): continue
    row = {"audio_filepath": p, "text": txt, "duration": dur, "lang": "sa",
           "tier": r.get("tier"), "percent": r.get("percent"), "hyp": r.get("hyp", "")}
    tier = r.get("tier")
    if tier in ("override", "pass"):
        if CLEAN_ONLY and (r.get("n_red") or 0) > 0 and tier == "pass": continue
        train.append(row)
    elif tier == "review":
        review.append(row)
    # low/unclear/unscored: archived (counted in stats above), not emitted to any manifest here

if INCLUDE_REVIEW: train += review

def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("audio_filepath", "text", "duration", "lang")},
                               ensure_ascii=False) + "\n")

tr = f"{FLYW}/manifest_train.jsonl"; rv = f"{FLYW}/manifest_review.jsonl"
write(tr, train); write(rv, review)
print(f"\n[out] train manifest : {len(train)} rows -> {tr}"
      f"  ({'incl. review' if INCLUDE_REVIEW else 'override+pass'}"
      f"{', clean-only' if CLEAN_ONLY else ''})")
print(f"[out] review queue   : {len(review)} rows -> {rv}  (triage before use)")
print("\nReview a few 'review'/'override' rows (audio vs text vs what-ASR-heard) before training:")
for r in (review[:3] + [x for x in train if x['tier'] == 'override'][:3]):
    print(f"  [{r['tier']} {r.get('percent')}%] ref={r['text'][:40]!r}  asr_heard={r.get('hyp','')[:40]!r}")

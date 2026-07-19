#!/usr/bin/env python3
"""Are the FIRST/SECOND aksharas systematically flagged (onset drop)? Measure per-position flag
rate from the real flywheel ops strings, and check whether onset flags are DELETIONS (amber, ASR
missed the sound) vs SUBSTITUTIONS (red)."""
import json, collections
ROOT = "/home/ece/BigDisk/Prathosh/ASR"
LOG = f"{ROOT}/data/practice_flywheel/log.jsonl"
import sys
TIERS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["pass", "override"]
pos_tot = collections.Counter(); pos_red = collections.Counter(); pos_amb = collections.Counter()
n = 0
for l in open(LOG):
    try: r = json.loads(l)
    except Exception: continue
    ops = r.get("ops")
    if not ops or r.get("tier") not in TIERS: continue     # restrict to trusted tiers -> flags = false pos
    if len(ops) < 4: continue
    n += 1
    for i, c in enumerate(ops[:12]):          # first 12 positions
        pos_tot[i] += 1
        if c == "r": pos_red[i] += 1
        elif c == "a": pos_amb[i] += 1
print(f"[n={n} clips | tiers={TIERS} — in trusted tiers a flag at pos k is a FALSE POSITIVE]")
print("pos | flagged%(red+amb) | red% | amber% | n")
for i in range(12):
    t = pos_tot[i]
    if not t: break
    fr = 100 * (pos_red[i] + pos_amb[i]) / t
    print(f" {i:2d} |   {fr:5.1f}%        | {100*pos_red[i]/t:4.1f}% | {100*pos_amb[i]/t:5.1f}% | {t}")
# summary: onset (0,1) vs body (2+)
def rate(idxs, cnt):
    tt = sum(pos_tot[i] for i in idxs); cc = sum(cnt[i] for i in idxs); return 100*cc/tt if tt else 0
onset = [0, 1]; body = list(range(2, 12))
print(f"\nONSET (pos 0-1): flagged {rate(onset,{i:pos_red[i]+pos_amb[i] for i in range(12)}):.1f}%  "
      f"(amber {rate(onset,pos_amb):.1f}%, red {rate(onset,pos_red):.1f}%)")
print(f"BODY  (pos 2+ ): flagged {rate(body,{i:pos_red[i]+pos_amb[i] for i in range(12)}):.1f}%  "
      f"(amber {rate(body,pos_amb):.1f}%, red {rate(body,pos_red):.1f}%)")

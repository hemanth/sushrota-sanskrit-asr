"""Single source of truth for the flywheel CLEAN training pool. Any retrain (v14+) should build
its flywheel share from clean_rows() so quality-tiering AND auto-rescued gold are always included.

Clean = (pass & n_red==0) OR (override user-corrections) OR (review clips auto-rescued to gold via
scripts/rescue.py — re-decoded by the then-deployed model and confirmed n_red==0 vs the reference),
subject to: non-empty label, audio exists, 0.4s ≤ dur ≤ 30s. Deduped by audio path."""
import os, json, collections
ROOT="/home/ece/BigDisk/Prathosh/ASR"; FLYW=f"{ROOT}/data/practice_flywheel"
RESCUED=f"{ROOT}/data/rescue/rescued.jsonl"

def clean_rows(include_rescued=True):
    rows=[]; seen=set()
    for l in open(f"{FLYW}/log.jsonl"):
        try: r=json.loads(l)
        except Exception: continue
        tier=r.get("tier")
        if not ((tier=="pass" and (r.get("n_red") or 0)==0) or tier=="override"): continue
        p=f"{FLYW}/audio/{r['id']}.wav"; t=(r.get("text") or "").strip(); dur=r.get("dur") or 0
        if t and os.path.exists(p) and 0.4<=dur<=30 and p not in seen:
            seen.add(p); rows.append({"audio_filepath":p,"text":t,"duration":round(dur,2),"lang":"sa","src":"flywheel"})
    if include_rescued and os.path.exists(RESCUED):
        for l in open(RESCUED):
            try: r=json.loads(l)
            except Exception: continue
            p=r["audio_filepath"]
            if p in seen or not os.path.exists(p): continue
            seen.add(p); rows.append({"audio_filepath":p,"text":r["text"],"duration":round(r.get("duration") or 0,2),"lang":"sa","src":"rescued"})
    return rows

if __name__=="__main__":
    rr=clean_rows(); c=collections.Counter(x["src"] for x in rr); h=sum(x["duration"] for x in rr)/3600
    print(f"CLEAN POOL: {len(rr)} clips / {round(h,2)} h  -> {dict(c)}")
    print(f"  (rescued adds {c['rescued']} hard confirmed-gold clips over the base clean filter)")

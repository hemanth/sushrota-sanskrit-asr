#!/usr/bin/env python3
"""Which voice/meter preserves PROSE onsets? Render the problem phrases with several meters via
:8020, ASR-decode each, and score onset preservation. Pick a non-gadya voice for prose mode."""
import os, io, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, soundfile as sf, torch, requests
import nemo.collections.asr as na
ROOT = "/home/ece/BigDisk/Prathosh/ASR"; OFF, V, BL = 4096, 256, 5632
M = na.models.EncDecHybridRNNTCTCBPEModel.restore_from(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location="cuda:0").eval()
LAB = json.load(open(f"{ROOT}/data/eval_logits/labels.json"))
def lse(x, ax): m = x.max(ax, keepdims=True); return m + np.log(np.exp(x - m).sum(ax, keepdims=True))
def greedy(wav):
    sig = torch.tensor(wav).unsqueeze(0).cuda(); sl = torch.tensor([len(wav)]).cuda()
    with torch.no_grad():
        enc, _ = M.forward(input_signal=sig, input_signal_length=sl); lp = M.ctc_decoder(encoder_output=enc)[0].cpu().numpy()
    P = lp[:, [BL] + list(range(OFF, OFF + V))]; P = P - lse(P, 1); ids = P.argmax(1)
    o = []; prev = -1
    for i in ids:
        i = int(i)
        if i != prev and i != 0: o.append(LAB[i - 1])
        prev = i
    return "".join(o).replace("▁", " ").strip()
def tts(text, meter):
    r = requests.post("http://localhost:8020/tts", data={"text": text, "meter": meter}, timeout=120)
    if r.status_code != 200: return None
    wav, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    if sr != 16000:
        idx = (np.arange(int(len(wav) * 16000 / sr)) * sr / 16000).astype(int); wav = wav[np.clip(idx, 0, len(wav) - 1)]
    return wav
METERS = json.loads(requests.get("http://localhost:8020/health").text)["meters"]
CANDS = [m for m in ["gadya", "gadya_mbtn", "anuṣṭubh", "upajāti", "vaṃśastha", "indravajrā"] if m in METERS]
PHRASES = ["अथातो ब्रह्मजिज्ञासा", "उक्तं मया", "शास्त्रयोनित्वात्", "इति एतत्", "सत्यमुक्तं किन्त्विह"]
print("bank meters:", METERS)
print("candidates:", CANDS, "\n")
def onset_ok(ref, heard):
    r2 = ref.replace(" ", "")[:2]; return heard.replace(" ", "").startswith(r2)
for m in CANDS:
    ok = 0; lines = []
    for ph in PHRASES:
        w = tts(ph, m)
        if w is None: lines.append(f"    FAIL {ph}"); continue
        h = greedy(w); good = onset_ok(ph, h); ok += good
        lines.append(f"    {'ok ' if good else 'DROP'} {ph!r:24s} -> {h!r}")
    print(f"=== meter '{m}': onset kept {ok}/{len(PHRASES)} ===")
    print("\n".join(lines))
print("\nDONE", flush=True)

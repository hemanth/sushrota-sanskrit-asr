#!/usr/bin/env python3
"""Does Vāgdhenu TTS drop onset syllables? Render varied-onset texts via :8020, then ASR-decode the
rendered audio (v5) to see what the TTS ACTUALLY produced at the start. Tests vowel-initial vs
consonant-initial, shloka (anuṣṭubh) vs gadya."""
import os, io, json, re, unicodedata
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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
TESTS = [
  ("gadya", "अथातो ब्रह्मजिज्ञासा"),        # vowel-initial (अ)
  ("gadya", "इति एतत्"),                      # vowel-initial (इ)
  ("gadya", "उक्तं मया"),                     # vowel-initial (उ)
  ("gadya", "एवमादि"),                        # vowel-initial (ए)
  ("gadya", "शास्त्रयोनित्वात्"),            # consonant-cluster onset (शा)
  ("gadya", "जन्माद्यस्य यतः"),              # consonant onset (ज)
  ("anuṣṭubh", "अथ शब्दानुशासनम्"),          # vowel-initial, shloka voice
  ("anuṣṭubh", "वसुदेवसुतं देवं"),           # consonant onset, shloka voice
]
for meter, txt in TESTS:
    w = tts(txt, meter)
    if w is None: print(f"[{meter}] RENDER FAIL: {txt}"); continue
    h = greedy(w)
    hit = "OK " if h.replace(" ", "").startswith(txt.replace(" ", "")[:2]) else "DROP"
    print(f"[{meter:9s}] {hit} ref={txt!r:34s} -> asr_heard={h!r}  ({len(w)/16000:.2f}s)")
print("DONE", flush=True)

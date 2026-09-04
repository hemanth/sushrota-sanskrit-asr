# Su-śrotā — Scholar-grade Sanskrit ASR & chant practice

Sanskrit speech recognition tuned for **śāstric / recitational** Sanskrit, plus **Vāgbodhinī**,
an interactive chant-practice tool built on it. This repository documents the full set of
experiments — including the ones that **did not** work, which were as instructive as the ones
that did.

- **ASR model:** IndicConformer-CTC (Sanskrit slice). "v5" is the studio-finetuned base; **v13b-ep3**
  is the current *deployed* model — v5 continued on consented real-user data (see §5b). Both are on
  [Hugging Face](https://huggingface.co/prathoshap/sushrota-sanskrit-asr), with the
  [training dataset](https://huggingface.co/datasets/prathoshap/sushrota-sanskrit-asr-data).
- **Live tools:** **[Vāgbodhinī](https://prathosh.in/vagbodhini/)** (chant/prose tutor) and
  **[Su-śrotā](https://prathosh.in/sushrotaa/)** (dictation/transcription) — both serve the *same*
  shared core ASR and both collect consented data that improves it.
- **Author:** Prof. Prathosh A P, Indian Institute of Science, Bengaluru.

> **Headline result (§5b):** the model looked *saturated on studio audio (~6% CER)* — but on **real,
> in-the-wild user audio it had large headroom**, and a consented **data flywheel** closed most of it.
> On a 327-clip leakage-free in-the-wild held-out, the deployed model cuts CER **7.7% → 4.4% (−43%)**
> and WER **45% → 30%** vs the studio-only base, with **no loss** on studio/chant/prose benchmarks.
> The flywheel now spans **~13,100 consented clips from ~1,500 users across 11 scripts**.

---

## 1. Motivation

Off-the-shelf Sanskrit ASR is trained on conversational IndicVoices-style data and degrades badly
on **recitation and śāstra** — dense compounds, sandhi, retroflex/aspirate contrasts, pitch, and
long metrical utterances. The goal here was a model and tooling good enough for **scholars**:
accurate on chant and prose, and useful as a *practice aid* rather than a transcription toy.

Two things shaped everything:
1. **Segmentation dominates Sanskrit WER.** Word boundaries in Sanskrit are largely a writing
   convention (sandhi fuses words). Standard WER punishes boundary disagreements that are not
   errors. We therefore report a **sandhi-normalized WER (SN-WER)** alongside CER/WER.
2. **The acoustic model saturates.** Past a point, adding data stops moving CER; the leverage moves
   to how you *use* the model (known-reference tasks, decoding, tooling), not how you train it.

---

## 2. Data & preprocessing

- **Recitation + śāstra prose** corpora (Bhāgavata chant, Vedānta prose), a TTS-speaker slice, and
  a **multi-scholar annotation** set collected via a purpose-built web portal.
- **Content-only normalization:** NFC, keep Devanāgarī letters + combining marks, drop digits,
  daṇḍa, avagraha, oṁ, and Vedic accent marks; collapse whitespace.
- **Metrics:**
  - **CER** — character error rate on space-stripped text (the primary, boundary-agnostic metric).
  - **WER** — whitespace-token error rate (segmentation-sensitive).
  - **SN-WER** — *sandhi-normalized* WER: strip spaces on both sides, char-align, a reference word
    counts correct iff all its characters are recovered. Reported as a band `lo..hi`.

Representative v5 numbers (held-out):

| domain | CER | WER | SN-WER |
|---|---|---|---|
| e-PG lecture (gold) | 3.6% | 13.0% | ~9–10% |
| Bhāgavata chant | 6.0% | 46.4% | ~22–26% |
| Vedānta prose | 7.3% | 30.8% | ~15–19% |

The gap between WER and SN-WER quantifies point (1): roughly **half of Sanskrit "WER" is spacing.**

---

## 3. The model (v5)

- **Architecture:** `EncDecHybridRNNTCTCBPEModel` (IndicConformer, ~129 M params). We use the **CTC
  head** on the Sanskrit token slice (`cols = [BLANK] + range(4096,4352)`, re-`log_softmax`), greedy
  decode.
- **Training data (v5):** recitation + disk-prose + TTS-speaker slice.
- **Result:** the studio base — chant CER **6.0%**; later continued on consented user data into the
  deployed **v13b-ep3** (§5b). Weights on Hugging Face (see §7).

---

## 4. Experiments — what we tried

A compressed log of the campaign. **Bold = shipped / kept.**

| # | Experiment | Idea | Outcome |
|---|---|---|---|
| 1 | **v5 finetune (CTC)** | finetune IndicConformer on recitation+prose+TTS | **shipped; chant CER 6.0%** |
| 2 | **SN-WER metric** | strip sandhi/spacing before scoring | **adopted; showed ½ of WER is spacing** |
| 3 | v8 pseudo-labels | self-train on high-confidence v5 outputs | ✗ confidence doesn't separate right/wrong (wrong words median conf 0.93); label noise → no gain |
| 4 | **Hard-negative annotation** | pick hardest clips by v5-vs-Whisper disagreement for scholars to label | **adopted for the annotation drive** |
| 5 | **Multi-scholar portal** | concurrency-safe claim-queue annotation web tool | **shipped; used to collect clean labels** |
| 6 | SSL pretraining | wav2vec2-style contrastive on 38 h of Sanskrit śāstric audio; freeze frontend+lower layers | ✗ SSL-init finetune (v9b) < clean-label finetune (v9a) |
| 7 | Semi-supervised pseudo-labels | v5/v9 teacher → confidence-gated pseudo-labels | ✗ confirmed to hurt (3 independent tests) |
| 8 | v9 finetunes (v9a / v9-v2 / v9-v3) | add 846 clean scholar labels (47 speakers) + gold, honest held-out split | ✗ CER flat vs v5; gold **WER worse** (segmentation drift). First "−28%" was a favorable-split artifact |
| 9 | Rule-based sandhi segmenter | DP split of merged tokens to fix WER | ✗ over-shatters (Sanskrit tiles into everything); WER worse |
| 10 | ByT5-Sanskrit post-corrector | byte-level seq2seq to fix ASR output | ✗ over-corrects rare terms (prose CER 7→22); segmentation concept helped chant WER 46→42 only |
| 11 | Blank-penalty / r-recall sweep | penalize CTC blank to recover dropped onsets/repha | partial; diagnostic — CTC blank bias explains onset/short-phone deletions |
| 12 | Onset & CER-distribution analysis | per-position and per-clip error structure | diagnostic — first 2–3 words dropped (encoder ramp-up); e-PG tail is speaker-driven & bimodal |
| 13 | Baselines | Whisper-sa, wav2vec2 finetunes | v5 (IndicConformer-CTC) remained the best on chant/prose |
| 14 | **GOP forced alignment** | Goodness-of-Pronunciation for chant scoring | validated (AUC 0.97–0.99) but too false-positive-prone for a tutor → superseded |
| 15 | **Vāgbodhinī chant tool** | exploit the *known reference* text: verify, don't transcribe | **shipped** (see §6) |
| 16 | **Consented data flywheel** | collect real user audio + labels from the live tools, tiered by trust | **shipped; grew to ~13.1k clips / ~1,500 users / 11 scripts** |
| 17 | v10 flywheel retrain (wholesale) | finetune v5-recipe + flywheel from base, 20 ep | in-the-wild CER 24→16 (big), but **gold regressed** 3.6→6.1 (distribution shift) |
| 18 | v11 flywheel retrain (domain-balanced) | continue *from v5*, low LR, few epochs, flywheel upsampled | first balanced retrain; superseded by v12/v13b as the flywheel grew |
| 19 | **v12 / v13 / v13b retrains** | same balanced recipe on the grown flywheel; validated on a 327-clip leakage-free held-out | **shipped as v13b-ep3** — in-the-wild CER **4.36** (beats v5 7.70 and v12-ep9 5.19); studio/chant/prose flat |
| 20 | Review-tier rescue audit | re-decode quarantined `review` clips with the current model; promote only perfect re-matches | **only 4.3% recoverable** → auto-grader validated: the pile is genuine reader deviations, not model error |

---

## 5. Key findings

1. **Clean human labels are the only training lever that helps at all — and even they don't beat v5
   honestly.** With a properly held-out eval, adding 846 scholar labels left CER flat and hurt gold
   WER (the annotators' spacing conventions drifted the model's word segmentation).
2. **The acoustic model is saturated *on studio audio* (~6% CER).** SSL, pseudo-labels, and more
   clean-studio finetuning did not move it — *but this turned out to be domain-specific* (see §5b:
   on real user audio it was far from saturated).
3. **Inference-side text fixes (rule segmenter, ByT5) don't transfer** — Sanskrit's productive
   sandhi makes naive segmentation/correction over-fire.
4. **The real win is task reformulation.** For chant practice we *know the target text*, so the
   problem is **verification, not recognition** — which sidesteps the ~6% CER ceiling entirely.
5. **CTC is spiky:** a correctly-heard syllable fires at one frame with blank ("continuation")
   around it. Any per-frame confidence measure must treat blank as neutral, not as evidence against
   the target — a subtlety that caused (and, once understood, fixed) a class of false negatives.

---

## 5b. The data flywheel — the key result

Once the live tools were public, every consented recording was logged with its label, **tiered by
trustworthiness** (`pass` ≥90% match · `override`/`corrected` = human-verified · `review` ·
`low`/`unclear` = archived). It has since grown to **~13,100 clips from ~1,500 users across 11
scripts** (a third non-Devanāgarī), including **~420 human corrections**.

We hold out flywheel data **by session** (no speaker leakage) as an in-the-wild eval set — the *real
deployment distribution* — and retrain periodically.

**v5 vs the deployed v13b-ep3 (CER / WER):**

| eval set | v5 (studio-only) | **v13b-ep3 (current, shipped)** |
|---|---|---|
| gold — studio lecture&nbsp;† | 3.61 / 13.0 | 4.40 / 20.2 |
| Bhāgavata chant | 6.00 / 46.4 | 5.99 / 46.3 |
| Vedānta prose | 7.27 / 30.8 | 7.23 / 30.4 |
| **in-the-wild (327-clip leakage-free held-out)** | **7.70 / 45.4** | **4.36 / 30.4** |

† The `gold` studio set was transcribed by correcting v5's *own* drafts, so v5 is flattered there —
it is not a fair cross-model set. The unanchored comparisons are chant/prose (independent
forced-align references) and the in-the-wild held-out.

**Findings:**
1. **v5 had large in-the-wild headroom.** On real user audio (phone mics, diverse speakers, rooms) it
   ran at 7.7% CER / 45% WER — far from its ~6% studio "ceiling". The saturation was studio-only.
2. **Consented flywheel data nearly halved that** — in-the-wild CER **7.70 → 4.36 (−43%)**, WER 45 →
   30 — with **chant and prose unchanged** (6.00→5.99, 7.27→7.23). Real robustness gained at no
   studio cost.
3. **Recipe matters, and small held-outs mislead.** A *wholesale* retrain (v10, from base, 20 ep) let
   flywheel clips over-pull the model and regressed studio audio; a **domain-balanced** recipe
   (continue from v5, low LR, ~3 epochs, flywheel upsampled) keeps the in-the-wild gain and holds
   studio. An early 48-clip held-out suggested ~24% in-the-wild CER; a larger **327-clip
   leakage-free** held-out showed that sample was unusually hard — the real figure is ~5–8%.
   Proper-sized, leakage-free evaluation is now standard.
4. **The auto-grader is trustworthy.** Re-decoding every quarantined `review` clip with the current
   model recovered only **4.3%** as model error — the other ~96% are genuine reader deviations, so
   the tiering hides no reservoir of usable data. The next lever is **new/diverse** data, not more of
   the same. Scripts: `scripts/v11_prep.py`, `scripts/v13b_prep.py`, `scripts/rescue.py`,
   `scripts/flywheel_clean.py`, `scripts/harvest_flywheel.py`.

**The loop, closed and self-improving:** better model → better live scoring → cleaner tiered data →
a better next model. Both tools feed one shared core ASR (`ft_ctc_current`), so they always update
together.

---

## 6. Vāgbodhinī — the chant-practice tool

`vagbodhini/` — a standalone web app.

- **Any-script input** (Devanāgarī/Kannada/Telugu/Tamil/Malayalam/Bengali/Grantha + Roman
  IAST/HK/ITRANS/SLP1) → auto-detected, confirmed by a round-trip echo, UI rendered in the user's
  script. Devanāgarī is the internal canonical form.
- **Metre-aware (vṛtta) pāda splitting:** identify the metre by matching pāda 1 to a known
  laghu/guru signature (tolerant of real metrical liberties), then split into **pāda (¼) / ardha
  (½) / full** learning units. Each is rendered *separately* by the **Vāgdhenu** metre-conditioned
  TTS (not stitched).
- **Feedback = akshara-level text comparison** of the ASR decode vs the reference, made robust by
  **decode-consensus + abstain**: only flag a syllable red when several decodes agree; when the
  model is unsure, say *"unclear"* rather than falsely accuse. **Be-strict / Be-liberal** modes let
  the user choose the precision/recall point.
- **Prose mode too:** a Śloka/Prose toggle — prose is split by phrase and recited (gadya voice),
  extending the tool to sūtra/bhāṣya/śāstra prose, not just chant.
- **Robust UX for an imperfect ASR:** decode-consensus + *abstain*, playback of your chant vs the
  reference, per-syllable **tap-to-correct** override, colour-blind-safe cues, any-script display.
- **Consented data flywheel:** every attempt is logged (audio + label) and tiered; ≥90% matches +
  human corrections become training pairs. **Su-śrotā** (the dictation tool) contributes a second
  stream — *corrections-only* (audio, human-corrected transcript) = gold open-vocabulary labels.
  Both feed the shared core ASR.

See `docs/` for the full system spec (services, ports, endpoints, data format).

---

## 7. Model weights

The Sanskrit ASR weights (IndicConformer-CTC) are on Hugging Face:

> **[`prathoshap/sushrota-sanskrit-asr`](https://huggingface.co/prathoshap/sushrota-sanskrit-asr)**
> — `sushrota_sanskrit_asr_v13b.nemo` is the **deployed** model (v13b-ep3: v5 continued on consented
> user data, §5b); `sushrota_sanskrit_asr_v5.nemo` is the studio-only base. *(See the model card for
> usage.)*

The **training data** (17.4 h / 6,438 utterances + a 327-clip in-the-wild benchmark) is released as a
dataset:

> **[`prathoshap/sushrota-sanskrit-asr-data`](https://huggingface.co/datasets/prathoshap/sushrota-sanskrit-asr-data)**

The metre-aware TTS for reference chants, **Vāgdhenu**, is separately at
[`prathoshap/vagdhenu`](https://huggingface.co/prathoshap/vagdhenu).

### In-Browser & ONNX Exports (Serverless ASR)

The Conformer-CTC Sanskrit model has also been exported to **ONNX** and **INT8** (~178 MB) for zero-dependency local execution and 100% serverless in-browser speech recognition:

> **[`gnumanth/sushrota-sanskrit-asr-onnx`](https://huggingface.co/gnumanth/sushrota-sanskrit-asr-onnx)**
> — Contains `sushrota_sanskrit_ctc_int8.onnx`, `preprocessor.onnx`, and `sanskrit_vocab.json`.
>
> - **In-browser web app:** [`browser_asr.html`](browser_asr.html) (runs via `onnxruntime-web` with WASM SIMD, ~250ms latency, zero backend servers).
> - **Export & quantization script:** [`scripts/export_sushrota_onnx.py`](scripts/export_sushrota_onnx.py).

---

## 8. Repository layout

```
vagbodhini/      the chant-practice app (FastAPI backend + single-file UI + TTS microservice)
scripts/         all experiment / training / evaluation / analysis scripts (§4)
docs/            system documentation
README.md        this report
```

Notes:
- Corpora, audio, and model checkpoints are **not** in the repo (weights → Hugging Face).
- Scripts assume the training/eval data layout on the lab server and are provided for
  documentation and reproducibility of method, not turnkey execution.

---

## 9. Citation

If you use this work, please cite:

> Prathosh A P, *Su-śrotā: Scholar-grade Sanskrit ASR and metre-aware chant practice*, Indian
> Institute of Science, Bengaluru, 2026.

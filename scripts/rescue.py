"""Auto-rescue the flywheel `review` tier: re-decode each clip with the deployed model
(v13b-ep3) using the SAME sa-slice CTC decode + akshara matcher as production. Promote a clip
to gold when the improved model now scores it n_red==0 against the reference — i.e. the old
partial-match was model error, and the user demonstrably read the reference correctly."""
import os, re, json, unicodedata, collections
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as nemo_asr
from indic_transliteration import sanscript

ROOT="/home/ece/BigDisk/Prathosh/ASR"; FLYW=f"{ROOT}/data/practice_flywheel"
OFF,V,BL=4096,256,5632; DEV="cuda:0" if torch.cuda.is_available() else "cpu"
MODEL=f"{ROOT}/exp/ft_ctc_v13b/ft_ctc_ep3.nemo"; MAX_WAV=30*16000

KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]'); NASAL=re.compile(r'म्(?=\s|[।॥]|$)')
def dedup_marks(s):
    out=[]
    for ch in s:
        if out and ch==out[-1] and unicodedata.category(ch) in ('Mn','Mc'): continue
        out.append(ch)
    return ''.join(out)
def canon(s): return NASAL.sub('ं', dedup_marks(s))
def norm(s):
    s=unicodedata.normalize('NFC',s); s=canon(s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
def lse(x,ax): mx=x.max(ax,keepdims=True); return mx+np.log(np.exp(x-mx).sum(ax,keepdims=True))
def is_base(ch):
    o=ord(ch); return (0x0905<=o<=0x0939) or (0x0958<=o<=0x0961) or (0x0972<=o<=0x097F)

print("[boot] loading v13b-ep3 on",DEV,flush=True)
M=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL,map_location=DEV).eval()
SUB=M.tokenizer.tokenizers_dict["sa"]; LAB=json.load(open(f"{ROOT}/data/eval_logits/labels.json"))
print("[boot] ready",flush=True)

def posteriors(wav):
    if len(wav)>MAX_WAV: wav=wav[:MAX_WAV]
    sig=torch.tensor(wav).unsqueeze(0).to(DEV); sl=torch.tensor([len(wav)]).to(DEV)
    with torch.no_grad():
        enc,_=M.forward(input_signal=sig,input_signal_length=sl)
        lp=M.ctc_decoder(encoder_output=enc)[0].cpu().numpy()
    cols=[BL]+list(range(OFF,OFF+V)); P=lp[:,cols]
    del sig,sl,enc
    if DEV!="cpu": torch.cuda.empty_cache()
    return P-lse(P,1)
def greedy(P):
    ids=P.argmax(1); o=[]; prev=-1
    for i in ids:
        i=int(i)
        if i!=prev and i!=0: o.append(LAB[i-1])
        prev=i
    return canon(''.join(o).replace('▁',' ')).strip()
def _cluster(text):
    t=norm(text)
    if not t: return []
    chars=[]
    for j,s in enumerate(SUB.text_to_tokens(t)):
        for ch in s: chars.append((' ',-1) if ch=='▁' else (ch,j))
    out=[];cur="";toks=[];prev_vir=False
    def flush(we):
        nonlocal cur,toks
        if cur: out.append({"text":cur})
        cur="";toks=[]
    for ch,tj in chars:
        if ch==' ': flush(True); prev_vir=False; continue
        if is_base(ch) and cur and not prev_vir: flush(False); cur=ch; toks=[]
        else: cur+=ch
        prev_vir=(ord(ch)==0x094D)
    flush(True)
    merged=[]
    for a in out:
        if merged and a["text"].endswith("्"): merged[-1]["text"]+=a["text"]
        else: merged.append(a)
    return merged
def align(ref,hyp):
    R=[a["text"] for a in ref]; H=[a["text"] for a in hyp]; n,m=len(R),len(H)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            c=0 if R[i-1]==H[j-1] else 1
            dp[i][j]=min(dp[i-1][j-1]+c,dp[i-1][j]+1,dp[i][j-1]+1)
    i,j=n,m; nsub=ndel=nmatch=0
    while i>0 or j>0:
        if i>0 and j>0 and dp[i][j]==dp[i-1][j-1]+(0 if R[i-1]==H[j-1] else 1):
            if R[i-1]==H[j-1]: nmatch+=1
            else: nsub+=1
            i-=1;j-=1
        elif i>0 and dp[i][j]==dp[i-1][j]+1: ndel+=1; i-=1
        else: j-=1
    nins=m-nmatch-nsub
    return nsub,ndel,nins,len(R)

# review clips
rev=[]
for l in open(f"{FLYW}/log.jsonl"):
    try: r=json.loads(l)
    except: continue
    if r.get("tier")!="review": continue
    p=f"{FLYW}/audio/{r['id']}.wav"; t=(r.get("text") or "").strip(); d=r.get("dur") or 0
    if t and os.path.exists(p) and 0.4<=d<=30:
        rev.append({"id":r["id"],"p":p,"text":t,"dur":d,"orig_red":r.get("n_red") or 0})
print(f"[rescue] {len(rev)} review clips to re-decode",flush=True)

promoted=[]; exact=[]; bucket=collections.Counter(); bh=collections.Counter()
for k,c in enumerate(rev):
    try:
        wav,sr=sf.read(c["p"],dtype="float32")
        if wav.ndim>1: wav=wav.mean(1)
        hyp=greedy(posteriors(wav))
    except Exception as e:
        continue
    refA=akshara=_cluster(c["text"]); hypA=_cluster(hyp)
    nsub,ndel,nins,nref=align(refA,hypA)
    nred_new=nsub+ndel
    ob=c["orig_red"] if c["orig_red"]<=2 else "3+"
    if nred_new==0:
        promoted.append({"audio_filepath":c["p"],"text":c["text"],"duration":round(c["dur"],2),"lang":"sa"})
        bucket[ob]+=1; bh[ob]+=c["dur"]
        if nins==0: exact.append(c["id"])
    if (k+1)%500==0: print(f"  {k+1}/{len(rev)} … promoted so far {len(promoted)}",flush=True)

os.makedirs(f"{ROOT}/data/rescue",exist_ok=True)
with open(f"{ROOT}/data/rescue/rescued.jsonl","w") as f:
    for r in promoted: f.write(json.dumps(r,ensure_ascii=False)+"\n")
ph=sum(r["duration"] for r in promoted)/3600
print("="*60)
print(f"RESCUED (v13b-ep3 now scores n_red==0): {len(promoted)} clips / {round(ph,2)} h")
print(f"  of which EXACT (also no insertions): {len(exact)} clips")
print(f"  from {len(rev)} review clips  ->  rescue rate {100*len(promoted)/len(rev):.1f}%")
print("  promoted by ORIGINAL n_red bucket:")
for k in [0,1,2,"3+"]:
    if k in bucket: print(f"    orig n_red={k}: {bucket[k]} clips / {bh[k]/3600:.2f} h")
print(f"WROTE data/rescue/rescued.jsonl")

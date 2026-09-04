#!/usr/bin/env python3
"""Export Sushrota Sanskrit ASR (Conformer-CTC, Sanskrit slice) to ONNX and INT8.

Produces:
- models/sushrota_sanskrit_ctc.onnx (FP32)
- models/sushrota_sanskrit_ctc_int8.onnx (INT8 quantized, ~115 MB, browser-ready)
- models/sanskrit_vocab.json (tokens mapping for decoding)
"""

import os, json, time
import torch
import nemo.collections.asr as nemo_asr
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

os.makedirs("models", exist_ok=True)

print("[1/5] Loading NeMo checkpoint...")
m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    "models/sushrota_sanskrit_asr_v13b.nemo",
    map_location="cpu",
    strict=False
).eval()

print("[2/5] Extracting Sanskrit slice and building CTC head...")
# Original Conv1d: (5633, 512, 1)
# Column 5632 is blank; 4096..4351 are Sanskrit BPE tokens
orig_conv = m.ctc_decoder.decoder_layers[0]
cols = [5632] + list(range(4096, 4096 + 256))

sa_conv = torch.nn.Conv1d(512, 257, kernel_size=1)
sa_conv.weight.data.copy_(orig_conv.weight.data[cols])
sa_conv.bias.data.copy_(orig_conv.bias.data[cols])

# Save Sanskrit vocabulary tokens
sa_tokenizer = m.tokenizer.tokenizers_dict["sa"]
tokens = ["<blank>"]
for i in range(256):
    tok = sa_tokenizer.ids_to_tokens([i])[0]
    tokens.append(tok)

vocab_path = "models/sanskrit_vocab.json"
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(tokens, f, ensure_ascii=False, indent=2)
print(f"  Saved {len(tokens)} tokens to {vocab_path}")

class SanskritASR(torch.nn.Module):
    def __init__(self, encoder, ctc):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc

    def forward(self, audio_signal, length):
        # audio_signal: [batch, 80, time]
        # length: [batch]
        enc, enc_len = self.encoder(audio_signal=audio_signal, length=length)
        logits = self.ctc(enc)  # [batch, 257, time_out]
        return logits.transpose(1, 2)  # [batch, time_out, 257]

model = SanskritASR(m.encoder, sa_conv).eval()

# Test with dummy input
dummy_audio = torch.randn(1, 80, 200, dtype=torch.float32)
dummy_len = torch.tensor([200], dtype=torch.int64)

print("[3/5] Exporting Sanskrit Conformer-CTC to ONNX...")
onnx_fp32_path = "models/sushrota_sanskrit_ctc.onnx"
torch.onnx.export(
    model,
    (dummy_audio, dummy_len),
    onnx_fp32_path,
    input_names=["audio_signal", "length"],
    output_names=["logits"],
    dynamic_axes={
        "audio_signal": {0: "batch", 2: "time"},
        "length": {0: "batch"},
        "logits": {0: "batch", 1: "time_out"}
    },
    opset_version=17,
    do_constant_folding=True
)
fp32_size_mb = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
print(f"  Exported FP32 ONNX: {onnx_fp32_path} ({fp32_size_mb:.1f} MB)")

print("[4/5] Verifying ONNX model with onnxruntime...")
sess = ort.InferenceSession(onnx_fp32_path, providers=["CPUExecutionProvider"])
ort_inputs = {
    "audio_signal": dummy_audio.numpy(),
    "length": dummy_len.numpy()
}
ort_out = sess.run(None, ort_inputs)[0]
with torch.no_grad():
    torch_out = model(dummy_audio, dummy_len).numpy()

max_diff = (abs(ort_out - torch_out)).max()
print(f"  Max numerical difference (ONNX vs PyTorch): {max_diff:.6f}")
assert max_diff < 1e-4, f"Difference too high: {max_diff}"

print("[5/5] Quantizing to INT8 for browser and local edge deployment...")
onnx_int8_path = "models/sushrota_sanskrit_ctc_int8.onnx"
quantize_dynamic(
    model_input=onnx_fp32_path,
    model_output=onnx_int8_path,
    op_types_to_quantize=["MatMul"],
    weight_type=QuantType.QInt8
)
int8_size_mb = os.path.getsize(onnx_int8_path) / (1024 * 1024)
print(f"  Exported INT8 ONNX: {onnx_int8_path} ({int8_size_mb:.1f} MB)")
print("\nExport & Quantization COMPLETE!")

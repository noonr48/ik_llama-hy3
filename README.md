# ik_llama-hy3

Fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) with support for **Tencent Hy3** (hy_v3) — a 295B parameter Mixture-of-Experts model with Multi-Token Prediction (MTP).

> **Download the quantized model (IQ4_NL, 158 GB):** [huggingface.co/jackasda211233/Hy3-IQ4_NL-GGUF](https://huggingface.co/jackasda211233/Hy3-IQ4_NL-GGUF)

## What is Hy3?

Tencent Hy3 is a 295B MoE model (8 active experts out of 192 per layer) with:
- Shared expert architecture (sigmoid gating)
- Built-in MTP layer (NextN at block 80) for speculative decoding
- Custom thinking tokens (`<think:opensource></think:opensource>`)
- Custom EOS token (`<｜hy_eos:opensource｜>`)
- 262K training context length

## Features

### Tested and Working

| Feature | Status | Notes |
|---------|--------|-------|
| Model loading (IQ4_NL quantization, ~158GB) | ✅ | Streaming conversion from HF to GGUF verified |
| Inference (token generation) | ✅ | Clean output, correct token stream |
| Chat template (custom Jinja2) | ✅ | Thinking tokens parsed, reasoning extracted |
| EOS token handling | ✅ | Fixed leak where EOS text appeared in output |
| Reasoning separation | ✅ | `reasoning_format=deepseek` splits thinking from answer |
| MTP speculative decoding | ✅ | 69-80% draft acceptance rate, ~54% speedup on large prompts |
| Graph split across multiple GPUs | ✅ | Verified across multi-GPU setups |
| Full context range (32K to 200K+) | ✅ | Supports the model's full training context length |
| `--flash-attn on` | ✅ | Required for correct attention with this architecture |
| `--override-kv tokenizer.ggml.eos_token_id` | ✅ | GGUF metadata has wrong eos_id (3), correct is 120025 |
| NVIDIA CUDA (multiple architectures) | ✅ | Build with `-DCMAKE_CUDA_ARCHITECTURES` matching your GPUs |

### Not Tested

| Feature | Notes |
|---------|-------|
| Other quantization formats (Q8_0, Q4_K_M, etc.) | Only IQ4_NL tested. Should work but unverified. |
| Single-GPU inference | Requires ~158GB+ VRAM. Not tested. |
| Non-CUDA backends (CPU-only, Metal, Vulkan) | Not tested. |
| Multi-user concurrent requests | Single-slot only tested. |
| Vision/multimodal inputs | Hy3 is text-only; not applicable. |
| Function calling / tool use | Not tested through the server API. |

### Not Implemented

| Feature | Notes |
|---------|-------|
| DeepSeek-V3 style external draft model | ik_llama's MTP uses the built-in NextN layer, not an external draft model. |
| GPU argmax optimization for MTP | Available in reference implementations (Qwen RYS fork) but not ported. Could improve MTP draft generation speed. |
| KV cache cleanup optimization for MTP | Available in reference implementations but not ported. |
| GGUF metadata fix (eos_token_id) | The converter writes eos_token_id=3 (wrong). Must use `--override-kv tokenizer.ggml.eos_token_id=int:120025` at runtime. Should be fixed in the converter. |
| Proper BPE pre-tokenizer hash registration | Hy3 uses a custom BPE pre-tokenizer not in the hash table. Currently handled via override. |

## Branches

- **`hy3-support`** — All Hy3 changes (default branch for this fork)
- **`main`** — Tracks upstream ik_llama.cpp main

## Commits

This fork adds these commits on top of upstream ik_llama.cpp:

1. **`hy_v3 architecture support: sigmoid MoE + shared expert + MTP`** — Core architecture: enum, hparams, graph builder dispatch, tensor loading, tensor name table
2. **`Fix Hy3 loader: tokenizer pre-tokenizer, tensor creation, rope type, tensor name table`** — 7 loader bug fixes
3. **`Enable MTP speculative decoding for hy_v3 + fix eos token leak`** — MTP architecture gate fix + chat handler with thinking token parsing and EOS leak fix

## Files Changed (vs upstream)

```
convert_hf_to_gguf.py          — Hy3 model detection + GGUF conversion
gguf-py/gguf/constants.py      — Hy3 tensor name constants
src/llama-arch.cpp             — LLM_ARCH_HY_V3 enum + arch registration
src/llama-arch.h               — LLM_ARCH_HY_V3 enum declaration
src/llama-build-context.cpp    — HY_V3 → build_glm4_moe() dispatch
src/llama-hparams.cpp          — Hy3 hparams (nextn_predict_layers, leading_dense_block_count)
src/llama-load-tensors.cpp     — create_hy3_tensors() with nextn/MTP tensor loading
src/llama-model.cpp            — Hy3 tensor name mapping
src/llama-model.h              — (no structural changes, uses existing structs)
src/llama-vocab.cpp            — Hy3 pre-tokenizer detection
src/llama.cpp                  — MTP architecture gate + rope type
common/chat.cpp                — Hy3 chat template handler + EOS fix
```

## Usage

### Build

```bash
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
cmake --build build --target llama-server -j$(nproc)
```

Set `CMAKE_CUDA_ARCHITECTURES` to match your GPU compute capabilities. For example, for RTX 3090 (sm_86) and RTX 5060 Ti (sm_120):
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;120" -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF ...
cmake --build build --target llama-server -j$(nproc)
```
Check your GPU's compute capability with `nvidia-smi --query-gpu=compute_cap --format=csv` and use those numbers.

### Serve (KV on GPU, with MTP)

When you have enough VRAM to hold both the model and KV cache on GPU:

```bash
./build/bin/llama-server \
  --model Hy3-IQ4_NL.gguf \
  --host 0.0.0.0 --port 9999 \
  --n-gpu-layers 999 \
  -sm graph \
  --override-kv tokenizer.ggml.eos_token_id=int:120025 \
  --ctx-size 32768 --batch-size 512 --ubatch-size 512 \
  --flash-attn on --cache-type-k f16 --cache-type-v f16 \
  --jinja --chat-template-file models/templates/Hy3.jinja \
  --reasoning-format deepseek --reasoning on \
  --spec-type mtp:n_max=1,p_min=0.0
```

Adjust `--ctx-size` based on how much VRAM you have for KV cache. Tune `--ubatch-size` based on your GPU memory — lower values reduce VRAM usage during prompt processing.

### Serve (large context with CPU KV offload, with MTP)

For contexts that exceed GPU VRAM, use `--no-kv-offload` to spill KV cache to system RAM:

```bash
./build/bin/llama-server \
  --model Hy3-IQ4_NL.gguf \
  --host 0.0.0.0 --port 9999 \
  --n-gpu-layers 999 \
  --no-kv-offload \
  --override-kv tokenizer.ggml.eos_token_id=int:120025 \
  --ctx-size 200000 --batch-size 512 --ubatch-size 512 \
  --flash-attn on --cache-type-k f16 --cache-type-v f16 \
  --jinja --chat-template-file models/templates/Hy3.jinja \
  --reasoning-format deepseek --reasoning on \
  --spec-type mtp:n_max=1,p_min=0.0
```

### MTP Speculative Decoding

Hy3 includes a built-in MTP layer (block 80, NextN architecture). Enable it with:

```
--spec-type mtp:n_max=1,p_min=0.0
```

- `n_max=1` is recommended (1 draft token per step, 69-80% acceptance rate)
- The old `-mtp` flag is deprecated; use `--spec-type`

## Credits

- Original ik_llama.cpp by [Ikawrakow](https://github.com/ikawrakow)
- Hy3 model by Tencent
- MTP infrastructure in ik_llama.cpp (originally for GLM_DSA / Qwen3.5-MoE)

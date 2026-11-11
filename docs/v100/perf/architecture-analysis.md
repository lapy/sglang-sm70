# Qwen3.8-Flash-Next-NVFP4 on 4×V100-SXM2: architecture & hardware analysis for the decode-tuning campaign

> **Provenance.** This report covers **sglang-sm70**, the SM70 (V100) fork of SGLang carried as ten slices on upstream `2c5b680aa6` (`main`).
> Measurements attributed to "this fork" were taken on 4× V100-SXM2-32GB at a 150 W cap with `bench*.py` (warm, clocks ≥1500 MHz, medians).
> Measurements attributed to "predecessor" are quoted from the records of the predecessor fork (sglang-sxm2) and may come from different hardware or power limits.
> Credits and licenses are in [`NOTICE`](../../../NOTICE).

Target: `RadixArk/Qwen3.8-Flash-Next-NVFP4` (`qwen4_exp`, ModelOpt NVFP4), SGLang, TP4 on 4×V100-SXM2-32GB (sm70), full NV2 clique.
Prepared 2026-11-11. Local repos read: `sglang-sm70@sm70/stack`, `sglang-sxm2@qwen38-kv-qsa-port-20261110`. Checkpoint config read from the box (`/workspace/models/Qwen3.8-Flash-Next-NVFP4`).
Everything below is **read-only analysis**; no repo was modified.

---

## 0. Status correction before anything else (read this first)

The brief states *74 tok/s (sm70 stack) vs ~80 tok/s (sxm2)*. **That premise is stale on the live box, and it changes what "beat it" means.** Measured today:

| measurement | value | how |
|---|---|---|
| `bench-baseline.json` on the box, 06:10 | **107.63 tok/s** decode (samples 105.7 / 107.6 / 108.6), TTFT 432 ms | `bench.py` two-point method, 32→160 tokens, contractive prompt |
| my run at 06:13 (parent's sweep booting concurrently) | 69.5 tok/s two-point; 60.2 tok/s incl. prefill | same method, 128/1.843 s |
| in-server `gen throughput` field | bimodal **62 ↔ 118 tok/s** step-to-step | `serve-final-check.log`, `decode_log_interval=40` |
| spec accept length | **2.81 / 4** (rate 0.703); histogram over 57 verify steps `[0:13, 1:9, 2:10, 3:25]` | live `/generate` meta_info |

Two conclusions:

1. **The current tree already does ~108 tok/s clean**, i.e. the 80 tok/s target is surpassed; the 74/80 numbers in the brief are from an older build/config. The bimodality (62 vs 118) and my contended 69.5 mean **the box's noise floor is larger than the entire 6-tok/s gap the brief is chasing**. Any A/B that does not pin the measurement protocol (same prompt, two-point, idle box, ≥3 reps, median) will report noise as progress.
2. **The two forks are not the same lineage, and the gap is not a subtle kernel difference.** `sglang-sm70` contains **zero** `SGLANG_SM70_*` env knobs; the predecessor fork (sglang-sxm2) contains **26** of them (fused hyper-connection GEMV, MTP fusions, NVLink push all-reduce, batched MoE, draft top-1, QSA combine, …). The running server is neither local clone: it is `/workspace/sxm2-verify/fork` (`qwen4_exp.py` = 113 471 B; local sm70 = 110 787 B, local sxm2 = 100 768 B), i.e. a **merged tree** built from the sm70 stack plus the sxm2 feature set. See §5.

Everything else in this document is therefore framed against **107.6 tok/s ≈ 26.0 ms per MTP iteration**, and I say explicitly where the 74-vs-80 framing would change a conclusion.

### The single most useful number in this document

```
per-iteration (1 verify of 4 tokens + 3 draft forwards), per TP rank:
    weight+state traffic            6 231 MB      -> 7.6 ms at measured 816 GB/s
    compute (fp16 TC, 2.7 GFLOP/GPU)               -> 0.02 ms  (0.1%)
    measured wall time (107.6 tok/s)  26.0 ms      -> achieved BW = 240 GB/s = 29% of peak
    UNACCOUNTED (launch/latency/collective) = 18.4 ms = 71% of the step
```

Decode here is **not bandwidth-bound and not compute-bound. It is dependency-latency-bound**, sitting 3.4× above its own memory floor. That single fact drives every recommendation in §3 and §4.

---

## 1. Model shape, and what actually dominates decode time

### 1.1 Shape (config.json + verified against safetensors headers)

| field | value | note |
|---|---|---|
| `hidden_size` | 2560 | |
| `num_hidden_layers` | 48 | `full_attention_interval: 4` |
| layer mix | **36 GDN linear-attention + 12 QSA full-attention** | pattern `L,L,L,F` ×12 — matches `layer_types` exactly |
| `num_attention_heads` / `head_dim` | 24 / 256 | q width 6144, `attn_output_gate` doubles it → q_proj out **12 288** |
| `num_key_value_heads` | 2 | KV width 512; `partial_rotary_factor 0.25` → 64 rotary dims |
| `num_experts` / `num_experts_per_tok` | 512 / 10 | |
| `moe_intermediate_size` | **640** | **→ 160 per rank at TP4** (this is the "160" in the brief) |
| `shared_expert_intermediate_size` | 640 | +1 always-on expert, bf16, not quantised |
| GDN | `linear_num_key_heads 16`, `linear_value_heads 48`, dims 128, conv 4 | in_proj_qkv 10240, in_proj_z 6144 |
| hyper-connection | `hc_count 4`, `hc_lowrank 320` | 4 residual streams → **10 240-wide** residual; 2 `GatedResidual` (attn+mlp) **per layer** |
| `vocab_size` | 248 320 | `lm_head` **not** quantised, not TP-lm-head-sharded-by-default |
| MTP | `mtp_num_hidden_layers 1`, `hybrid`, `full_attention`, `num_steps 3`, `draft_tokens 4`, `eagle_topk 1` | |
| QSA | `indexer_n_heads 4`, `indexer_head_dim 128`, `indexer_kv_heads 1`, **`indexer_budget 2048`**, **`indexer_compress_ratio 4`** | `block_topk = 2048/4 = 512` compressed blocks = 2048 tokens |
| PLE | `ple_embed_dim 2560`, `ngram_size 3`, `heads_per_ngram 8` → **16 heads × 160 dims**; vocab base 20 M/head; `float8_e4m3fn`; layer index 1 (i.e. `layer_id 0`, `ple_layer_ids [2]` is 1-based) | **51.2 GB table**, one layer only |

Checkpoint ledger (verified from file headers, `du`):

| group | bytes | per rank |
|---|---|---|
| 512 experts × 48 layers, NVFP4 g128 + fp8 scales | 68.0 GB | 17.0 GB |
| PLE n-gram table, fp8, 128 shards × [2 500 012 × 160] | 51.2 GB | **11.92 GiB — pinned host only** |
| everything else (bf16: attn, GDN, HC, embed, lm_head, MTP) | ~15 GB | ~3.7 GB |
| **total** | **134 GB** | 33.5 GB → **19.89 GB on device** after PLE offload |

Per-rank expert cost, which is the number that matters: `gate 160×2560 + up 160×2560 + down 2560×160` in 4-bit + fp8 scales = **0.691 MB per distinct expert touched**.

### 1.2 Traffic and FLOPs per forward (batch 1–3, 4-token MTP verify)

Per rank, per TP4 shard. NVFP4 = 0.5 B/param + fp8 group-16 scales. bf16 = 2 B/param. Assumes standard SGLang TP sharding (expert intermediate /4, GDN /4, LM head vocab /4); the hyper-connection mixers are `nn.Linear` created outside the TP plan and are therefore **replicated**.

| component | MB/step (B=4, D=40 experts) | share | ms @816 GB/s |
|---|---|---|---|
| routed experts (48 layers × 40 distinct × 0.691 MB) | 1 327.1 | 21.3% | 1.63 |
| **lm_head × 4** (1 verify + 3 draft passes, 318 MB each) | 1 271.2 | 20.4% | 1.56 |
| **hyper-connection ×96** (48 layers × 2 stages × [10240→320→10240]) | 1 265.9 | 20.3% | 1.55 |
| GDN, 36 layers (qkv 13.1 + z 7.9 + out 7.9 MB) | 1 054.8 | 16.9% | 1.29 |
| shared expert ×48 | 471.9 | 7.6% | 0.58 |
| QSA attention, 12 layers (q 15.7 + o 7.9 + kv replicated) | 355.8 | 5.7% | 0.44 |
| draft MTP layers ×3, non-head | 223.8 | 3.6% | 0.27 |
| router ×48 | 125.8 | 2.0% | 0.15 |
| PLE layer (key_proj 10240×2560 **replicated** = 52 MB) | 65.5 | 1.1% | 0.08 |
| GDN state read+write (12 v-heads/rank × 128×128 fp32 × 36) | 57.0 | 0.9% | 0.07 |
| KV read, B=4, ctx≈256 | 12.6 | 0.2% | 0.02 |
| **total** | **6 231** | 100% | **7.64** |

Batch sensitivity (routed experts dominate the delta; `D = min(512, B·10)`, dedup assumes 0.6× because the 4 verify tokens are one draft chain):

| case | D | MoE MB | total MB | ms @816 | ms @900 |
|---|---|---|---|---|---|
| B=1 verify | 10 | 331.8 | 4 004 | 4.91 | 4.45 |
| B=2 | 20 | 663.6 | 4 339 | 5.32 | 4.82 |
| B=3 | 30 | 995.3 | 4 674 | 5.73 | 5.19 |
| **B=4 (MTP verify)** | 40 | 1 327.1 | **5 009** | 6.14 | 5.57 |
| B=1 dedup 0.6 | 6 | 199.1 | 3 872 | 4.74 | 4.30 |
| B=4 dedup 0.6 | 24 | 796.3 | 4 478 | 5.49 | 4.98 |

FLOPs (aggregate, then per GPU):

| per token | GFLOP |
|---|---|
| routed experts (10 × 3 × 2560 × 640 × 2) | 0.098 |
| GDN (3 projections) | 0.115 |
| QSA (q/k/v/o) | 0.100 |
| hyper-connection (2 stages) | 0.026 |
| PLE | 0.066 |
| **lm_head (248 320 × 2560 × 2)** | **1.271** |
| total | **1.68** |

Iteration = 4 verify rows + 3 draft rows ≈ **10.8 GFLOP → 2.7 GFLOP/GPU → 0.022 ms at 125 TFLOPS fp16-TC** (0.4 ms even at 5% MFU). **Compute is 0.1–1.5% of the step.** The LM head alone is 76% of all FLOPs — and it is still not the limiter, because it is a GEMV (M≤4), not a GEMM.

Arithmetic intensity check, because it decides several ideas below:
- routed expert at M=1/token: `2·3·2560·160 / 0.691 MB = 3.6 FLOP/byte`. V100 ridge point = 125 TFLOPS / 816 GB/s = **153 FLOP/byte**. → **42× below the ridge**: each expert GEMM is a pure memory/launch-limited GEMV. Even at M=40 tokens/expert it only reaches ~145 FLOP/byte.
- the LM head at M=4: `2·4·248320·2560 / 318 MB = 16 FLOP/byte` → also 10× below the ridge. Same story.

### 1.3 Ranked answer: what dominates

1. **Dependency latency between ~1 100–1 250 small kernels per iteration (18.4 ms, 71%)** — not memory, not FLOPs, not launch *count* per se but the serialized tail of dependent launches plus 96 collective waits. Evidence: 6 231 MB / 26.0 ms = **240 GB/s effective = 29% of peak**; a step contains 48 layers × ~20 kernels (3–4 projections + 3 grouped norms + 2 HC mixes + 2 HC combines + indexer/top-k/gather + MoE + router + shared + AR waits) + 3 draft forwards × ~30.
2. **Weight traffic (7.6 ms, 29%)**, of which **62% is three components: routed experts, the LM head, and the hyper-connection mixers.**
3. **Collectives: 96 TP all-reduces per step** (2/layer: `o_proj` and `down_proj` row-parallel outputs, plus the shared-expert path). On this box they run through **NCCL, not SGLang's custom all-reduce** — the server log says *"CustomAllReduceV2 is disabled because it's not supported on more than two PCIe-only GPUs"* and *"CustomAllreduce is disabled …"*, even though `nvidia-smi topo -m` shows a full **NV2 clique** (6 links × 25.781 GB/s = 154.7 GB/s per direction, P2P read OK on every pair). Estimate at 20–35 µs per NCCL op = **1.9–3.4 ms**, plus it sits in the dependency chain.
4. **Compute: negligible** (0.02–0.4 ms).
5. **KV reads: negligible at this context** (12.6 MB; ctx ≈ 180 prompt + 128 out). QSA prunes nothing — see §3(e).

**Speculation, clearly labelled:** the decomposition of the 18.4 ms between "kernel tail" and "collective" is a model, not a profile. The experiment that settles it is in §4 (nsys with `--enable-layerwise-nvtx-marker`, which exists as a flag and is currently `False`).

---

## 2. V100 (sm70) limits, and what each one does to this model

| limit | V100 fact | consequence for this model |
|---|---|---|
| SMs | **80** SM, 4 warp schedulers each, 64 FP32 lanes, 2nd-gen TC | 80 SMs × 4 = 320 warp slots for latency hiding. A dependent chain of 1 200 kernels cannot use them; there is nothing to overlap. **Latency-bound.** |
| HBM2 | 900 GB/s peak, 4096-bit, **~816 GB/s measured** on this class of box (91%) | Our 240 GB/s = 29%. Bandwidth is available; we are not using it. |
| HBM latency | ~500–700 ns class (unhidden) | At 80 SMs, one dependent GEMV tile round trip ≈ 1 µs. 48 layers × several such chains ⇒ the 18 ms. |
| FP16 tensor core | 125 TFLOPS (m8n8k4 / wmma 16×16×16 HMMA), FP32 accumulate | Needed only for prefill. At M≤4 we are 42× below the ridge point → **tensor cores are irrelevant to decode** except for the LM head's batch rows. |
| **no bf16** | bf16 arrives with sm80 | The stack runs `--dtype float16` (confirmed in the live `server_args`), and `SGLANG_SM70_FORCE_FP16` defaults True. Every bf16 weight on the checkpoint is served as fp16 → **2× the bytes of a bf16-native model** for the 15 GB dense group and for `shared_expert`. Also: the MTP layer is *excluded* from NVFP4 by the quantiser (`ignore: ["mtp.*"]`) so it is 16-bit and is re-read 3× per iteration. |
| **no cp.async (LDGSTS)** | sm80+ | Global→shared must go LDG→reg→STS, spending the registers/occupancy that would otherwise hide latency. Directly measured elsewhere as "`num_stages` sweeps do nothing on V100" ([V100 findings](https://raw.githubusercontent.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/refs/heads/main/docs/V100_OPTIMIZATION_FINDINGS.md)). Implication: **do not fund deeper software pipelining; fund fewer dependent steps.** |
| no TMA / `cp.async.bulk` | sm90+ | no bulk async copies, no cluster DSMEM; every gather (QSA top-k KV gather, PLE n-gram gather) is a synchronous dependent load. |
| **no `MUFU.TANH`** | `tanh.approx.f32` is sm75+ | Triton lowers `tanh` to `tanh.approx` only for sm_75+, so on sm70 it emits an exp/div sequence inside GDN. This is why `--linear-attn-decode-backend` must be chosen carefully and why the GDN elementwise math is comparatively expensive. It also blocks cheap sigmoid-in-kernel tricks in GDN gating. |
| NVLink | **NV2, 6 links × 25.781 GB/s per GPU** (154.7 GB/s/direction), full 4-way clique, P2P OK | All-reduce of a 5 KB fp16 vector is **latency**, not bandwidth: ~5–10 µs of wire time, 20–35 µs of NCCL software. 96 of them per step is the second-largest latency item and is *fixable* (§4). |
| Triton on sm70 | **no tensor-core `tl.dot`** (FMA fallback), register spill above `BLOCK_K=64` | Any Triton MoE/GEMM we write must assume FMA + small `BLOCK_K`. Cited measurement: stock `BLOCK_K=128` in the decode branch cost 4–9× ([source](https://raw.githubusercontent.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/refs/heads/main/docs/V100_OPTIMIZATION_FINDINGS.md)). Applies directly to any hand-written expert kernel in §4. |

**Classification per candidate bottleneck:**

| bottleneck | class on sm70 | implication |
|---|---|---|
| routed-expert GEMMs (160 wide, M≤4) | **latency-bound** (3.6 FLOP/byte, tiny tiles) | more parallelism or fewer launches — *not* more bandwidth. Batching experts helps tile utilisation, not bytes. |
| hyper-connection mix (96 GEMV pairs) | **latency-bound + replicated bytes** | fuse into fewer kernels; the bytes are unavoidable but 2 kernels/stage → 1. |
| LM head (318 MB, 4×/iteration) | **bandwidth + launch-bound, batch-invariant** | its cost is per *call*, not per token → amortise by batching, or shrink it (hot vocab). |
| GDN projections | **bandwidth-bound** (28.8 MB/layer, big GEMMs at M≤4 → GEMV) | fusion with norms; fp8 the weights (they are bf16/fp16 today!). |
| QSA attention + indexer | **latency-bound** (KV traffic is 0.2%) | fewer launches; the indexer is pure overhead at 256-token ctx. |
| TP all-reduce ×96 | **latency-bound, in the chain** | custom AR / push / overlap. |
| PLE n-gram gather | **PCIe-latency-bound** (11.92 GiB in pinned host) | keep off the critical path (prefetch already exists); 1 layer only. |

---

## 3. Where the remaining headroom is, ranked by expected tok/s

Working point: 26.0 ms/step, 2.81 tokens/step, 107.6 tok/s. Two independent axes:

- **Axis A — cut the fixed per-step cost** (18.4 ms is not doing useful work).
- **Axis B — raise tokens per step at flat cost** (accept-length / draft-head arithmetic).

| # | lever | mechanism | est. tok/s | conf. | cost |
|---|---|---|---|---|---|
| 1 | **hot-vocab draft head + 8-token verify** | draft head 318 → ~41 MB/pass makes 7 draft passes cheaper than today's 3; accept arithmetic gives +30% tokens | **+25–30%** (→135–140) | med | small change (code exists: `tp_hot_vocab.py`) |
| 2 | **kill the 96 NCCL all-reduces** | enable custom AR v2 / `SGLANG_SM70_AR_PUSH16=1` on the NV2 clique | +6–15% (→115–125) | med-high | config (+ allocator setting) |
| 3 | **single-replay decode graph** (fold draft×3 + verify into one capture; stop the PLE graph break) | removes 3 graph launches + Python between draft steps and verify | +5–15% | med | structural |
| 4 | **draft top-1 without full-vocab all-gather** (`SGLANG_SM70_MTP_LOCAL_TOP1=1`) | 3 full-vocab gathers + 248 k-wide argmax per iteration disappear | +2–5% | high | flag |
| 5 | **batch/fuse the MoE for the 4-token verify** (`SGLANG_SM70_BATCHED_MOE`/`_MOE32`/`MTP_WIDE_MOE`) | 48 → fewer, fuller tiles; D=40 experts × N=160 is 8% of one SM-wave | +3–8% | med | flag (parent already sweeping) |
| 6 | **dense-attention bypass when `ctx ≤ indexer_budget`** | at ctx 256 the 512-block QSA selection selects everything; skip indexer q/k + top-k + gather | +1–4% | med-high | small kernel change |
| 7 | **fp8/4-bit the GDN + shared-expert bf16 weights** | 1 527 MB/step of 16-bit weights are read as GEMVs | +3–6% | med | structural (quantisation) |
| 8 | **prompt-lookup / n-gram second draft source** (`SGLANG_SM70_PROMPT_LOOKUP`) | the model *is* an n-gram monster; a 3-gram lookup draft can raise accept on repetitive text | +0–20% (workload-dependent) | low-med | flag / structural |
| 9 | **`num_steps` retune** | fewer draft head passes per accepted token | +5–10% if accept holds | med | flag |
| 10 | **HC mix batched across layers** | 96 GEMV pairs → 1 batched kernel per step | +3–5% | low-med | structural |

### 3(a) The tiny per-expert intermediate (160): does batching across the 10 routed experts help?

**Quantified: yes, but it is a latency/occupancy fix worth ~3–8%, not a bandwidth fix.** An expert GEMM is `M≤4 × N=160 × K=2560`, intensity **3.6 FLOP/byte vs a 153 FLOP/byte ridge** — 42× memory-bound, and 0.691 MB per expert. The bytes are unavoidable: at D=40 distinct experts the 48-layer MoE totals **1 327 MB = 21.3%** of the step. What batching changes is (i) the number of dependent kernel/tile phases and (ii) tile shape: with `N=160` and `BLOCK_N=64` you get 3 column tiles × 40 K-tiles per expert = 120 tiles, and 40 experts = 4 800 tiles — *enough* to fill 80 SMs, but each tile is a dependent K-loop of only 0.69 MB/40 = 17 KB of weights. So the expert GEMM is **latency-bound per tile, not starved of work**. Batching 10 experts into one grouped kernel converts 10 sequential-ish tiny GEMMs into one wider grid; it does not reduce bytes and cannot beat ~1 bandwidth-limited pass. **Expected 3–8%.**

The bigger structural point in the same direction: **the per-expert weight (0.691 MB) is smaller than the L2 slice any SM will stream**, so at M≤4 *every expert is a fresh cold read* — there is no reuse to exploit and no cache to warm. Adding tokens per expert (bigger verify window, larger batch) is almost free in the MoE sense: going B=1→B=4 raises MoE bytes 4× but raises the *total* step only 25% (4 004 → 5 009 MB).

### 3(b) LM head vs `--tp-lm-head` / sharding

**The LM head is 20.4% of step traffic and 76% of step FLOPs, and its cost is per call, not per token.** It is a vocab-parallel GEMV: each rank reads its 62 112 × 2560 shard = **318 MB per call**, and it is called **4× per iteration** (1 verify + 3 draft steps) = **1 271 MB**. Consequences:

- `--enable-tp-lm-head` (currently **False**) changes *how* the full-vocab logits are assembled (all-to-all vs all-gather) but **does not reduce the 318 MB weight read per call**. Expect ~0 from it alone; it matters only for its interaction with sampling memory.
- The levers that do work: **(i) hot vocab for the draft** (318 → ~41 MB/pass), **(ii) fewer draft calls per accepted token**, **(iii) batching** (the head is read once per call regardless of B, so at B=3 the head cost per output token falls 3×). Today's `--max-running-requests 3` means the single-stream number pays the full head cost alone.
- Under greedy/`eagle_topk 1`, a fused `GEMV + argmax` (never materialising `[B, 248320]`) plus a 2-float all-gather is exactly what `sxm2_draft_top1` implements — **sm70 has no equivalent** (§5).

### 3(c) CUDA graph coverage of the MTP draft/verify path

Coverage is **already good and is not the problem in the way the brief assumes.** From the boot log:
`Capture target verify CUDA graph … num_tokens_per_req=4, bs=[1,2,3]` → 9.4 s, 0.22 GB · `Capture draft decode … num_tokens_per_req=1` → 2.5 s · `Capture draft extend … num_tokens_per_req=4` → 1.4 s. All three exist.

What is *not* covered: (i) the **PLE layer forces a graph break** (`_breakable_cuda_graph` / `_breakable_ple_forward` in `qwen4_exp.py`), (ii) the **sampling/draft-top-k glue runs between** the three draft replays, and (iii) `--disable-prefill-cuda-graph` is set. Three separate replays + eager glue per iteration means **3 graph launches + Python + a sync point per draft step**. Since 71% of the step is unaccounted latency, folding draft-decode + draft-extend + verify into **one** capture is one of the few changes that attacks the dominant term directly.

### 3(d) NVLink all-reduce per layer: in the critical path, and can overlap help?

**Yes, in the path; and overlap is the wrong first fix — using the right all-reduce is.**

- Count: 48 layers × 2 row-parallel outputs (`o_proj`, `down_proj`) = **96**, + PLE's vocab-parallel gather (1), + LM-head logits gather (4) ≈ **101 collectives/iteration**.
- Payload is tiny: `[4, 2560]` fp16 = **20 KB** per AR. Wire time on NV2 at 154.7 GB/s = **0.13 µs**; realistic NCCL op for this size = **20–35 µs**. So **>99% of each AR is software/sync latency**, and it is serialized against the next layer's GEMV.
- The box *has* the fast path available: full NV2 clique + P2P OK. SGLang disabled `CustomAllreduce` **and** `CustomAllReduceV2` on the theory that these are "PCIe-only GPUs" — a detection outcome, not a hardware fact. `custom_all_reduce_v2.can_use_custom_all_reduce_v2()` requires `is_one_nvlink_clique(...) and _is_vmm_backed_allocator(device)`, and the push-plane specialisation for exactly this shape exists but is **off by default**: `SGLANG_SM70_AR_PUSH16` → `num_push_blocks = 16` when `world_size == 4`, sm70, and NVLink is usable.
- Expected: 1.9–3.4 ms of the 26 ms (7–13%), plus it removes 96 serialization points.
- Overlap (TBO/SBO, both **False** today) is a *second* step: `--enable-two-batch-overlap` needs ≥2 batches to interleave, which at bs=1 does nothing; `--enable-single-batch-overlap` splits one batch and can overlap the AR with the next layer's compute. Worth trying **after** the AR path is fixed, at c≥2 only.

### 3(e) QSA split-K tuning for tiny batch

**Don't fund it. Fund the opposite: prove QSA is doing nothing here.**

- `indexer_budget = 2048` tokens, `compress_ratio = 4` → `block_topk = 512` compressed blocks = **2 048 tokens selectable**. At the benchmark's context (~180 prompt + 128 output ≈ 300 tokens → **≈75 blocks**) the top-k selects **everything**. QSA is dense attention plus an indexer plus a gather.
- Meanwhile KV traffic is **12.6 MB of 6 231 MB (0.2%)**. `triton_attention_num_kv_splits: 8` tunes a kernel that is 0.02 ms of bandwidth. Split-K tuning can therefore move at most ~0.5% — and it may move it the *wrong* way, because split-K adds a reduction kernel per layer (12 more dependent launches).
- The real QSA item is **removing the indexer and the gather at short context**: if `num_blocks ≤ block_topk`, the selection is the identity, so the indexer's q/k projections, the logits, the top-k and the gather-then-attend are all pure overhead across 12 layers. Call the dense path instead. This also removes ~3 launches × 12 layers.
- Caveat that makes it worth doing anyway: if the sparse kernel loops over `block_topk=512` slots regardless of valid blocks, it is doing **~7× the loop iterations** the 300-token context needs; that cost grows as ctx approaches 2 048 and *vanishes* above it. So this change is worth most exactly at the benchmarked regime.

### 3(f) The lever nobody lists: acceptance rate

Tokens/step is 2.81 of 4. From the observed histogram `[0:13, 1:9, 2:10, 3:25]` over 57 verify steps, per-position accept probability is **p₁=0.77, p₂=0.61, p₃=0.44** (decay ≈0.72/step). Extrapolating: 8-draft-token window ⇒ ≈**3.65 tokens/step (+30%)**; and each +0.1 in accept length at fixed step cost is +3.6% tok/s.

| accept len | tok/s at 26 ms | tok/s at 20 ms |
|---|---|---|
| 2.81 (today) | 108 | 140 |
| 3.00 | 115 | 150 |
| 3.30 | 127 | 165 |
| 3.65 (8-token window) | 140 | 182 |
| 4.00 | 154 | 200 |

---

## 4. Ten implementable ideas for this exact model + hardware

All gains are relative to 107.6 tok/s / 26.0 ms. "Boot" = the server must restart, and **a boot costs ~11 min on this box** (measured: 01:04:12 start → 01:15:15 `max_total_num_tokens` — 134 GB from disk, 12 GB pinned-host PLE per rank). So a "12-minute A/B" is really *one boot + one short bench*; I write the recipes that way, and flag the two that are genuinely reboot-free.

Baseline bench command (≈100 s, reuse for every idea):
```bash
cd /workspace/sxm2-verify && venv/bin/python bench.py <label>      # two-point 32->160, 3 reps, median
# plus one accept probe (cheap, 40 s) — accept length moves before throughput does:
python - <<'PY'
import json,urllib.request
P="Explain, in plain prose, what a KV cache is and why it matters for decoding."
req=urllib.request.Request("http://127.0.0.1:11435/generate",
    data=json.dumps({"text":P,"sampling_params":{"max_new_tokens":400,
        "temperature":0,"ignore_eos":True}}).encode(),
    headers={"Content-Type":"application/json"},method="POST")
m=json.loads(urllib.request.urlopen(req,timeout=900).read())["meta_info"]
print(m["spec_accept_length"], m["spec_correct_drafts_histogram"], m["spec_verify_ct"])
PY
```
Use `bench.py`'s two-point number **only** — the in-server `gen throughput` field swings 62↔118 and will lie to you.

---

**1. Hot-vocab draft head + widen the verify window to 8 tokens** — *the big one.*
*Mechanism.* The draft head is 318 MB per pass and runs 3× per iteration. Reshard it to the top ~64 k tokens (`build_sharded_hot_head` in `sxm2/srt/speculative/tp_hot_vocab.py` already does exactly this, re-sharding to keep TP4). At 41 MB/pass, **7 draft passes (809 MB) cost less than today's 3 (1 177 MB)**, which buys a `--speculative-num-steps 7 --speculative-num-draft-tokens 8` window. Accept arithmetic (§3f) takes 2.81 → ~3.65 tokens/step.
*Gain.* **+25–30%** (→135–140 tok/s) even at flat step cost; more if the step also gets shorter (traffic drops 350 MB). *Confidence:* medium (depends on hot-vocab coverage of the draft distribution; a 64 k hot set covering ~85% of draft mass would erode it).
*Cost.* Small change + 2 flags (`--speculative-token-map`, num-steps). *Risk.* Accept collapse if the hot set misses; verify-window >8 will hit the `num_tokens_per_req` graph tiers and the `_qsa_prefill_topk_bridges` sizing.
*Measure.* One boot: `bench.py wide8hot` and read `spec_accept_length` **and** `decode_tok_s` (both must rise; if accept rises but tok/s falls, the extra 4 draft replays cost more than the tokens are worth — that is the decisive datapoint).

**2. Make SGLang use the NV2 clique for the 96 all-reduces.**
*Mechanism.* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (satisfies `_is_vmm_backed_allocator`) + `SGLANG_SM70_AR_PUSH16=1` + `SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage`; if detection still refuses, `SGLANG_CUSTOM_AR_ALLOW_PCIE=1` (push plane is correct on PCIe too). Confirm the log stops printing *"not supported on more than two PCIe-only GPUs"*.
*Gain.* **+6–15%** (1.9–3.4 ms of 26). *Confidence:* medium-high on the hardware, medium on the plumbing.
*Cost.* Config only. *Risk.* A wrong-plane AR is a **silent wrong-answer** class of bug on this box (see `sm70_paths.py` docstring: "the loaders report the kernel as unavailable rather than raising … a missed path produces wrong output, not a crash"). Validate output on 3 prompts before believing the number.
*Measure.* One boot, then `bench.py arpush` + the 3 reference completions. Also grep the new log for `CustomAllReduceV2` to prove which path initialised.

**3. Collapse the MTP chain into one graph replay.**
*Mechanism.* Today: `draft decode` graph, `draft extend` graph, `target verify` graph, plus eager sampling glue between them, plus a PLE graph break. Capture `[draft step ×N → verify]` as a single graph (or at minimum move the draft top-k/argmax inside the graph and remove the PLE break by making the n-gram gather a fixed-shape device op).
*Gain.* **+5–15%** (it attacks the 18.4 ms term directly; each removed replay is ~50–150 µs of launch+sync). *Confidence:* medium. *Cost.* Structural (graph runner + PLE).
*Risk.* Shape polymorphism (`num_tokens_per_req=1` vs `4`) and the pinned-host PLE gather do not capture cleanly; a partial fix can silently disable graphs (`cuda graph: False` in the log = instant 20% regression).
*Measure.* One boot; assert `cuda graph: True` on every decode log line, then `bench.py onegraph`.

**4. Draft top-1 with a 2-float exchange (no full-vocab all-gather).**
*Mechanism.* `SGLANG_SM70_MTP_LOCAL_TOP1=1`. Logits stay shard-local; `argmax` per shard, then `all_gather` of `[value, global_id]` pairs → one tiny collective per draft step instead of materialising `[B, 248320]` on every rank and reducing it.
*Gain.* **+2–5%.** *Confidence:* high (this is a straight port of `sxm2/srt/speculative/sm70_draft_top1.py`, which sm70 lacks).
*Cost.* Flag (+ the module port). *Risk.* Requires topk=1, no LoRA/DP-attention, full vocab — all true here.
*Measure.* One boot; A/B on the same boot is impossible, so pair it with idea 1 (they compose: hot head + local top-1).

**5. Batch the verify's MoE and fuse the small ops.**
*Mechanism.* `SGLANG_SM70_BATCHED_MOE=1` (or `_MOE32`, or `MTP_WIDE_MOE`). §3(a): 40 experts × N=160 at M≤4 → one grouped grid instead of many tiny dependent tiles.
*Gain.* **+3–8%.** *Confidence:* medium. *Cost.* Flag. *Risk.* Triton on sm70 has no TC `tl.dot` and spills above `BLOCK_K=64` — check the variant doesn't silently pick a 128-wide K tile.
*Measure.* Parent is already sweeping this (`sweep.sh`: wide8 / batched16 / batched32). Don't duplicate; read `sweep-results.jsonl`.

**6. Dense-attention bypass under the indexer budget.** *(not a flag flip)*
*Mechanism.* In `Qwen4ExpAttentionDecoderLayer.self_attention`, if `forward_batch`'s max context blocks ≤ `block_topk` (512), skip `_compute_qsa_topk_indices` entirely and call `self.attn(q,k,v,forward_batch)` with no `topk_indices` — the selection is provably the identity. Also skip the indexer's q/k projections and layernorms for that step.
*Gain.* **+1–4%** (12 layers × ~3 dependent launches + a 512-slot gather loop replaced by a 75-block dense kernel). Grows with context up to 2 048 tokens. *Confidence:* medium-high (the identity argument is exact).
*Cost.* Small kernel/wrapper change. *Risk.* Two code paths that must agree numerically at the crossover; graph capture has separate extend/decode variants and the QSA path has a `should_reuse_mtp_sparse_indices` reuse mechanism that must stay consistent across verify steps.
*Measure.* One boot with a 256-token prompt vs a 3 k-token prompt; expect a *larger* delta at 256 than at 3 k. If the gain is flat, the gather wasn't the cost.

**7. Batch all 96 hyper-connection mixers into one kernel per step.** *(not a flag flip)*
*Mechanism.* The HC mix is `10240→320` (SiLU) `→10240` (sigmoid, mean over 4 streams), **twice per layer, replicated, bf16** = 1 266 MB (20.3%) and 96 sequential GEMV pairs. The per-row weights are independent of the token row, so stack all 48 layers' `down`/`up` into `[48, 320, 10240]`/`[48, 10240, 320]` and run **one batched GEMV per plane per step**, writing the 4×4 mixing matrices to a scratch buffer, then a single elementwise pass for the mixes. (sxm2 already fuses the *pair* via `sm70_hc_down_gemv_silu` / `sm70_hc_up_gemv_reduce`; batching *across layers* is the remaining step.)
*Gain.* **+3–5%** (launch count 192 → ~4). *Confidence:* low-medium (the reduce plane must still produce 10 240 values per row — keep `up` as a split-K reduce to avoid a 10 240-wide write).
*Cost.* Structural (a kernel + a weight-layout pass at load). *Risk.* HC weights are load-order sensitive (they are `nn.Linear` outside the TP plan); a bad re-layout breaks numerics subtly. Also fp16/bf16 tolerance: `moe-tol*.py` scripts in `sm70-stack/` show tolerance testing is already a pain point.
*Measure.* One boot, `SGLANG_SM70_HC_NATIVE=0/1` as the control arm; compare `bench.py hcbatch` against the fused-pair baseline.

**8. Use the 51.2 GB n-gram table as a second, free draft source.** *(not a flag flip)*
*Mechanism.* The PLE table is a hashed 3-gram memory of 320 M rows already resident in pinned host memory and already prefetched every step. Today it is used only as an input embedding at layer 1. Point a **prompt-lookup / suffix-automaton drafter** at the same 3-gram statistics (or reuse the PLE hash to score candidates) and feed candidates into the same verify window, so accept length rises without another draft forward. `SGLANG_SM70_PROMPT_LOOKUP=1` is the cheap version of this; making MTP and lookup *co-draft* is the interesting one.
*Gain.* **+0–20%**, workload-dependent (high on repetitive/structured output, ~0 on open prose). *Confidence:* low-medium. *Cost.* Flag for the cheap version; structural for co-drafting.
*Risk.* The verify path must accept a mixed candidate set (tree masks, `ragged_verify`); accept rate can *fall* if candidates are wrong; PLE prefetch traffic grows if the lookup touches the same host pages at a different cadence.
*Measure.* One boot, two prompts (one repetitive — the `LONG_PAD` prompt in `bench.py` — one open prose); report accept_len per prompt, not just the aggregate. Reboot-free variant: measure accept length only, on the already-booted server, if the flag is set at boot.

**9. Re-tune the MTP chain depth for tokens-per-head-pass, and pin the trade with a table.**
*Mechanism.* Each draft pass costs one full LM-head read (318 MB). Today 4 head passes buy 2.81 tokens = **0.70 tokens/pass**. With `--speculative-num-steps 2 --speculative-num-draft-tokens 3` you get 3 passes for ~2.4 tokens = **0.80 tokens/pass**; with idea 1's hot head the optimum moves the other way (toward more steps). This is a **2-D** search (head width × chain depth), and today only the chain-depth axis is being swept.
*Gain.* **+5–10%** at current head width. *Confidence:* medium. *Cost.* Flags. *Risk.* `spec_correct_drafts_histogram` needs enough samples; 128-token runs give ~45 verify steps — use ≥400 tokens for the accept estimate or you will fit noise.
*Measure.* Reuse the parent's `wide8_steps2` arm; add `steps2` **without** `MTP_WIDE_MOE` so the two effects don't confound.

**10. Quantise the GDN and shared-expert weights (they are the last big 16-bit block).**
*Mechanism.* `quantization_config.ignore` exempts `*.linear_attn.*`, `*.mlp.shared_expert.*`, `lm_head`, `mtp.*` — so 36 GDN layers (1 055 MB/step) + shared experts (472 MB) + MTP (224 MB) are read as fp16 GEMVs. FP8 (or NVFP4 for the GDN projections, which are 2560-wide and quantise well) cuts ~1 500 MB/step by 2–4×, i.e. **up to 1.4 ms**.
*Gain.* **+3–6%** on its own, and it frees ~7 GB of device memory (GDN is ~4.2 GB of the 15 GB bf16 group), which buys KV/graph headroom at 30/32 GB. *Confidence:* medium (accuracy risk on the delta-rule state). *Cost.* Structural (offline requantisation + loader). *Risk.* GDN is numerically delicate (`mamba_ssm_dtype: float32` state); fp8 on `in_proj_qkv` may degrade long-context recall.
*Measure.* Requires a requantised checkpoint — the only idea here that does **not** fit a 12-minute slot; gate it behind a quality check (the `aime26_metrics.json` / `gsm8k_metrics.json` in the checkpoint dir are the reference points).

**Composition.** Ideas 1+4+9 are one coherent bundle (draft-head economics); 2+3 are the latency bundle; 5+7 are the kernel-count bundle. Individually the median idea is ~4%; the bundles are where the 26 ms goes.

---

## 5. What sxm2 likely does better (concrete decode-path differences)

The gap is **not** one clever kernel. It is that `sxm2` carries an entire sm70 specialisation layer that `sm70/stack` does not have at all. Evidence, in order of expected impact:

1. **26 `SGLANG_SM70_*` env knobs vs 0 at analysis time.** The upstream tree that became
   `sglang-sm70` contained **no** `SGLANG_SM70_*` symbol; `sglang-sxm2` contained 26. (The fork
   now carries 39 of them, 26 of which came from this list -- see the environment reference.) Defaults that matter: `SM70_HC_NATIVE=True`, `SM70_MTP_HC=True`, `SM70_MTP_QKVZBA=True`, `SM70_MTP_GDN=True`, `SM70_MTP_SMALL_GEMM=True`, `SM70_QSA_COMBINE=True`, `SM70_QSA_DRAFT_EXTEND_GRAPH=True`, `SM70_FORCE_FP16=True`; and *off* by default: `SM70_AR_PUSH16`, `SM70_MTP_LOCAL_TOP1`, `SM70_BATCHED_MOE`, `SM70_BATCHED_MOE32`, `SM70_BATCHED_FUSIONS`, `SM70_HC_COMBINE_NORM`, `SM70_MTP_WIDE_MOE`, `SM70_PROMPT_LOOKUP`, `SM70_MTP_SHARED_GATE`, `SM70_DENSE_GEMV`, `SM70_QWEN_FUSIONS`. The two forks therefore do not run the same decode graph even for identical flags.

2. **Fused Volta hyper-connection GEMVs.** `kernels/ops/gemm/hc_mix.py`: sm70 = 6 588 B (only the persistent `fused_hc_mix`), sxm2 = 16 881 B with `sm70_hc_down_gemv_silu`, `sm70_hc_down_gemv_silu_supported`, `sm70_hc_up_gemv_reduce` and **`_SM70_MAX_MIX_ROWS = 4`** (explicitly sized for a 4-row MTP verify), gated by `SGLANG_SM70_HC_NATIVE` / `SGLANG_SM70_MTP_HC`. `hyperconnection.py` imports these in sxm2 and not in sm70. Given HC is **20.3% of step traffic and 96 sequential GEMV pairs**, this is the largest single mechanical difference — and it explains why the merged tree hits 107 while the sm70 stack reports 74.

3. **Draft top-1 without a full-vocabulary gather** — `srt/speculative/sm70_draft_top1.py` exists only in sxm2 (79 lines, quoted in §4.4). Removes a 248 k-wide all-gather + argmax **three times per iteration**.

4. **NVLink push all-reduce for a 4-way clique** — `custom_all_reduce_v2.py:220-230`: `world_size == 4 and SGLANG_SM70_AR_PUSH16 and capability == (7,0) and can_use_custom_all_reduce_with_nvlink(...) → num_push_blocks = 16`, described as *"Small TP4 messages keep fewer push CTAs resident"*. `sm70` has no such branch. Also `SGLANG_CUSTOM_AR_ALLOW_PCIE` and `SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage` exist only in sxm2, with the comment that the V100 mesh is *"only partially NVLink-connected (in-domain links plus a single cross-pair link per rank), where the two-stage algorithm is pathological, so the production launcher pins `1stage`."* — i.e. the reference launcher avoids two-shot, and the stack may not.

5. **A single place that knows where SM70 artifacts live** — `kernels/sm70_paths.py` (sxm2 only). Its docstring is a warning worth heeding: the prebuilt `_sm70_*.so` are gitignored, every loader *fails silently*, and *"the stock Marlin MoE path is an empty stub on SM70 that writes nothing — so a missed path produces wrong output, not a crash."* Any sm70 stack that loses the `.so` files scores great and lies.

6. **Hot-vocab sharded draft head** — `srt/speculative/tp_hot_vocab.py` (sxm2 only), 45 lines, re-shards a selected hot set to keep TP4 projection parallelism.

7. **Larger speculative/spec-infra surface in sxm2**: `ragged_verify.py` (11 273 B vs 10 712 B), `eagle_draft_cuda_graph_runner.py`, `frozen_kv_mtp_*`, `uncertainty`/`dflash`/`uno` workers, plus `third_party/flash_v100` (a FlashAttention V100 port with a `vision_host_compat.patch`). Conversely, sm70 has `sglang/kernels/ops/…` code that sxm2 dropped: the trees differ by 1 270 files (+60 098 / −132 760 lines), so *neither is a superset* — the fast configuration is a merge, which is exactly what `/workspace/sxm2-verify/fork` is.

8. **QSA differences**: sxm2 reuses a single `_qsa_prefill_topk_bridge` buffer per bucket while sm70 keeps a *list* of bridges (`_qsa_prefill_topk_bridges`, newest first). Prefill-only; irrelevant to decode, but it shows sm70's QSA plumbing is a different generation.

### What I would NOT blame for the 6 tok/s
- **MoE dispatch / expert-parallel**: `ep_size 1`, `moe_a2a_backend none`, `moe_dp_size 1` in the live config — both trees run plain TP4 FusedMoE. Nothing to win or lose here.
- **Sampling**: `--sampling-backend flashinfer`, `--warmups sampling`; the FlashInfer pin + signature matches (`SM70.md` §19). Not a differentiator.
- **Attention backend**: both run `tilelang_fa_v100`. The decode attention is 0.2% of traffic; a backend change cannot explain 6 tok/s.

---

## 6. Appendix — what I measured, and how to reproduce

Environment facts confirmed on the box (2026-11-11):

| fact | value |
|---|---|
| GPUs | 4× Tesla V100-SXM2-32GB + 1× RTX 3090 Ti (host-attached, not in the group) |
| topology | `nvidia-smi topo -m`: full **NV2** mesh among GPU0-3; `topo -p2p r`: **OK** on all 6 pairs |
| NVLink | 6 links/GPU × 25.781 GB/s (`nvidia-smi nvlink -s`) |
| live flags | `--dtype float16 --quantization modelopt_fp4 --attention-backend tilelang_fa_v100 --linear-attn-decode-backend triton --linear-attn-prefill-backend tilelang --tp 4 --mem-fraction-static 0.86 --max-running-requests 3 --max-mamba-cache-size 20 --cuda-graph-bs-decode 1 2 3 --disable-prefill-cuda-graph --enable-hierarchical-cache --hicache-size 8 --ple-offload-embedding --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-two-batch-overlap False --enable-single-batch-overlap False` |
| memory | PLE table **11.92 GiB/rank in pinned host**; device 19.89 GB/rank; KV 324 160 tokens (K 1.86 + V 1.86 GB); `qsa_indexer` pool 0.41 GB host; mamba cache 0.97 GB host / 65 state slots; HiCache host pool 6.14 GiB; **avail 4.73 GB** |
| boot time | **11 min** (01:04:12 → 01:15:15), dominated by 134 GB of checkpoint reads |
| graphs | target verify (bs 1-3, 4 tok/req, 0.22 GB), draft decode, draft extend — all captured |

Reproduce the headline measurement (no repo change, ~2 min):
```bash
cd /workspace/sxm2-verify
venv/bin/python bench.py baseline            # two-point decode, TTFT, c3 aggregate
grep -E "Decode batch" serve-final-check.log | tail -5   # accept_len + in-server throughput
```
Then read `spec_accept_length` and `spec_correct_drafts_histogram` out of any `/generate` `meta_info` — **those two fields, not tok/s, are the fastest signal** because accept length moves before throughput does.

**Caveats I want on the record.**
- The 6 231 MB ledger assumes standard TP sharding and treats HC/PLE projections as replicated (they are created as plain `nn.Linear`/`ReplicatedLinear`, outside the TP plan), and the shared expert as sharded. Shifting the shared expert to replicated changes the total by ±0.5 GB (±6%); the 71%-unaccounted conclusion is robust to that.
- "Distinct experts" per step is bounded above by `B·topk` (40) and below by ~`0.6·B·topk` (24) because the 4 verify tokens are a draft chain and route similarly. Both are tabulated.
- The split of the 18.4 ms between kernel-tail latency and collectives is a **model**, not a profile. `--enable-layerwise-nvtx-marker` + one `nsys` capture is the 12-minute experiment that replaces it with data, and it should be the *first* thing run — every other estimate in §3 inherits its uncertainty.

### Sources

- V100 architecture support matrix (sm_70/75+ feature gates): [veda-sparse/miowtion arch_support.md](https://github.com/veda-sparse/miowtion/blob/ffbb4c29d9b1f19f026dec0c0485840fffe4639f/docs/features/arch_support.md)
- V100 (sm_70) measured optimisation findings — no `cp.async`/TMA, ~816 GB/s measured, Triton sm70 FMA fallback and `BLOCK_K` register spill: [V100_OPTIMIZATION_FINDINGS.md](https://raw.githubusercontent.com/KumphanartDansiri/vllm-fp8-w8a16-sm70/refs/heads/main/docs/V100_OPTIMIZATION_FINDINGS.md)
- CUTLASS GPU architecture support matrix (tensor-core shapes per arch): [NVIDIA/cutlass on DeepWiki](https://deepwiki.com/NVIDIA/cutlass/1.3-gpu-architecture-support)
- Benchmarking V100 and tensor cores: [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-030-10549-5_35)
- SGLang scheduling/parallelism/overlap (TBO/SBO, comm-stream structure): [llm-infra-atlas §06](https://github.com/llm-infra-atlas/llm-infra-atlas.github.io/blob/main/docs/serving/06_scheduling_parallelism_and_overlap.md), [SGLang PR #19216 (two-batch overlap)](https://github.com/sgl-project/sglang/pull/19216)
- Qwen3-Next / Gated DeltaNet hybrid attention (the architecture family this model belongs to): [NeurIPS 2025 Qwen attention award note](https://www.alizila.com/alibaba-qwen-wins-neurips-2025-best-paper-award-for-breakthrough-in-attention-mechanisms/), [Gated DeltaNet overview](https://baike.baidu.com/item/Gated%20DeltaNet/68793531)
- Local: `sglang-sxm2/python/sglang/srt/layers/attention/qsa/config.py` (`block_topk = budget/compress_ratio`), `kernels/ops/gemm/hc_mix.py`, `distributed/device_communicators/custom_all_reduce_v2.py`, `speculative/sm70_draft_top1.py`, `speculative/tp_hot_vocab.py`, `kernels/sm70_paths.py`; `sglang-sm70/python/sglang/srt/models/qwen4_exp.py`, `layers/hyperconnection.py`; box logs `/workspace/sxm2-verify/serve-final-check.log`, `sweep-wide8.log`, `bench-baseline.json`.

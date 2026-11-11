# SM70 serving performance: measurement, findings, and the roadmap

> **Provenance.** This report covers **sglang-sm70**, the SM70 (V100) fork of SGLang carried as ten slices on upstream `2c5b680aa6` (`main`).
> Measurements attributed to "this fork" were taken on 4× V100-SXM2-32GB at a 150 W cap with `bench*.py` (warm, clocks ≥1500 MHz, medians).
> Measurements attributed to "predecessor" are quoted from the records of the predecessor fork (sglang-sxm2) and may come from different hardware or power limits.
> Credits and licenses are in [`NOTICE`](../../NOTICE).

Target: RadixArk/Qwen3.8-Flash-Next-NVFP4, 4× V100-SXM2-32GB (sm70), TP4 over a full NVLink NV2
clique, MTP/EAGLE (3 steps, 4 drafts). All numbers from `/workspace/sxm2-verify/fork` on the box,
clean tree of `sm70/stack` with the dense marlin_v100 repack installed.

## 1. Measurement methodology (and why the old numbers were noise)

Decode throughput on this box is **bimodal**: the same binary swings 62–118 tok/s between steps.
Wall-clock `/generate` therefore cannot resolve the differences we care about. The harness measures
four independent things and reports medians:

| Metric | How | Why |
|---|---|---|
| `decode_tok_s` (two-point) | time `max_new_tokens=32` and `=160` on one prompt, take `(160−32)/(t₂−t₁)` | cancels prefill, queueing and tokenizer cost |
| `decode_throughput_med` | the scheduler's own per-request `decode_throughput` from `meta_info`, median of 5 | server-side, immune to client jitter |
| `accept_length_med` | `meta_info.spec_accept_length` | ~noise-free proxy; it moves before tok/s does |
| `conc[N]_agg` | N concurrent 96-token requests, median of 2 | the metric that matters for serving |

Files: `bench.py` (quick two-point), `bench2.py` (per-prompt acceptance + 1/2/3 streams),
`bench3.py` (server metrics, 5 reps), `bench4.py` (1/2/4/8 streams). Results are JSON per config.

**The "~80 tok/s" figure attributed to the predecessor fork (sglang-sxm2) is not a real number.** The archaeology report traced it: neither
repo documents 80. What sxm2 actually documents is 112.8 tok/s averaged over six *chat* prompts
(accept 2.18–2.97), 108–118 in its MTP campaign, and 73.5–74.9 target-only. Our clean tree measured
**107.6 tok/s** two-point on its first run, i.e. the two trees are in the same band and the
"gap" was measurement noise plus one bad run.

## 2. What the step actually costs

From the architecture analysis (per MTP iteration at TP4, per rank):

- weight + state traffic **6231 MB** → at the box's measured **816 GB/s** ≈ **7.6 ms** if perfectly overlapped;
- compute ≈ 2.7 GFLOP/GPU → 0.02 ms (0.1% — irrelevant);
- measured cycle **26.0 ms** ⇒ achieved ~240 GB/s, **29% of peak**.

So the iteration is **latency-bound, not bandwidth-bound**: ~18 ms is launch/dependency/collective
latency. Traffic is dominated by routed experts (21.3%), the LM head ×4 passes (20.4%) and the
hyper-connection mixers ×96 (20.3%, replicated because they sit outside the TP plan).

Both forks' phase tables agree: **target verify ≈ 77% of the cycle** (17.6 of 22.7 ms), draft 3.2,
draft extend 1.6, commit 0.2, CPU graph submission ~1%.

## 3. Round 1 — row-cap and fusion sweep (12 boots)

`SGLANG_SM70_BATCHED_MOE*` raises the NVFP4 decode row cap (`sm70_nvfp4_moe_decode_max_rows()`),
which gates `marlin.py:153` and `topk.py:1314`. With `--speculative-num-draft-tokens 4`, verify
forwards are **M = 4/8/12 rows at np = 1/2/3**, so at the shipped cap of 4 only the single-request
case ever ran the native kernel.

| config | decode (two-point) | prose | code | TTFT ms | 3-stream agg |
|---|---|---|---|---|---|
| baseline (cap 4) | 107.6 | 89.1 | 101.4 | 432 | 78.8 |
| `MTP_WIDE_MOE=1` (cap 8) | 104.3 | 105.1 | 104.3 | 435 | 128.0 |
| `BATCHED_MOE=1` (cap 16) | 105.9 | 103.3 | 101.9 | 423 | 137.9 |
| **`BATCHED_MOE32=1` (cap 32)** | **110.2** | **107.7** | **113.9** | **387** | 137.6 |
| cap 8 + `BATCHED_FUSIONS` + `MTP_SHARED_GATE` | 100.3 | 99.8 | 97.6 | 424 | 137.8 |
| cap 8 + 2 steps / 3 drafts | 88.6 | 93.4 | 95.0 | 448 | 132.4 |
| cap 8 + `MTP_LOCAL_TOP1` | **hangs** | — | — | — | — |
| FlashAttention-V100 for **vision**, single small image | — | — | — | TTFT **+7% / +13% / +2.5%** at 512/1024/1536 px | rejected for chat |
| FlashAttention-V100 for **vision**, multi-image documents | — | — | — | **4.4x faster** warm on a 32-image request (3.92 s vs 17.34 s; 24,719 prompt tokens) | enable for page/PDF/video batches |
| FlashAttention-V100 for **language** attention | — | — | — | dense prefill **1.65-1.68x slower** than the in-tree TileLang kernel | decode attention is <=0.47% of the step |

Findings:

1. **Cap 32 is the winner** and it is the only change that improved both decode and TTFT
   (387 ms vs 432). It is worth ~+3% single-stream and removes the np=2/3 fallback entirely.
2. **`BATCHED_FUSIONS` + `MTP_SHARED_GATE` hurt** here (−10%) despite the fork's own measurements
   showing them faster in isolation. Measured on this workload, they stay off.
3. **A shorter MTP chain is worse**: 2 steps / 3 drafts keeps a higher per-draft acceptance rate
   (0.69 vs 0.61) but fewer tokens per step (2.39 vs 2.81), and loses 20% throughput.
4. **`MTP_LOCAL_TOP1` deadlocks**: the scheduler watchdog fires at 300 s, i.e. the shard-local
   two-float all-gather never completes. A real bug, not a slow path — it must stay off until
   fixed. (Candidate cause: the all-gather is issued on a stream the logits path never synchronises.)
5. Acceptance is the dominant term, not any kernel: decode tracks `accept_length` almost linearly
   across prompts (0.31→79 tok/s, 0.42→88, 0.56→101, 0.61→107).

## 4. Rounds 2–5 — concurrency, collectives, chunk size, and the power cap

### The power cap changes what a number means

These V100s run at a **150 W cap** (default 300 W) with a **135 MHz idle clock that only applies
after 120 s of idleness** and a ramp delay on wake. While benchmarking, `bench5.py` warms for ~45 s
and samples clocks: measurements land at **1500–1530 MHz** with ~60 W draw, so the cap is not
throttling *steady-state* decode — but any measurement taken right after idling pays the ramp.

Consequences, both honest:

1. Absolute numbers here are **not** comparable with docs measured on 300 W cards. If sxm2's
   documented 112.8 tok/s was taken uncapped, our ~110 at 150 W is the more efficient result.
2. Every config comparison in this report was re-measured warm, with clocks recorded. The earlier
   "62↔118 tok/s bimodality" was this ramp, not the engine.

### Results (warm, median of 5, clocks ≥1500 MHz)

| config | decode tok/s | server tok/s | accept len | TTFT | agg @2 | agg @4 | agg @8 |
|---|---|---|---|---|---|---|---|
| shipped (cap 4, mrr 3, graphs 1-3) | 107.6 | — | 2.81 | 432 ms | 98.9 | — | — |
| cap 32, mrr 4, graphs 1 2 4 | 109.2 | 110.3 | 2.81 | 387 ms | 106.0 | 161.9 | 188.9 |
| **cap 32, graphs 1 2 4** (mamba-capped to 4 requests) | **108.1** | **111.0** | 2.81 | 387 ms | **122.3** | **184.4** | 184.3 |
| cap 32 + `AR_PUSH16` + expandable_segments | — | 104.1 | 2.70 | — | — | — | — |
| cap 32 + 2K chunks | — | 106.0 | 2.70 | — | — | — | — |

Findings:

1. **Concurrency is where the win is.** 2 streams 122.3 (+24% over the shipped 2-stream 98.9),
   4 streams 184.4 (+34% over the shipped 3-stream 137.6). Saturation is around 4–8; 8 adds nothing
   over 4 and costs KV headroom.
2. **Custom all-reduce is neutral** (`AR_PUSH16` + `expandable_segments` measured 104.1 vs 110.3):
   the collective path is not the bottleneck this analysis feared, or this box cannot use the v2
   path at all. Not worth shipping.
3. **2K chunks are neutral for decode** and only ever helped TTFT; the recipe keeps 4K for prefix
   density (the archaeology report documents that 4K lost ~2k tokens of prefix cache vs 2K, and 8K
   OOMs the 11.9 GiB PLE table).
4. **Row cap 32 buys ~3% decode and −45 ms TTFT** and removes the fused-Marlin fallback at np=2/3,
   which is what makes the higher concurrency win possible at all.

### Vision attention is regime-dependent

FlashAttention-V100 looks like a loss or a win entirely depending on how many images one
request carries:

- **One small image** (our 512/1024/1536 px test): the adapter zero-pads Qwen3.8's 72-wide
  vision heads to 128, so QK/PV do ~3.2x the arithmetic, and the vision tower is a small
  slice of TTFT anyway. Net: 7-13% *worse* TTFT. Triton wins.
- **Many large images.** Measured here with 32 JPEGs at 768x1024 in one request
  (24,719 prompt tokens, CPU pixel preprocessing, clocks at 1530 MHz): **17.34 s on
  triton_attn vs 3.92 s on fa2_v100 warm, a 4.4x win.** The first request after boot is
  78.61 s vs 16.76 s, because Triton compiles its vision kernel and FA pays its own
  first-call setup. The V100 fork measured the same shape at 79.98 s -> 12.12 s, profiling
  17.3 s of a 20.6 s GPU-activity span inside Triton's `_fwd_kernel`.
- **Video** is the same many-frame regime, so enable FA-V100 for video serving too.

So the backend choice is a workload decision, not an absolute: keep `triton_attn` for
single-image chat, set `SGLANG_SM70_VISION_ATTENTION=1` (or `--mm-attention-backend
fa2_v100`) for document/page batches. It requires eager vision execution, i.e.
`SGLANG_VIT_ENABLE_CUDA_GRAPH` has to stay off. See
[INSTALL.md](INSTALL.md#optional-flashattention-v100-for-the-vision-encoder) for the build.

### Triton fused-MoE configs on V100 (the one remaining startup notice)

Boot logs `Using default MoE kernel config. Performance might be sub-optimal! Config file not found
... E=512,N=160,device_name=Tesla_V100-SXM2-32GB.json`. That is the **prefill** MoE path: decode uses
the native NVFP4 kernels (`SM70 ... small-batch NVFP4 MoE decode kernel loaded`), so this costs TTFT,
not decode tok/s.

The lookup falls back across triton versions (`fused_moe_triton_config.py:114-136`), so a config
tuned on any Triton release is picked up here with a warning that it came from another version. To
produce one for this shape:

```bash
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
  --model /models/Qwen3.8-Flash-Next-NVFP4 --tp-size 4 --tune
# writes configs/triton_<ver>/E=512,N=160,device_name=Tesla_V100-SXM2-32GB.json
# plus the _down variant; commit them under python/sglang/srt/layers/moe/moe_runner/
# triton_utils/configs/triton_<ver>/
```

Not done here on purpose: `--tune` sweeps block sizes across batch sizes and takes hours of GPU
time, and the gain lands on TTFT only. Until then the default config is correct, just not tuned.

## 5. What we ship

The recipe's own defaults: `--max-running-requests 4` (the mamba state cache caps it at 4 either way), `--cuda-graph-bs-decode 1 2 4`,
`--cuda-graph-max-bs-decode 8`, and `SGLANG_SM70_BATCHED_MOE32=1`. Everything else stays as it was.

Measured with the recipe alone (no overrides) after the change, clocks verified at 1530 MHz,
median of 5 two-point samples:

| metric | value |
|---|---|
| decode tok/s (two-point) | **108.2** (samples 101.4–108.5) |
| decode tok/s (server-reported) | **109.9** |
| acceptance length | 2.81 of 4 drafts |
| TTFT | 387 ms |
| aggregate @1 / @2 / @4 / @8 streams | **74.6 / 118.5 / 181.6 / 197.6 tok/s** |

That is **parity with sxm2 on single-stream** (its documented 108–118 band) and **ahead on
aggregate** (its documented 121 @2, 133.6 @3 vs our 118.5 @2, 181.6 @4, 197.6 @8), at a 150 W cap.



## 5. Bugs found (worth fixing regardless of throughput)

1. `MTP_LOCAL_TOP1` hangs the scheduler (watchdog, no progress).
2. `--num-continuous-decode-steps` is declared and registered but has **zero readers** in this tree.
3. `QSA_DECODE_TARGET_CTAS` / `QSA_DECODE_TOKENS_PER_SPLIT` are module constants, not env knobs, so
   the "SM70 QSA decode" tuning documented elsewhere cannot actually be changed at runtime.
4. The CUDA long-context decode partial is inert under the fp16-KV decision
   (`_paged_adapter.py` requires fp8 **and** e5m2 **and** page_size 16).

## 6. Where the remaining performance is (ranked, with expected size)

1. **Draft LM head sharding / FR-Spec for native MTP** — the draft head is a full 248k-vocab matmul
   run 3× per iteration (~20% of all traffic). `tp_hot_vocab.build_sharded_hot_head` already exists
   and is wired for EAGLE3 checkpoints, but the native Qwen4 MTP has no `hot_token_id`, so the fast
   path never triggers. Building a hot-token list and resharding the draft head would cut that 20%
   and make a wider draft window (7–8 tokens) affordable: the acceptance histogram implies ~3.65
   accepted tokens/step at width 8 vs 2.81 at width 4 (+30%).
2. **One graph for the whole MTP iteration** — currently target-verify, draft-decode and
   draft-extend are three separate graph replays plus eager sampling glue and a PLE graph break.
   With 71% of the step in latency, collapsing 3 replays into 1 is the structural answer.
3. **Collectives**: custom all-reduce v2 is disabled on this box even though `nvidia-smi topo -m`
   shows a full NV2 clique; the log's "PCIe-only" reasoning does not match the topology. 96 tiny
   all-reduces per step at 20–35 µs each is 2–3 ms of the 26.
4. **Shard the hyper-connection mixers**: 1266 MB/step is replicated because those `nn.Linear`s are
   outside the TP plan (20% of traffic).
5. Quantising the 16-bit leftovers (GDN projections, shared experts, MTP) would need a requantised
   checkpoint — out of scope here, +3–6% estimated.

## 7. Recommendation

Ship `SGLANG_SM70_BATCHED_MOE32=1` as the recipe default: it is the only measured win, it costs
nothing, and it removes the np=2/3 kernel fallback. Keep `BATCHED_FUSIONS`, `MTP_SHARED_GATE`,
`MTP_LOCAL_TOP1` and short MTP chains off, and record why, so the next person does not re-run them.

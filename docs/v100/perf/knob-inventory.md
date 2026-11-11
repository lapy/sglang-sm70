# SM70 fork — decode-throughput knob inventory

> **Provenance.** This report covers **sglang-sm70**, the SM70 (V100) fork of SGLang carried as ten slices on upstream `2c5b680aa6` (`main`).
> Measurements attributed to "this fork" were taken on 4× V100-SXM2-32GB at a 150 W cap with `bench*.py` (warm, clocks ≥1500 MHz, medians).
> Measurements attributed to "predecessor" are quoted from the records of the predecessor fork (sglang-sxm2) and may come from different hardware or power limits.
> Credits and licenses are in [`NOTICE`](../../../NOTICE).

Target: `RadixArk/Qwen3.8-Flash-Next-NVFP4` on 4x V100-SXM2-32GB, TP4, NVLink NV2 full clique.
Baseline: **~74 tok/s** decode at 128 generated tokens with MTP/EAGLE (`mtp` mode of the serve script).
Goal: **>= 80 tok/s** — the reference attributed to the predecessor fork (sglang-sxm2) on the same hardware.

## How this was read (important)

The working tree of `sglang-sm70` is checked out on **`main`**, which contains **none** of the SM70
work (`git grep SGLANG_SM70_` on `main` → 0 hits; `scripts/serve_qwen38_flash_next_nvfp4_v100.sh`
does not exist there). The SM70 stack lives on branch **`sm70/stack`** (`9e908c9158`), with the
per-area branches checked out under `sglang-sm70-wt/*`.

Everything below was read **read-only** from `sm70/stack` via `git show sm70/stack:<path>` and
`git grep <pat> sm70/stack`. No checkout, no worktree, no file in the repo was modified.
**All `file:line` references are line numbers on `sm70/stack`.**

Path corrections vs. the task brief:
* `kernels/ops/moe/sm70_nvfp4_moe_decode.py` is really
  `python/sglang/kernels/ops/moe/sm70_nvfp4_moe_decode.py`.
* `srt/layers/attention/tilelang_fa_v100/_decode_cuda.py` exists as stated.
* CLI flags are **not** in `srt/server_args.py` in this tree; they are `msgspec` structs under
  `python/sglang/srt/arg_groups/fields/*.py`. Defaults cited below are from those files.

Legend for **fast setting**: "FAST" = the value that helps decode tok/s, "SLOW" = hurts it,
"NEUTRAL" = no decode-path effect.

---

## 0. The one mechanical fact that drives most of this inventory

MTP/EAGLE target-verify forwards **`bs * speculative_num_draft_tokens` token rows per step**, and the
SM70 NVFP4 MoE fast path is gated on a **row cap that defaults to 4**:

| Evidence | What it says |
|---|---|
| `python/sglang/srt/model_executor/model_runner.py:917-936` | `decode_num_tokens_per_req()` = logits rows per decode **batch slot** |
| `python/sglang/srt/speculative/spec_utils.py:95-105` | `draft_decode` → `eagle_topk`; `draft_extend` → `num_draft_tokens`; `target_verify` → `num_draft_tokens` |
| `python/sglang/srt/model_executor/model_runner.py:960-971` | `max_decode_logits_rows() = max(capture_bs) * num_tokens_per_req` (i.e. rows are **per request**, not per batch) |
| `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py:387,419-424` | decode graphs are captured in **requests**, `max_bs = max(capture_bs)`, and `num_tokens = bs * captured_req_width` (`:1131`, `:1420`) |
| `python/sglang/kernels/ops/moe/sm70_nvfp4_moe_decode.py:28-42` | `sm70_nvfp4_moe_decode_max_rows()` → **4** default, 8 / 16 / 32 with `MTP_WIDE_MOE` / `BATCHED_MOE` / `BATCHED_MOE32` |
| `python/sglang/srt/layers/moe/moe_runner/marlin.py:150-178` | native NVFP4 decode used only if `1 <= rows <= max_rows()` |
| `python/sglang/srt/layers/moe/topk.py:1313-1330` | the **router** top-k (512→10) is gated by the **same** cap |

So with the shipped script (`--max-running-requests 3`, `--speculative-num-draft-tokens 4`):

| concurrent requests | verify rows | hits `max_rows()==4` fast kernel? |
|---|---|---|
| 1 | 4 | **yes** |
| 2 | 8 | **no** → `fused_marlin_moe` + generic top-k |
| 3 | 12 | **no** → `fused_marlin_moe` + generic top-k |

At `--max-running-requests 3` the fast kernel therefore covers only the one-request case and the
engine spends most of its steady state on the Marlin fallback for both the routed experts and the
router. **`SGLANG_SM70_BATCHED_MOE=1` (cap 16) is the single highest-value experiment in this
document.**

---

## 1. `SGLANG_SM70_*` / `SGLANG_V100_*` fields

### 1a. Fields declared in `python/sglang/srt/environ.py`

| knob | default | effect (hot path) | fast setting | how to A/B | P |
|---|---|---|---|---|---|
| `SGLANG_SM70_BATCHED_MOE` | `False` — environ.py:1785 | Raises the NVFP4 decode row cap 4→**16** (`sm70_nvfp4_moe_decode.py:38`), gating both the routed-expert GEMM (`marlin.py:153`) and the router top-k (`topk.py:1314`) | **ON** (slow default) | `SGLANG_SM70_BATCHED_MOE=1 bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp` | **P0** |
| `SGLANG_SM70_BATCHED_FUSIONS` | `False` — environ.py:1789 | Extends the FP16 shared-expert/projection fusions to 8/12/16/20/24/28/32 rows (`sm70_qwen_fusions.py:26-27`, `:75`) | **ON** (slow default) | `SGLANG_SM70_BATCHED_FUSIONS=1 …` | **P0** |
| `SGLANG_SM70_MTP_SHARED_GATE` | `False` — environ.py:1776 | Extends the shared-expert sigmoid-gate fusion from 1 row to the 2/4-row verify shapes (`sm70_qwen_fusions.py:19-31`) | **ON** (slow default) | `SGLANG_SM70_MTP_SHARED_GATE=1 …` | **P0** |
| `SGLANG_SM70_MTP_LOCAL_TOP1` | `False` — environ.py:1949 | Draft head keeps the vocab sharded and exchanges only each shard's winner (2-float all-gather) instead of a full-vocab logits all-gather per draft step (`speculative/sm70_draft_top1.py:15-59`) | **ON** (slow default) | `SGLANG_SM70_MTP_LOCAL_TOP1=1 …` | **P0** |
| `SGLANG_SM70_AR_PUSH16` | `False` — environ.py:1793 | Fewer resident push CTAs (16) for the TP4 all-reduce; small decode reductions were measured faster with 16 blocks (`distributed/device_communicators/custom_all_reduce_v2.py:304-316`) | **ON** for a 4-way NVLink clique (slow default) | `SGLANG_SM70_AR_PUSH16=1 …` | **P0** |
| `SGLANG_V100_GREEDY_TP_TOP1` | `False` — environ.py:1822 | Strict-greedy batches exchange one exact top-1 candidate per TP rank instead of all-gathering full-vocabulary logits (`layers/logits_processor.py:245-248`) | **ON** if traffic is greedy (slow default) | `SGLANG_V100_GREEDY_TP_TOP1=1 …` | **P0** (conditional) |
| `SGLANG_SM70_BATCHED_MOE32` | `False` — environ.py:1786 | Row cap 4→**32** (`sm70_nvfp4_moe_decode.py:36`); covers verify at np=4 with 4 drafts (16 rows) and 8-draft trees | **ON** only if `--max-running-requests` or draft tokens grow | `SGLANG_SM70_BATCHED_MOE32=1 …` | **P1** |
| `SGLANG_SM70_MTP_WIDE_MOE` | `False` — environ.py:1784 | Row cap 4→**8** (`sm70_nvfp4_moe_decode.py:40`); covers bs=2×4-draft verify and the 8-row draft-extend | ON if you test 5-8 draft tokens at np<=1 | `SGLANG_SM70_MTP_WIDE_MOE=1 …` | **P1** |
| `SGLANG_SM70_MTP_GDN` | `True` — environ.py:1768 | Native small-batch GDN linear-attn decode/verify for MTP rows | already ON (fast default) | `SGLANG_SM70_MTP_GDN=0 …` = control | P2 |
| `SGLANG_SM70_MTP_QKVZBA` | `True` — environ.py:1767 | Fused QKV+Z+B+A MTP projection for 2/4/8 rows (`sm70_qwen_fusions.py:69-79`) | already ON | `…=0` = control | P2 |
| `SGLANG_SM70_MTP_SMALL_GEMM` | `True` — environ.py:1766 | Small-GEMM path for MTP rows (`kernels/ops/gemm/sm70_small_gemm.py:39`) | already ON | `…=0` = control | P2 |
| `SGLANG_SM70_MTP_HC` / `_HC_GATE` / `HC_NATIVE` | `True` — environ.py:1763-1765 | Hyperconnection kernels (`kernels/ops/gemm/hc_mix.py:137,274`) | already ON — but **Qwen3.8 has no hyperconnection layers**, so inert | `…=0` = control | P2 |
| `SGLANG_SM70_HC_COMBINE_NORM` | `False` — environ.py:1772 | Fuses HC combine with the next Gemma norm (`kernels/ops/layernorm/sm70_hc_combine_norm.py:16`) | N/A — DSV4.1-family only, **not** exercised by Qwen3.8 | — | P2 |
| `SGLANG_SM70_DENSE_GEMV` | `False` — environ.py:1761, **serve script sets `=1` (line 70)** | Native batch-1 Volta GEMV for projections + logits (`kernels/ops/gemm/sm70_dense_gemv.py:48`, `sm70_rows_gemv.py:28`, `logits_processor.py:254,1592`) | already ON via script (fast) | `SGLANG_SM70_DENSE_GEMV=0 …` = control | P1 |
| `SGLANG_SM70_QWEN_FUSIONS` | `False` — environ.py:1762, **serve script sets `=1` (line 71)** | Master switch for the Qwen3.8 small-batch fusions (`sm70_qwen_fusions.py:11`) | already ON via script (fast) | `SGLANG_SM70_QWEN_FUSIONS=0 …` = control | P1 |
| `SGLANG_SM70_QSA_COMBINE` | `True` — environ.py:1794 | Uses the native combine kernel for the QSA split-K decode partial when `1 <= batch <= 4` (`tilelang_fa_v100/_decode_cuda.py:232-248`) | already ON; **breaks above batch 4** (see area 5) | `…=0` = control | P1 |
| `SGLANG_SM70_QSA_DRAFT_EXTEND_GRAPH` | `True` — environ.py:1795 | Draft-extend may use a CUDA-graph backend (`speculative/eagle_worker_v2.py:232`) | already ON | `…=0` = control | P1 |
| `SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS` | `8192` — environ.py:2374; **script sets 8192 (line 72)** | Prefill: no-prefix prefill up to N tokens attends densely via flash-attn-v100 (`qwen_sparse_attn_backend.py:72,1763`). Also serves as the gate for `_can_use_sm70_dense_decode` (`:2452`), which additionally needs `max_context_len <= N` — never true at `--context-length 262144` | keep 8192; prefill/TTFT lever, not decode | `SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=0 …` | P2 |
| `SGLANG_SM70_QSA_GROUPED_PREFILL` | `False` — environ.py:1800 | Grouped Page4 QSA chunk-prefill via out-of-tree Flash-V100 bindings (`_decode_cuda.py:147-163`, `_grouped_qsa.py:59`) | prefill-only | `SGLANG_SM70_QSA_GROUPED_PREFILL=1 …` | P2 |
| `SGLANG_SM70_MOE_PREFILL_2K` | `False` — environ.py:1780 | Two measured 2K-prefill Marlin MoE stages with 64-row CTAs (`moe/moe_wna16_marlin.py:69`) | **prefill only** — zero decode effect | `SGLANG_SM70_MOE_PREFILL_2K=1 …` | P2 |
| `SGLANG_SM70_PROMPT_LOOKUP` | `False` — environ.py:1953 | Draft from a verified prompt-suffix lookup instead of the MTP model (`eagle_worker_v2.py:1519`); gate at `:1532` requires `--max-running-requests 1` + non-adaptive | **incompatible** with `--max-running-requests 3` | would need `--max-running-requests 1` | P2 |
| `SGLANG_SM70_FP16_MOE` | `False` — environ.py:1369 | TurboMind FP16 MoE runner (`layers/quantization/sm70_fp16_moe.py:52`) | N/A — this checkpoint is NVFP4, not FP16 MoE | — | P2 |
| `SGLANG_SM70_FP8_BACKEND` | `"marlin"` — environ.py:1349 | Block-FP8 linear backend (`sm70_turbomind_fp8.py:264`) | N/A — FP8 checkpoints only | — | P2 |
| `SGLANG_SM70_FP8_PREFILL_BACKEND` | `"auto"` — environ.py:1351 | FP8 prefill GEMM backend | N/A | — | P2 |
| `SGLANG_SM70_FP8_PREFILL_MIN_TOKENS` | `3920` — environ.py:1354 | Token count that switches FP8 prefill to dense TurboMind GEMM | N/A | — | P2 |
| `SGLANG_SM70_FP8_PREFILL_EXACT_DENSE` | `True` — environ.py:1356 | Exact-shape dense prefill kernel | N/A | — | P2 |
| `SGLANG_SM70_FP8_DECODE_QPN8` | `True` — environ.py:1359 | QPN8 W8A16 kernels for FP8 decode, M<=8 | N/A (FP8 only) | — | P2 |
| `SGLANG_SM70_FP8_QPN8_FASTDEC` | `True` — environ.py:1361 | Word-parallel QPN8 decoder | N/A | — | P2 |
| `SGLANG_SM70_FP8_QPN8_FUSED_GATE` | `True` — environ.py:1363 | Fuse QPN8 gate/up with SiLU multiply at decode | N/A | — | P2 |
| `SGLANG_SM70_QPN8_BUILD_DIR` | `/tmp/sglang_qpn8_build` — environ.py:1365 | JIT build dir | N/A | — | P2 |
| `SGLANG_SM70_TURBOMIND_OPS_PATH` | `None` — environ.py:1367 | Prebuilt TurboMind SM70 extension path | N/A | — | P2 |
| `SGLANG_SM70_GLM_NVFP4_GEMV` / `_MOE_DECODE` | `False` — environ.py:1810-1811 | GLM-5.3 NVFP4 HMMA decode instead of Marlin (`glm_nvfp4_moe_decode.py:203-229`) | N/A — GLM only | — | P2 |
| `SGLANG_SM70_GLM_NVFP4_BUILD_DIR` | `~/.cache/…` — environ.py:1812 | JIT dir | N/A | — | P2 |
| `SGLANG_SM70_VISION_ATTENTION` | `False` — environ.py:1806 | V100 ViT encoder attention | prefill/vision only | — | P2 |
| `SGLANG_SM70_WARMUP_FLUSH_CACHE` | `False` — environ.py:1818 | `POST /flush_cache` after warmup | startup only | — | P2 |
| `SGLANG_SM70_FORCE_FP16` | `True` — environ.py:2014 | Holds mamba/GDN state+conv in fp16 (`configs/mamba_utils.py:112`, `load_model_utils.py:135,159`); script also sets `=1` (line 67) | already ON (fast on Volta) | `SGLANG_SM70_FORCE_FP16=0 …` = slower control | P1 |
| `SGLANG_CUSTOM_ALLREDUCE_ALGO` | `None` = size heuristic — environ.py:1741; **script sets `1stage` (line 62)** | Restricts the AR to one algorithm family: `1stage`/`oneshot` vs `2stage`/`twoshot` (`custom_all_reduce_v2.py:95-117`; C++ side `aot/csrc/allreduce/custom_all_reduce.cuh:717`) | **`1stage`** on a mesh where two-stage is pathological — already set | `SGLANG_CUSTOM_ALLREDUCE_ALGO=2stage …` = slower control | P1 |
| `SGLANG_CUSTOM_AR_ALLOW_PCIE` | `False` — environ.py:1747 | Opt-in custom AR on >2 GPUs without a full NVLink clique; one-shot push only (`custom_all_reduce_utils.py:521-526`, `custom_all_reduce_v2.py:265,864`) | **keep OFF** on this NVLink mesh (script comment lines 63-64 says the same) | `SGLANG_CUSTOM_AR_ALLOW_PCIE=1 …` = expected regression | P2 |
| `SGLANG_CUSTOM_AR_PCIE_MAX_BYTES` | `131072` — environ.py:1748 | 128 KiB cap for the PCIe one-shot push | only with the flag above | — | P2 |

### 1b. `SGLANG_V100_*` / `SGLANG_SM70_*` read directly from `os.environ` (NOT in `environ.py` — invisible to `envs`, no validation, no doc)

These are the ones you cannot find by grepping `environ.py`; they matter for attention/linear-attn.

| knob | default | effect | fast setting | how to A/B | P |
|---|---|---|---|---|---|
| `SGLANG_V100_NVFP4_MOE_DECODE` | `"1"` — `kernels/ops/moe/sm70_nvfp4_moe_decode.py:25` | Master kill-switch for the whole native NVFP4 decode kernel (all row caps are moot when 0) | ON (fast) | `SGLANG_V100_NVFP4_MOE_DECODE=0 …` = control | P1 |
| `SGLANG_V100_NVFP4_MOE_BUILD_DIR` | `/tmp/sglang_sm70_nvfp4_moe` — `:68` | JIT build dir | — | — | P2 |
| `SGLANG_V100_DECODE_CUDA` | `"1"` — `tilelang_fa_v100/_decode_cuda.py:42` | Enables the hand-written SM70 CUDA long-context decode partial | ON, but **gated off by fp16 KV** (see area 5/6) | `SGLANG_V100_DECODE_CUDA=0 …` = control | P2 |
| `SGLANG_V100_DECODE_CUDA_BUILD_DIR` | `/tmp/sglang_sm70_longctx_decode` — `:72` | JIT build dir | — | — | P2 |
| `SGLANG_V100_DECODE_TOKENS_PER_SPLIT` | `64` — `_kernels_paged_decode.py:23` (constant `DECODE_MIN_TOKENS_PER_SPLIT`, `:18`) | **TileLang** split-K granularity for grouped decode; smaller = more splits = more CTAs | try `32` (only affects the non-QSA TileLang partial) | `SGLANG_V100_DECODE_TOKENS_PER_SPLIT=32 …` | P2 |
| `SGLANG_V100_DECODE_BLOCK_M` | `64` — `_kernels_paged_decode.py:337` | TileLang decode tile height; `threads = block_m*4` (`:340`); must be 16/32/64 | 64 on V100 (measured default) | `SGLANG_V100_DECODE_BLOCK_M=32 …` | P2 |
| `SGLANG_V100_VERIFY_TOKENS_PER_SPLIT` | `128` — `_kernels_paged_verify.py:26` (`VERIFY_SM_TARGET=80` at `:27`; read at `:32`; splits at `:354`) | Split-K granularity for the **verify** attention kernel — this *is* on the MTP hot path. The comment at `:22-25` records that the former 1024 left only 8 CTAs active at a 4K prefix on an 80-SM V100 | 128 is the tuned value; `64` is the remaining sweep point (more splits, smaller slices) | `SGLANG_V100_VERIFY_TOKENS_PER_SPLIT=64 …` | P1 |
| `SGLANG_V100_GROUPED_DECODE` / `SGLANG_V100_LONG_DECODE_XQA` | `1` — `flash_attn_v100_backend.py:123-124` | flash-attn-v100 grouped/long-decode kernels | already ON | `…=0` = control | P2 |
| `SGLANG_V100_PREFILL_D256_SPLIT_KV` | `"auto"` — `_paged_adapter.py:320` | Prefill split-KV | prefill only | — | P2 |
| `SGLANG_V100_PREFILL_D256_LOGICAL_TAIL` / `_GATHER` | `1` — `_paged_adapter.py:74,81` | Prefill D256 tail/gather | prefill only | — | P2 |
| `SGLANG_V100_GDN_COLUMN_GROUPS_PER_BLOCK` | unset (1/2/4/8) — `linear/kernels/gdn_chunked_tilelang.py:1384` | GDN prefill tile | prefill only | — | P2 |
| `SGLANG_V100_GDN_FULL_TILELANG` | `1` — `gdn_chunked_tilelang.py:1725` | GDN prefill TileLang | prefill only | — | P2 |
| `SGLANG_V100_GDN_VALUE_BLOCK` / `_K_REUSE_MODE` | `32` / `3` — `gdn_chunked_tilelang.py:1731-1736` | GDN prefill tile | prefill only | — | P2 |
| `SGLANG_V100_GDN_DIRECT_MAX_TOKENS` | `448` — `gdn_tilelang.py:398` | Threshold for the direct (non-chunked) GDN prefill | prefill only | — | P2 |
| `SGLANG_V100_GDN_PACKED_V_DIRECT` | `1` — `gdn_chunked_tilelang.py:1559` | GDN packed-V path | prefill only | — | P2 |
| `SGLANG_V100_FP8_GATHER_THREADS` | `128` — `_paged_adapter.py:530` | FP8 KV gather thread count | prefill/gather path | — | P2 |
| `SGLANG_V100_BFLA_*` | off / `1.0` — `_paged_adapter.py:152-167,277` | BFLA approximate block-sparse attention | **approximate — not a correctness-neutral knob** | do not use | — |
| `SGLANG_SM70_FP16_MOE_FAST_SELECTOR` | C++ default — `kernels/jit/csrc/sm70_fp16_moe_gemm.cu:71` | FP16 MoE selector | N/A (NVFP4 ckpt) | — | P2 |
| `SGLANG_SM70_FP16_MOE_TUNE_ROWS` | C++ default — `sm70_fp16_moe_gemm.cu:102` | FP16 MoE row tuning | N/A | — | P2 |

---

## 2. CUDA graphs (decode)

Defaults are `None` on the CLI and get materialized by resolution hooks; **do not assume the CLI
default is the effective default**.

| knob | default | effect (hot path) | fast setting | how to A/B | P |
|---|---|---|---|---|---|
| `--cuda-graph-bs-decode` | `None` → generated list (`cuda_graph_hook.py:534-566`): no-spec `[1,2,4,8,12]+…`; **spec** `[1..8]+[10..32 step2]+…` | Explicit list of **request** buckets to capture for decode/verify; `max_bs = max(capture_bs)` (`decode_cuda_graph_runner.py:419-424`) | script pins `1 2 3` (line 224) to match `--max-running-requests 3`; add `4` if you raise the request cap | `FLASH_NEXT_EXTRA_ARGS='--cuda-graph-bs-decode 1 2 3 4'` (last-wins, script line 275-278) | **P0** |
| `--cuda-graph-max-bs-decode` | `None` → 160/256/512 by GPU size (`memory_hook.py:157-172`), then **clamped to `--max-running-requests`** (`memory_hook.py:184-190`) | Upper bound used to generate the bucket list | script pins `3` (line 223) | `FLASH_NEXT_EXTRA_ARGS='--cuda-graph-max-bs-decode 4'` | P1 |
| graph vs eager selection | — | `model_runner.py:2111-2134`: `can_run_graph = mode.is_cuda_graph() and runner.can_run_graph(fb)`; the runner then requires `cuda_graph_bs <= self.max_bs` (`decode_cuda_graph_runner.py:997-1001`) **else eager**; it also refuses when `spec_info.num_tokens_per_req != captured_req_width` (`:972-977`) | keep spec width uniform | — (structural) | — |
| `--disable-prefill-cuda-graph` | `False` — `exec_.py:692-695`; **script sets it (line 230)**; prefill default is `breakable` on CUDA (`cuda_graph_config.py:119-123`) | Prefill-phase graph only; no decode-path effect | decode-neutral; re-enabling buys TTFT, costs memory (script comment lines 225-229) | remove line 230 / `FLASH_NEXT_EXTRA_ARGS='--cuda-graph-backend-prefill breakable'` | P2 |
| `--cuda-graph-backend-decode` | `None` → `full` (`cuda_graph_config.py:149-151`) | Decode capture backend: `full` / `breakable` / `disabled` (`exec_.py:643-650`, folded at `cuda_graph_hook.py:68-69`) | `full` is the decode default and the only sensible choice for a dense decode graph | `FLASH_NEXT_EXTRA_ARGS='--cuda-graph-backend-decode breakable'` | P1 |
| `--cuda-graph-backend-prefill` | `None` → `breakable` on CUDA (`cuda_graph_config.py:119-123`) | Prefill capture backend | see `--disable-prefill-cuda-graph` | `FLASH_NEXT_EXTRA_ARGS='--cuda-graph-backend-prefill disabled'` | P2 |

### What happens if you raise decode graph batch sizes beyond `--max-running-requests`

Three separate guards, in order:

1. `memory_hook.py:184-190` — when `--cuda-graph-max-bs-decode` is **not** explicitly set, the
   generated `max_bs` is silently **lowered** to `--max-running-requests`.
2. `base_cuda_graph_runner.py:84,112-120` — `num_max_requests = req_to_token_pool.size` (which is
   sized from `max_running_requests`, `mem_cache/kv/build.py:93`). If the largest requested bucket
   exceeds it, `num_max_requests` is **appended** to the list, and then every `bs > num_max_requests`
   is **filtered out** (`:119`). So `--cuda-graph-bs-decode 1 2 3 8` with `--max-running-requests 3`
   captures `1 2 3` and silently drops 8.
3. `decode_cuda_graph_runner.py:997-1001` — replay requires `cuda_graph_bs <= max_bs`;
   blocks above it replay **eagerly** (slower, and it also drops Graph capture of attention
   metadata).

Consequence: raising decode buckets above `--max-running-requests` is a **no-op at best and a
silent eager fallback at worst** — raise `--max-running-requests` first, then the buckets. It is
*not* a way to buy padding room for the MoE row cap: the MoE row cap is per-token-row
(`bs * num_draft_tokens`) and is controlled by the `SGLANG_SM70_*MOE*` env flags, not by the graph
bucket list.

---

## 3. Scheduler knobs

| knob | default | effect on the decode loop | fast setting | how to A/B | P |
|---|---|---|---|---|---|
| `--stream-interval` | `1` — `fields/serving.py:262-265` | Tokens buffered before a stream chunk is emitted; `1` does a detokenize+send per step (`scheduler.py:569` → `output_streamer.py:588-609`) | **20** on a throughput run (cookbook recipe uses 20) — fewer host wakeups per decode step | `FLASH_NEXT_EXTRA_ARGS='--stream-interval 20'` | **P0** |
| `--max-running-requests` | `None` — `fields/schedule.py:31-34`; script pins `3` (line 141) | Sets `req_to_token_pool.size` → **clamps every decode graph bucket** (`base_cuda_graph_runner.py:84,119`); also caps the decode batch the scheduler may form (`scheduler.py:1277-1290`) | 3 is the measured aggregate peak **on the Marlin fallback path**; re-measure 4 once verify rows <=16 hit the native kernel | `FLASH_NEXT_EXTRA_ARGS='--max-running-requests 4'` (+ `--cuda-graph-bs-decode 1 2 3 4`) | **P0** |
| `--schedule-policy` | `"fcfs"` — `fields/schedule.py:86-102`; consumed at `scheduler.py:548` → `managers/schedule_policy.py:184-351` | Only orders the **waiting** queue (prefill admission); does not touch a formed decode batch | `lpm` (longest prefix match) → more radix hits → fewer prefill tokens stealing decode time | `FLASH_NEXT_EXTRA_ARGS='--schedule-policy lpm'` | P1 |
| `--num-continuous-decode-steps` | `1` — `fields/schedule.py:200-203` | **DEAD KNOB in this tree**: declared and registered (`field_order.py:68`) but has **no consumer anywhere** under `python/sglang/srt` (`git grep num_continuous_decode_steps` → only field + docs) | no effect; do not spend a restart on it | — | P2 |
| `--disable-overlap-schedule` | `False` (= overlap **on**) — `fields/schedule.py:193-199` | Overlaps CPU scheduling with the GPU step (`scheduler.py:2160 event_loop_overlap`); off = `event_loop_normal` (`:2125`), adding a host bubble per step | keep **off** (overlap on) | `FLASH_NEXT_EXTRA_ARGS='--disable-overlap-schedule'` = expected regression | P2 |
| `--enable-mixed-chunk` | `False` — `fields/schedule.py:208-211`; gated at `scheduler.py:1454` (`chunked_prefill_size is not None`) | Allows prefill chunks and decodes in the same forward | off is right for a 3-request decode-bound run; enabling adds prefill rows to decode steps | `FLASH_NEXT_EXTRA_ARGS='--enable-mixed-chunk'` | P2 |
| `--max-prefill-tokens` | `16384` — `fields/schedule.py:71-81`; script pins `4096` (line 149) | Caps prefill tokens per forward (`scheduler.py:1279,1470`); a big prefill chunk stalls every decoding stream for its duration | 4096 is the script's OOM-safe value; it is a **TTFT/OOM lever, not a decode-rate lever** | `FLASH_NEXT_EXTRA_ARGS='--max-prefill-tokens 8192'` (OOM risk per script comment) | P2 |
| `--chunked-prefill-size` | `None` → 8192/16384/4096 by GPU (`memory_hook.py:151-180`) | Max tokens per chunked-prefill chunk | script pins 4096 (line 143) | `FLASH_NEXT_EXTRA_ARGS='--chunked-prefill-size 8192'` | P2 |
| `--schedule-conservativeness` | `1.0` — `fields/schedule.py:139-142` | Scales the retraction threshold via `new_token_ratio_tracker.py:24` — only decides when running requests get **retracted** under cache pressure | 1.0 is fine at 3 requests; higher = less retraction, more preemption stalls | `FLASH_NEXT_EXTRA_ARGS='--schedule-conservativeness 1.3'` | P2 |

---

## 4. Speculative decoding

Script block: `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4` (lines 262-266).

| knob | default | effect (hot path) | fast setting | how to A/B | P |
|---|---|---|---|---|---|
| `--speculative-num-draft-tokens` | `None` — `fields/spec.py:66-69`; script `4` | **Sets the verify row width per request** (`spec_utils.py:99-104`) and the draft-extend width; together with `bs` it decides which NVFP4 kernel runs (see §0) | 4 at np<=1; at np=3 it needs `BATCHED_MOE` (16 rows) to stay on the fast kernel | `FLASH_NEXT_EXTRA_ARGS='--speculative-num-draft-tokens 8'` (with `BATCHED_MOE32`) | **P0** |
| `--speculative-num-steps` | `None` — `fields/spec.py:58-61`; script `3` | Draft steps per round; `leaves = steps+1` must equal `num_draft_tokens` for a chain | 3 with 4 leaves is the validated shape; more steps = more draft FLOPs per accepted token | `FLASH_NEXT_EXTRA_ARGS='--speculative-num-steps 7'` (paired with `-num-draft-tokens 8`) | P1 |
| `--speculative-eagle-topk` | `None` — `fields/spec.py:62-65`; script `1` | Draft branching; also the **draft-decode** row width (`spec_utils.py:95-96`) | **must stay 1** for every SM70 fast path: `MTP_LOCAL_TOP1` demands topk=1 (`sm70_draft_top1.py:20`), the dense-GEMV/fusion kernels assume 1 row, and `is_no_spec_infer_or_topk_one` (`model_override_base.py:402-408`) keys off it | keep 1 (do not A/B) | P1 |
| `--speculative-use-rejection-sampling` | `False` — `fields/spec.py:117-119` | Correctness/sampling-quality switch (requires topk=1); adds per-token draws on the verify path | off (fast) — leave off unless you need it | `FLASH_NEXT_EXTRA_ARGS='--speculative-use-rejection-sampling'` | P2 |
| `--speculative-adaptive` | `False` — `fields/spec.py:235-238`; consumers `eagle_worker_v2.py:566,1532,1544`, `spec_info.py:276` | Varies `num_steps` with the acceptance rate → **verify rows change per step**, which breaks the uniform `num_tokens_per_req` the decode graph needs (`decode_cuda_graph_runner.py:972-977` → eager fallback) | off (fast) | `FLASH_NEXT_EXTRA_ARGS='--speculative-adaptive'` = expected regression | P2 |
| `--speculative-attention-mode` | `"prefill"` — `fields/spec.py:129-136` | Backend family used for draft-extend + target-verify attention | `prefill` is the documented default; `decode` is the alternative | `FLASH_NEXT_EXTRA_ARGS='--speculative-attention-mode decode'` | P2 |

### MTP tree shape vs the NVFP4 fast kernels (row-by-row)

`max_rows()` (`sm70_nvfp4_moe_decode.py:28-42`) is checked **in this order**: `BATCHED_MOE32`→32,
`BATCHED_MOE`→16, `MTP_WIDE_MOE`→8, else 4. Both the routed-expert GEMM (`marlin.py:153`) and the
512→10 router top-k (`topk.py:1314`) use the same number.

| config | rows (np=1 / 2 / 3 / 4) | stays inside the fast kernel? |
|---|---|---|
| draft 4, no env | 4 / 8 / 12 / 16 | np=1 only |
| draft 4 + `MTP_WIDE_MOE` (8) | 4 / 8 / 12 / 16 | np<=2 |
| draft 4 + `BATCHED_MOE` (16) | 4 / 8 / 12 / 16 | np<=4 |
| draft 8 + `BATCHED_MOE32` (32) | 8 / 16 / 24 / 32 | np<=4 |
| draft 8 + `MTP_WIDE_MOE` (8) | 8 / 16 / 24 / 32 | np=1 only |
| draft 16 + `BATCHED_MOE32` (32) | 16 / 32 / 48 / 64 | np<=2 |

The kernels themselves are row-independent (`grid = rows * topk`, `sm70_nvfp4_moe_decode.py:29-34`);
the cap only exists to keep dispatch inside measured shapes. Above the cap the fallback is
`fused_marlin_moe` (`marlin.py:199+`), i.e. block-padded grouped GEMM on a 10-row-wide problem.

Also note the **draft-extend** width is `num_draft_tokens` (`spec_utils.py:97-98`) and the
**draft-decode** width is `eagle_topk` (`:95-96`), so a 4-draft chain drafts at 1 row/request/step
and extends at 4 rows/request — the extend pass hits the same cap whenever the MTP head routes
through the same Marlin/NVFP4 runner.

### `SGLANG_SM70_MTP_LOCAL_TOP1` applicability (checked against this deployment)

`speculative/sm70_draft_top1.py:19-52` requires: topk==1 ✅ (script 265), `hot_token_id is None`,
no DP attention ✅ (`--dp-size 1`), no LoRA ✅, `pp_size==1` ✅, fp16 lm_head ✅ (script `--dtype
float16`, `SGLANG_SM70_FORCE_FP16=1`), model is `Qwen4ExpForCausalLMMTP`, full draft vocabulary.
All satisfied → this flag is live for this config, and it replaces a **full-vocabulary all-gather
per draft step** with a two-float all-gather.

---

## 5. Attention / linear attention

| knob | default | effect (hot path) | fast setting | how to A/B | P |
|---|---|---|---|---|---|
| `--attention-backend` | `None` — `fields/exec_.py:225-232`; script `tilelang_fa_v100` (line 98) | Selects the attention kernel family; on sm70 the QSA/sparse dispatch lives in `qwen_sparse_attn_backend.py` | `tilelang_fa_v100` (only V100 backend with the QSA specializations) | fixed; no useful A/B | P1 |
| `--linear-attn-decode-backend` | `None` → `--linear-attn-backend` = `"triton"` (`exec_.py:572-585`); script `triton` (line 100) | GDN decode kernel (`layers/attention/linear/utils.py:95`) | script default; `tilelang` is the untested alternative on Volta | `FLASH_NEXT_EXTRA_ARGS='--linear-attn-decode-backend tilelang'` | P1 |
| `--linear-attn-prefill-backend` | `None` → `"triton"` (`exec_.py:586-592`); script `tilelang` (line 99) | GDN prefill/extend kernel | script's `tilelang` + `SGLANG_V100_GDN_*` tiles | `FLASH_NEXT_EXTRA_ARGS='--linear-attn-prefill-backend triton'` | P2 |
| `--linear-attn-verify-backend` | `None` → follows decode (`exec_.py:593-599`) | GDN **speculative verify** kernel — this is on the MTP hot path and is separate from decode | unset (= triton today); `tilelang` is worth one restart | `FLASH_NEXT_EXTRA_ARGS='--linear-attn-verify-backend tilelang'` | P1 |
| `--enable-linear-replayssm` | `False` — `exec_.py:602-611` | Buffered output-only linear-attn decode (~1.2-1.5x at **batch >= 64**) | **incompatible**: requires `--mamba-radix-cache-strategy no_buffer`, script uses `extra_buffer` (line 231); also a large-batch win we cannot reach at np<=4 | n/a | P2 |
| `QSA_DECODE_TARGET_CTAS` | `160` — **module constant**, `tilelang_fa_v100/_decode_cuda.py:31` | Split count for the QSA decode CUDA op: `max_splits = ceil(160 / batch)` (`:209`) | 160 matches the 80-SM V100 (2 CTAs/SM); tuning needs a **code edit** | not A/B-able from the script | P1 |
| `QSA_DECODE_TOKENS_PER_SPLIT` | `32` — **module constant**, `_decode_cuda.py:32` | Tokens per split in the QSA decode op (`:227`); also the combine kernel's `selected_tokens` stride (`:247`) | 32; tuning needs a code edit | not A/B-able from the script | P2 |
| QSA native combine gate | — | `_decode_cuda.py:232-248`: native combine only when `1 <= batch <= 4` and `heads==6, dim==256, max_splits<=160` — above batch 4 it falls to the TileLang combine | keep `--max-running-requests <= 4` if you want the native combine | structural | P1 |
| QSA sparse decode gate | — | `qwen_sparse_attn_backend.py:2417-2446`: decode/verify-paged, sm70, fp16, `q.shape[1:]==(6,256)`, KV dtype in `{e5m2, fp16}` — **accepts fp16**, which is why the fp16-KV decision does not cost the sparse decode kernel | works with the current fp16 KV | structural | — |
| QSA **dense** decode gate | — | `qwen_sparse_attn_backend.py:2448-2470` requires `page_size == 16` **and** `k_buffer.dtype == float8_e5m2` **and** `max_context_len <= SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS` | unreachable at `--context-length 262144` / fp16 KV | — | P2 |
| TileLang split-K heuristic | — | `_kernels_paged_decode.py:17-18,330-340`: `DECODE_SM_TARGET=80`, `max_splits = ceil(80/(batch*heads_kv))`, `min_tokens_per_split=64`; verify variant `_kernels_paged_verify.py:27,354` with `VERIFY_SM_TARGET=80` | `SGLANG_V100_VERIFY_TOKENS_PER_SPLIT=32` doubles verify splits — the one lever that is a plain env var | `SGLANG_V100_VERIFY_TOKENS_PER_SPLIT=32 …` | P1 |
| CUDA long-context partial gate | — | `_paged_adapter.py:487-511`: needs sm70 + **fp8 KV and specifically e5m2** + `heads==6*heads_kv` + `dim==256` + `page_size==16`; otherwise logs `SM70 CUDA decode partial disabled: kv_dtype(...)` **every call** | off under the current fp16-KV decision (see area 6) | structural | P2 |
| `SGLANG_SM70_QSA_COMBINE` | `True` — environ.py:1794 | See §1a: native combine only for batch<=4 | already ON | `…=0` = control | P1 |
| `--mamba-ssm-dtype` / `SGLANG_MAMBA_SSM_DTYPE` | `None` → model dtype (`environ.py:2010`); script sets `float16` (line 66) | GDN state dtype; fp16 is required for the Volta TileLang path | fp16 (already set) | `SGLANG_MAMBA_SSM_DTYPE=bfloat16` = expected slower | P2 |
| `--mamba-radix-cache-strategy` | script `extra_buffer` (line 231) | Mamba checkpoint buffering strategy | `extra_buffer` is what the deployment runs; `no_buffer` unlocks ReplaySSM but not at this batch size | — | P2 |

---

## 6. Memory / layout

| knob | default | effect | fast setting | how to A/B | P |
|---|---|---|---|---|---|
| `--mem-fraction-static` | `None` → auto by GPU (`fields/schedule.py:27-30`); script `0.86` (line 131) | Trades KV pool size against prefill headroom; **does not change per-step decode cost** (pool size is not on the hot path once the batch fits) | 0.86 per the script's OOM analysis; raising it buys pool tokens, not tok/s | `FLASH_NEXT_MEM_FRACTION=0.84` (env the script reads, line 131) | P2 |
| `--page-size` | `None` → model/platform default; **serve script does not pin it** (no `--page-size` in lines 90-258). The qwen3_5 override only forces 64-vs-1 on **sm100** (`arg_groups/model_overrides/qwen3_5.py:26-49`), so it does not apply here | Page granularity of the KV pool and page table. Page 16 is what the SM70 dense/CUDA decode kernels were validated on (`_decode_cuda.py:30`); page 64 is the high-throughput recipe value | 16 **only if** you also switch KV to e5m2 — page size alone does not unlock any kernel (`_paged_adapter.py:504`) | `FLASH_NEXT_EXTRA_ARGS='--page-size 16 --kv-cache-dtype fp8_e5m2'` (pair with a lower `--mem-fraction-static`) | P2 |
| `--kv-cache-dtype` | `auto` — script line 108 (resolves to the model's fp16) | fp16 KV = 2 B/elem; e5m2 = 1 B/elem (~2x pool) and is the **only** dtype that unlocks the CUDA long-context partial (`_paged_adapter.py:502`) | fp16 is the documented decision of record (script lines 101-107: no decode-speed cost on the sparse path, better MTP acceptance). e5m2 is the alternative, **not** an unambiguous win | `FLASH_NEXT_EXTRA_ARGS='--kv-cache-dtype fp8_e5m2'` + lower `--mem-fraction-static` | P1 |
| `--enable-hierarchical-cache` | `False` — `fields/memory.py:100-103`; script ON (line 159) | Adds the host/disk tiers. Per **idle** scheduler iteration `check_hicache_events()` does a TP-group all-reduce (`drain_storage_control_queues`), which the script's `--sleep-on-idle` (line 219) works around; it is not free on the decode path either | keep ON for prefix reuse, but this is the first thing to measure if per-step overhead looks high | `FLASH_NEXT_EXTRA_ARGS='--disable-hierarchical-cache'` (diagnostic only — expect a big TTFT/cache-hit regression) | P1 |
| `--hicache-size` | `0` — `fields/memory.py:115-118`; script `8` (line 198) | Host pool GB **per rank**, applied to **both** KV and Mamba pools (script comment lines 160-166) | 8 per the script's sizing note | `FLASH_NEXT_HICACHE=8` (env, line 198) | P2 |
| `--hicache-write-policy` | `"write_through"` — `fields/memory.py:123-129`; script `write_back` (line 208) | Disk write policy; write_through was measured to saturate the SATA array (script lines 168-186) and stall read-back | `write_back` (already set — fast) | `--hicache-write-policy write_through` = expected disaster | P2 |
| `--hicache-mem-layout` | `None` (`fields/memory.py:137+`); script `page_first` (line 206) | Host pool layout; Mamba host pool accepts only `page_first`/`page_first_direct` | already correct | — | P2 |
| `--mamba-full-memory-ratio` | `None` → 0.9 (`fields/schedule.py:220-227`); script `0.2` (line 232) | Share of the KV-sized pool given to mamba state; too low throttles `max_running_requests` | 0.2 is what the script validated at np=3 | `FLASH_NEXT_EXTRA_ARGS='--mamba-full-memory-ratio 0.3'` | P2 |
| `--max-mamba-cache-size` | `None` — `fields/schedule.py:216-219`; script `20` (line 142) | Hard slot cap on the mamba cache — an upper bound on concurrent mamba checkpoints, hence on effective concurrency | 20 is >> 3 requests, so not binding today; matters only if you raise np or the draft window | `FLASH_NEXT_EXTRA_ARGS='--max-mamba-cache-size 40'` | P2 |
| `--sleep-on-idle` | `False` (upstream default) — script ON (line 219) | Idle CPU only; no decode-path cost, wakes immediately on a new request | ON (already, avoids ~7.5 cores of spin) | remove line 219 = CPU regression | P2 |
| `--enable-metrics` | `False`; script ON (line 251) | Prometheus counters on the batch path | keep **constant** across the A/B campaign (it is per-step instrumentation; changing it mid-campaign confounds results) | do not toggle | P2 |
| `--ple-offload-embedding` | script ON (line 255) | Keeps the 51 GB PLE n-gram table in host memory (GPU OOM otherwise) | required, not a perf choice | — | P2 |

---

## 7. Recommended experiment list — the 12 P0 experiments

Rules: one restart each (~12 min), each experiment **layers on the previous winners** and is
reverted on regression. Keep `--enable-metrics` and the rest of the launch line fixed while scoring
anything else. Score with the same 128-token decode workload that produced 74 tok/s, 3 runs, median.
`mtp` mode is assumed; env vars go in front of the script, CLI flags go via
`FLASH_NEXT_EXTRA_ARGS` (appended last, so they win — script lines 275-278).

These are the 12 experiments the campaign treats as **P0**, i.e. worth a restart each.
Four of them (#8 `MTP_WIDE_MOE`, #9 `BATCHED_MOE32`, #11 draft-token widening, #12 `lpm`) are
P1-marked in the tables above as *knobs*; they are promoted here because they are cheap, they
disambiguate a P0 result, or they gate a later experiment — not because they are expected to
out-pay #1-#7.

| # | experiment | exact line | why it should pay |
|---|---|---|---|
| 1 | stream interval 20 | `FLASH_NEXT_EXTRA_ARGS='--stream-interval 20' bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp` | zero-risk host-side win: 1 detokenize+send per step → 1 per 20; safest possible first restart |
| 2 | native MoE for 8/12-row verify | `SGLANG_SM70_BATCHED_MOE=1 FLASH_NEXT_EXTRA_ARGS='--stream-interval 20' bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp` | biggest structural win: at np=2,3 the verify rows (8,12) move from `fused_marlin_moe` to the native NVFP4 kernel for **both** the router top-k and the expert GEMM |
| 3 | batched fusions | `SGLANG_SM70_BATCHED_MOE=1 SGLANG_SM70_BATCHED_FUSIONS=1 FLASH_NEXT_EXTRA_ARGS='--stream-interval 20' bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp` | same 8/12-row shapes in the shared-expert/projection fusions, which otherwise fall back to per-row calls |
| 4 | shared-expert gate at 4 rows | `… SGLANG_SM70_MTP_SHARED_GATE=1 …` | the np=1 verify is exactly 4 rows; today the shared-expert sigmoid gate only fuses at 1 row |
| 5 | local top-1 draft head | `… SGLANG_SM70_MTP_LOCAL_TOP1=1 …` | removes a full-vocabulary all-gather **per draft step**; all preconditions (topk=1, TP4, no LoRA/DP-attn, fp16) hold |
| 6 | 16 push CTAs on the TP4 AR | `… SGLANG_SM70_AR_PUSH16=1 …` | decode reductions are small messages; 16 resident push blocks measured faster than the default on a 4-way NVLink clique |
| 7 | TP top-1 for greedy batches | `… SGLANG_V100_GREEDY_TP_TOP1=1 …` | one exact top-1 exchange instead of a full-vocab logits all-gather. **Only if the workload is strictly greedy**; skip if sampling/logprobs are used |
| 8 | 8-row cap in isolation | `… SGLANG_SM70_MTP_WIDE_MOE=1 …` | cheap sanity point: confirms the cap (not something else) is what changes the numbers, and covers the 8-row draft-extend |
| 9 | 32-row cap | `… SGLANG_SM70_BATCHED_MOE32=1 …` | makes np=4 (16 rows) and any 8-draft tree (24 rows) land on the native kernel; prerequisite for #10 and #11 |
| 10 | re-measure the np peak | `… FLASH_NEXT_EXTRA_ARGS='--stream-interval 20 --max-running-requests 4 --cuda-graph-bs-decode 1 2 3 4 --cuda-graph-max-bs-decode 4'` | the script's "3 peaks, 4 falls" was measured **on the Marlin fallback**; with rows 4-16 native, the aggregate peak may move to 4 |
| 11 | deeper MTP tree | `… FLASH_NEXT_EXTRA_ARGS='… --speculative-num-steps 7 --speculative-num-draft-tokens 8'` | more accepted tokens per verify; at np=3 that is 24 rows — only viable because #9 raised the cap to 32 |
| 12 | prefix-aware scheduling | `… FLASH_NEXT_EXTRA_ARGS='… --schedule-policy lpm'` | more radix hits → less prefill work interleaved into the decode stream (agentic sessions re-hit the same prefix every turn) |

Ordering rationale: **expected payoff per restart, cheapest and least risky first** — a pure host
knob (#1), then the single mechanical unlock (#2, #3, #4 are three independent gates on the same
8/12-row shapes), then the two biggest collective-removal wins (#5, #6), then a conditional win (#7)
and a sanity point (#8), then the cap raise that unlocks the higher-concurrency and deeper-tree
experiments (#9 → #10, #11), and finally a scheduling change (#12) that trades a small amount of
policy risk for cache locality.

### Expanded commands (copy-paste, from the repo root)

Each line assumes every prior winner is kept (revert on regression). `…` in the table above means
the env prefix accumulated here.

```bash
# 1 — stream interval 20
FLASH_NEXT_EXTRA_ARGS='--stream-interval 20' \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp

# 2 — + native NVFP4 MoE for 8/12-row verify (cap 4 -> 16)
SGLANG_SM70_BATCHED_MOE=1 \
FLASH_NEXT_EXTRA_ARGS='--stream-interval 20' \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp

# 3 — + batched FP16 fusions (8..32 rows)
SGLANG_SM70_BATCHED_MOE=1 SGLANG_SM70_BATCHED_FUSIONS=1 \
FLASH_NEXT_EXTRA_ARGS='--stream-interval 20' \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp

# 4 — + shared-expert gate fusion at 2/4 rows
SGLANG_SM70_BATCHED_MOE=1 SGLANG_SM70_BATCHED_FUSIONS=1 SGLANG_SM70_MTP_SHARED_GATE=1 \
FLASH_NEXT_EXTRA_ARGS='--stream-interval 20' \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp

# 5 — + local top-1 draft head (no full-vocab draft all-gather)
SGLANG_SM70_BATCHED_MOE=1 SGLANG_SM70_BATCHED_FUSIONS=1 SGLANG_SM70_MTP_SHARED_GATE=1 \
SGLANG_SM70_MTP_LOCAL_TOP1=1 \
FLASH_NEXT_EXTRA_ARGS='--stream-interval 20' \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp

# 6 — + 16 push CTAs on the TP4 all-reduce
SGLANG_SM70_BATCHED_MOE=1 SGLANG_SM70_BATCHED_FUSIONS=1 SGLANG_SM70_MTP_SHARED_GATE=1 \
SGLANG_SM70_MTP_LOCAL_TOP1=1 SGLANG_SM70_AR_PUSH16=1 \
FLASH_NEXT_EXTRA_ARGS='--stream-interval 20' \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp

# 7 — + TP top-1 exchange for strictly-greedy batches (skip if sampling/logprobs)
SGLANG_SM70_BATCHED_MOE=1 SGLANG_SM70_BATCHED_FUSIONS=1 SGLANG_SM70_MTP_SHARED_GATE=1 \
SGLANG_SM70_MTP_LOCAL_TOP1=1 SGLANG_SM70_AR_PUSH16=1 SGLANG_V100_GREEDY_TP_TOP1=1 \
FLASH_NEXT_EXTRA_ARGS='--stream-interval 20' \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp

# 8 — + 8-row cap in isolation (sanity point; superseded by 9)
SGLANG_SM70_BATCHED_MOE=1 SGLANG_SM70_BATCHED_FUSIONS=1 SGLANG_SM70_MTP_SHARED_GATE=1 \
SGLANG_SM70_MTP_LOCAL_TOP1=1 SGLANG_SM70_AR_PUSH16=1 SGLANG_V100_GREEDY_TP_TOP1=1 \
SGLANG_SM70_MTP_WIDE_MOE=1 \
FLASH_NEXT_EXTRA_ARGS='--stream-interval 20' \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp

# 9 — + 32-row cap (np=4 verify = 16 rows; 8-draft tree = 24 rows)
SGLANG_SM70_BATCHED_MOE=1 SGLANG_SM70_BATCHED_FUSIONS=1 SGLANG_SM70_MTP_SHARED_GATE=1 \
SGLANG_SM70_MTP_LOCAL_TOP1=1 SGLANG_SM70_AR_PUSH16=1 SGLANG_V100_GREEDY_TP_TOP1=1 \
SGLANG_SM70_BATCHED_MOE32=1 \
FLASH_NEXT_EXTRA_ARGS='--stream-interval 20' \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp

# 10 — + re-measure the concurrency peak at 4 requests
SGLANG_SM70_BATCHED_MOE=1 SGLANG_SM70_BATCHED_FUSIONS=1 SGLANG_SM70_MTP_SHARED_GATE=1 \
SGLANG_SM70_MTP_LOCAL_TOP1=1 SGLANG_SM70_AR_PUSH16=1 SGLANG_V100_GREEDY_TP_TOP1=1 \
SGLANG_SM70_BATCHED_MOE32=1 \
FLASH_NEXT_EXTRA_ARGS='--stream-interval 20 --max-running-requests 4 --cuda-graph-bs-decode 1 2 3 4 --cuda-graph-max-bs-decode 4' \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp

# 11 — + deeper MTP tree (8 leaves -> 24 verify rows at np=3, needs cap 32)
SGLANG_SM70_BATCHED_MOE=1 SGLANG_SM70_BATCHED_FUSIONS=1 SGLANG_SM70_MTP_SHARED_GATE=1 \
SGLANG_SM70_MTP_LOCAL_TOP1=1 SGLANG_SM70_AR_PUSH16=1 SGLANG_V100_GREEDY_TP_TOP1=1 \
SGLANG_SM70_BATCHED_MOE32=1 \
FLASH_NEXT_EXTRA_ARGS='--stream-interval 20 --max-running-requests 4 --cuda-graph-bs-decode 1 2 3 4 --cuda-graph-max-bs-decode 4 --speculative-num-steps 7 --speculative-num-draft-tokens 8' \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp

# 12 — + prefix-aware scheduling
SGLANG_SM70_BATCHED_MOE=1 SGLANG_SM70_BATCHED_FUSIONS=1 SGLANG_SM70_MTP_SHARED_GATE=1 \
SGLANG_SM70_MTP_LOCAL_TOP1=1 SGLANG_SM70_AR_PUSH16=1 SGLANG_V100_GREEDY_TP_TOP1=1 \
SGLANG_SM70_BATCHED_MOE32=1 \
FLASH_NEXT_EXTRA_ARGS='--stream-interval 20 --max-running-requests 4 --cuda-graph-bs-decode 1 2 3 4 --cuda-graph-max-bs-decode 4 --schedule-policy lpm' \
  bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp
```

---

## 8. Verification notes / non-findings (things that look like knobs but are not)

1. **`--num-continuous-decode-steps` is dead** in this tree — declared at `fields/schedule.py:200-203`
   and registered in `field_order.py:68`, but no reader exists under `python/sglang/srt`. A/B-ing it
   cannot move tok/s.
2. **`QSA_DECODE_TARGET_CTAS` / `QSA_DECODE_TOKENS_PER_SPLIT` are module constants**
   (`_decode_cuda.py:31-32`), not env vars. The env-var-adjacent knobs
   (`SGLANG_V100_DECODE_TOKENS_PER_SPLIT`, `SGLANG_V100_DECODE_BLOCK_M`) belong to the **TileLang**
   partial, which the QSA sparse decode path does not use. Tuning the QSA split counts requires a
   code edit.
3. **The CUDA long-context decode partial is inert under the current KV choice** — it needs
   `float8_e5m2` KV *and* `page_size == 16` (`_paged_adapter.py:502-505`); the deployment runs fp16 KV
   with an unpinned page size.
4. **`--speculative-eagle-topk 1` is load-bearing**, not a tunable: `MTP_LOCAL_TOP1`, the 1-row
   fusion/GEMV shapes, and `is_no_spec_infer_or_topk_one` all assume it.
5. **`SGLANG_SM70_HC_*` / `SGLANG_SM70_HC_COMBINE_NORM` / `SGLANG_SM70_FP16_MOE` /
   `SGLANG_SM70_FP8_*` / `SGLANG_SM70_GLM_*`** do not apply to Qwen3.8-Flash-Next NVFP4 (they are
   DSV4.1-hyperconnection, FP16-MoE, block-FP8 and GLM paths respectively). Listed for completeness.
6. **`SGLANG_SM70_MOE_PREFILL_2K` and `SGLANG_SM70_QSA_*_PREFILL*` are prefill-only** — they can
   improve TTFT but cannot raise decode tok/s.
7. **`--cuda-graph-bs-decode` above `--max-running-requests` is silently dropped**
   (`base_cuda_graph_runner.py:112-120`), so it is not a lever for the MoE row cap.
8. Branch/checkout caveat: all of the above was collected on `sm70/stack`. Since 2026-11-11 `main`
   *is* that stack (ten slices on `2c5b680aa6`), so the greps work on either; this note is kept
   because it explains why the paths were branched when the inventory was written.

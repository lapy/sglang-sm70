# Qwen3.8-Flash-Next-NVFP4 on 4×V100-SXM2-32GB (TP4) — performance archaeology of three SGLang forks

Consolidated, evidence-first report. Every claim carries `repo/path:line-line`.
Read-only: no repo file was modified (see §0.3 for one incidental `git gc`).

Workspace root for all relative paths: `C:\Users\vlapy\Documents\deepseek-harness\default-workspace\`

---

## 0. How to verify (read this before you diff anything)

### 0.1 Refs actually read

| Fork | Ref read | Note |
|---|---|---|
| A `sglang-sm70` | **`sm70/stack`** for `docs/v100/*` + `scripts/*`; **`main`** for `SM70.md` | `.git` is `blob:none` partial — never run `git ls-tree -r -l` here |
| B `sglang-sxm2` | `qwen38-kv-qsa-port-20261110` (checked out) | worktree intact, files on disk |
| C `sglang-V100` | `qwen38-kv-qsa-optimization-20261102` (`HEAD` = `2a4c37ef61`) | **worktree is empty** (all paths staged-deleted) — read via `git show` only |

### 0.2 Pitfalls that will waste your time

- **A is not checked out on `sm70/stack` — and that is the biggest finding in this report.** `git -C sglang-sm70 rev-parse --abbrev-ref HEAD` → `main`. `docs/v100/` does **not exist on `main`**; it exists only on `sm70/*` branches (`git -C sglang-sm70 ls-tree -r --name-only sm70/stack -- docs/v100` → 6 files). `SM70.md` also differs per branch: `main` = 4094 B (66 lines), `sm70/stack` = 1560 B (23 lines). Worse, **`main` contains none of the SM70 python layer**: `git grep -ohE 'SGLANG_SM70_[A-Z0-9_]+' main -- python | sort -u` → **0** vars (vs **43** on `sm70/stack`), and `git ls-tree -r --name-only main -- python | grep -i sm70` → **0** files (vs **81** on `sm70/stack`); `git diff --stat main sm70/stack -- python` → **339 files, +52,570 / −1,429**. Anyone who builds and serves from A's checkout gets upstream SGLang with no Volta kernels, and A's launcher env block becomes inert. See §5.12.
- **C's worktree is empty.** `git -C sglang-V100 ls-tree -r --name-only HEAD` → 7,616 files; `Get-ChildItem sglang-V100` → `.git` only. Every C citation below must be read as `git -C sglang-V100 show <ref>:<path>`.
- **A and B `docs/v100/` are byte-identical.** MD5 of `Qwen3.8-Flash-Next.md` and of `scripts/serve_qwen38_flash_next_nvfp4_v100.sh` match across A@`sm70/stack` and B@`qwen38-kv-qsa-port-20261110`. Any "A vs B docs" difference is *not* in those files — it is in B's `benchmark/` tree (which A does not have) and in A's `SM70.md`.
- **There is no documented ~80 tok/s sxm2 reference.** Grep for `80 tok/s|80.\d tok/s|about 80` across all three refs returns only prefill numbers (`3,080 tok/s`). See §1.6 for the closest documented values and why 80 is not verifiable from these repos.

### 0.3 Side effect (disclose)

Extracting C's blobs triggered git's automatic housekeeping in `sglang-V100` (`Auto packing the repository for optimum performance`, twice). That repacks `.git` objects only; tracked content, refs and the index are untouched, and the worktree was already empty. No repo file was created, edited or deleted.

---

## 1. Measured numbers

Legend for `proto`: `nat` = real prose/code text, `rand` = random-token `bench_serving` corpus, `filler` = `ignore_eos` padding, `chat` = six hand-written chat prompts, `copy` = copy-heavy code edit, `25K/2K-out` = 25,000 input / 2,048 output.

### 1.1 A — `sglang-sm70` (all of A's *own* Qwen measurements)

A has exactly **three** numbers of its own, all 128-token spot checks on `/generate`; everything else in A is inherited doc text (§5.1).

| # | Value | Metric | Config | Source |
|---|---|---|---|---|
| A1 | **84 tok/s** decode | decode, `ignore_eos`, 128 out | TP4, MTP on, `--mem-fraction-static 0.86`; **measured while output was degenerate** (prompt echo then zeros) | `sglang-sm70/SM70.md:3-5` (main) |
| A2 | **75 tok/s** (MTP) / **54 tok/s** (target-only) | decode, 128 out | same box, MTP on/off; after dense-marlin repack fix | `sglang-sm70/SM70.md:37` (main) |
| A3 | **73.7 tok/s** | decode, 128 out | clean-tree rebuild of `sm70/stack` (#12/#15/#16/#18/#19), MTP on | `sglang-sm70/SM70.md:59` (main) |
| A4 | 19.89 GB/rank, 128 GB total | VRAM at load | same | `sglang-sm70/SM70.md:3` (main) |
| A5 | 116.7 tok/s @3 streams vs 108.8 @4 | aggregate decode | TP4, `--max-running-requests 3`, temperature 0, real text | `sglang-sm70/scripts/serve_qwen38_flash_next_nvfp4_v100.sh:133-141` |
| A6 | 67 / 53 / 43 / 38 tok/s | per-stream decode at np=1/2/3/4 | same | `…/serve_qwen38_flash_next_nvfp4_v100.sh:136-137` |
| A7 | TTFT 656 ms → 2.5 s | TTFT at np=3 → np=4 | same | `…/serve_qwen38_flash_next_nvfp4_v100.sh:134-136` |
| A8 | accept **2.47** mean | tokens per target forward, 1,015 decode samples | MTP 3/4 | `…/serve_qwen38_flash_next_nvfp4_v100.sh:246-248` |
| A9 | 47 → 74 tok/s | decode "rises monotonically with accept length" | quoted from A's own log history; exact A/B framing is ambiguous in the comment | `…/serve_qwen38_flash_next_nvfp4_v100.sh:246-248` |
| A10 | accept **3.98 of 4** (random) vs **2.6–2.7** (real text) | acceptance, same server | random-token corpus is degenerate → draft predicts it almost perfectly | `…/serve_qwen38_flash_next_nvfp4_v100.sh:138-140` |
| A11 | KV pool **402,912 → 352,864** tokens (−12%) | KV pool, `--mem-fraction-static` 0.88 → 0.86 | E5M2-class pool (see §5.4 for why the dtype is unclear) | `…/serve_qwen38_flash_next_nvfp4_v100.sh:123-125` |
| A12 | `available_gpu_mem` **3.89 → 4.54 GB** | prefill headroom, 0.88 → 0.86 | same | `…/serve_qwen38_flash_next_nvfp4_v100.sh:122-124` |
| A13 | **694k tokens/rank** device KV pool; `--hicache-size 5` clears it by ~23% and measured **racy** | KV pool + host-pool assert | `--hicache-size 8` chosen for ~2× margin | `…/serve_qwen38_flash_next_nvfp4_v100.sh:160-166`, `:188-197` |
| A14 | **92–119 ms** → **20 ms** pre-prefill wait | disk-tier lookup latency with/without metadata cache, at 50–68k files | 600-token tool result | `…/serve_qwen38_flash_next_nvfp4_v100.sh:81-87` |
| A15 | **344 tok/s** (3 tok/s at 55k tokens) | disk readback | `write_through` on a saturated SATA array | `…/serve_qwen38_flash_next_nvfp4_v100.sh:168-186` |
| A16 | 30 → 678 MB/s | cold `dd` during `write_through` vs 5 s after | same array | `…/serve_qwen38_flash_next_nvfp4_v100.sh:171-175` |
| A17 | 134 MiB alloc failed | OOM at 8k GDN prefill chunk | after KV pool + hicache GPU overhead | `…/serve_qwen38_flash_next_nvfp4_v100.sh:144-149` |
| A18 | 3.89→4.54 GB, 0.88 crash at ~2 min vs 0.86 clean 5 min | beyond-spec 32k × np=8 stress | 0.88 OOM under load | `…/serve_qwen38_flash_next_nvfp4_v100.sh:117-130` |

### 1.2 A/B shared doc — `docs/v100/Qwen3.8-Flash-Next.md` (identical bytes in A and B)

Config block for all of §1.2: **2026-10-02**, `serve_qwen38_flash_next_nvfp4_v100.sh mtp`, 4× V100-SXM2-32GB, TP=4, MTP **3 steps / 4 draft tokens**, `--mem-fraction-static 0.86`, `--max-running-requests 3`, temperature 0, **KV pool 332,736 tokens**, context 262,144, thinking **on** (`sglang-sm70/docs/v100/Qwen3.8-Flash-Next.md:41-43`).

| # | Value | Metric | Proto | Source |
|---|---|---|---|---|
| S1 | 103.5 / 102.5 / 113.4 / 139.8 / 123.5 / 111.5 | decode tok/s, six chat prompts (code / refactor / explain / math / agent step / German prose) | chat | `…/Qwen3.8-Flash-Next.md:47-54` |
| S2 | **112.8** | decode tok/s, all six | chat | `:55` |
| S3 | 2.27 / 2.18 / 2.40 / 2.97 / 2.63 / 2.36 | accept length per prompt | chat | `:47-54` |
| S4 | **~47 target forwards/s** at one stream | target-forwards rate | chat | `:57` |
| S5 | **121.0** total / 67.9 per stream @2 streams; **133.6** / 56.3 @3 | decode | chat | `:57` |
| S6 | **~2.1** accept → **~99 tok/s** | decode, "low end" | filler `ignore_eos` | `:57` |
| S7 | 9.28 s / 40.10 s / 61.86 s | cold first-turn TTFT at 32,349 / 128,349 / 190,353 ctx | agent | `:61-65` |
| S8 | 409/377/341 ms; 510/487/502 ms; 610/575/548 ms | tool-turn TTFT at 32k / 128k / 190k | agent | `:61-65` |
| S9 | **3,500 tok/s @32k, 3,080 tok/s @190k** | cold prefill | agent | `:67` |
| S10 | 61.86 s cold / 606 ms warm; subagent 5.57 s / 420, 647 ms; main 629 ms; subagent2 5.89 s / 403, 650 ms; main 550 ms | TTFT, main session + 2 subagents | agent | `:69-77` |

### 1.3 B — `benchmark/` measured record (the real experiment log)

**B-B1 MTP first pass** — E5M2 KV, 8,192-token prefill chunks, 262,144 ctx, 256 forced greedy tokens, one request, seeds 20260829 (`sglang-sxm2/benchmark/qwen38_flash_next_nvfp4_mtp_v100_20260829/README.md:8-16`):

| # | Input | TTFT | Prefill | TPOT | Decode | Accept |
|---|---|---|---|---|---|---|
| B1 | 1,000 | 496.133 ms | 2,015.6 tok/s | 14.579 ms | **68.59 tok/s** | 2.7875 |
| B2 | 25,000 | 5,541.488 ms | 4,511.4 tok/s | 13.094 ms | **76.37 tok/s** | 3.2250 |
| | | | | | | `…/qwen38_flash_next_nvfp4_mtp_v100_20260829/README.md:50-53` |

**B-B2 first target-only optimization pass** — 256 out, seed 20260830, 1 discarded warmup + 3 cold requests, mem-fraction not stated, E5M2 KV (`…/qwen38_nvfp4_v100_20260907/README.md:17-30`):

| # | Input | main prefill | cand prefill | gain | main decode | cand decode | gain |
|---|---|---|---|---|---|---|---|
| B3 | 1,000 | 2,538.2 | 2,645.2 | +4.2% | 61.375 | 64.565 | +5.2% |
| B4 | 8,192 | 5,335.3 | 5,512.7 | +3.3% | 60.472 | 63.412 | +4.9% |
| B5 | 25,000 | 4,800.2 | 4,907.5 | +2.2% | 60.278 | 63.189 | +4.8% |
| | | | | | | | `…/qwen38_nvfp4_v100_20260907/README.md:26-30` |

**B-B3 sustained 25K target-only sweep** — 25,000 in / 2,048 out, 4×V100 SXM2, TP4, FP16 act/Mamba, NVFP4 experts, **E5M2 KV**, one request, graph bs 1, ctx 262,144, **8,192 prefill chunks** (no MTP) (`…/qwen38_nvfp4_v100_70tps_20260907/README.md:5-9`):

| # | Development point | Median prefill | Median decode | Slowest window |
|---|---|---|---|---|
| B6 | First optimization pass | 4,872.1 | 63.206 | 63.092 |
| B7 | + native dense GEMVs | 4,903.6 | 67.646 | 67.561 |
| B8 | + shared-expert/GDN fusions | 4,922.1 | 69.662 | 69.558 |
| B9 | + remove shared input copies | 4,906.1 | 70.036 | 69.897 |
| B10 | + parallel QSA metadata / PLE GEMVs | 4,900.8 | 70.108 | 70.016 |
| B11 | Rejected HC block-size change | 4,928.9 | 70.070 | 69.924 |
| B12 | Stable router sort / PLE gate | 4,893.4 | 70.067 | 69.973 |
| B13 | **Final: parallel QSA split merge** | **4,876.3** | **71.977** | **71.824** |
| | | | | `…/qwen38_nvfp4_v100_70tps_20260907/README.md:19-28` |
| B14 | seeds 20260830/31/32 → 71.997 / 71.977 / 71.874 decode; 4,876.3 / 4,851.0 / 4,893.7 prefill | same protocol | | `:38-42` |
| B15 | 25,000 in / **8,192 out** (≈33K ctx, seed 20260832) → **71.833** decode, windows 71.711–71.947, prefill 4,946.0 | same | | `:53` |
| B16 | median sustained decode **+13.9%** over the first optimization pass | 25K | | `:3` |
| B17 | same-protocol 256-out comparison vs main: 1K 61.375→73.460 (+19.7%), 8K 60.472→72.261 (+19.5%), 25K 60.278→71.965 (+19.4%); prefill −3.7% / +3.2% / +1.3% | | | `:70-74` |
| B18 | 1K prefill recheck median 2,425.1 (2,209.4–2,448.7) — **no 1K prefill gain established** | | | `:76-81` |

**B-B4 the MTP-3/4 kernel campaign (selected = `mtp34_vector`)** — 4×V100 SXM2, TP4, FP16 act/state, NVFP4 experts, **E5M2 KV**, 1 req, ctx 262,144, **8,192 prefill chunks**, EAGLE top-k 1, 3 steps / 4 verify, MTP expert weights FP16 (`…/qwen38_nvfp4_v100_mtp_20260908/README.md:94-100`):

| # | Measurement | Value | Source |
|---|---|---|---|
| B19 | Selected MTP, **natural** decode @1K/8K/25K/70K | 115.99–118.02 / 116.24–118.12 / 108.44–119.06 / 113.88–116.93 | `:10-15` |
| B20 | Selected MTP, **random** decode, same lengths | 122.05–139.03 / 121.62–137.93 / 148.08–172.02 / 138.90–176.80 | `:10-15`, `:332-337` |
| B21 | Natural prefill @1K/8K/25K/70K | 2,971–3,157 / 5,455 / 4,681–4,687 / 4,625–4,627 | `:10-15` |
| B22 | Lowest windows | natural 105.36, random 84.30; initial MTP random sweep 61.49–103.35 | `:17-20` |
| B23 | Median estimated round cost | **38.9 → 21.6 ms (−44.4%)** | `:20-23` |
| B24 | **target-only regression** | **74.28–74.32 @1K**, **73.55–73.58 @25K** (2,048 out); min windows 73.53; prefill 4,916–4,923 | `:25-28` |
| B25 | target-only after QSA accumulator rewrite | 72.649/72.639 @25K, 73.360/73.345 @1K | `:266-269` |
| B26 | target-only after vector scale loads | 73.581/73.549 @25K (windows 73.565/73.527, prefill 4,916/4,923), 74.323/74.280 @1K | `:342-345` |
| B27 | 8K-output sustained 25K, earlier | 71.83 | `:28-29` |
| B28 | MTP round time, 4-token candidate | 23.5–24.3 ms; natural accept **2.5–2.7** tokens/round | `:137-139` |
| B29 | `mtp12_gdn` (2 draft tokens) | round **18.8–19.3 ms**, natural only **90.7–95.0 tok/s**; 3 of 4 short responses differ → not selected | `:138-142` |
| B30 | `mtp23_native` (2 steps / 3 tokens) | round **19.0–19.7 ms**, natural 115.71/121.65, 114.69/117.50, 109.35/124.79, 112.98/119.72; 3 of 4 short answers differ → isolated patch only | `:353-368` |
| B31 | `mtp34_gdn` (4-token validated) natural | 105.30–113.62 (all lengths), lowest window 97.49–99.44 | `:130-135` |
| B32 | `mtp34_layout_resume` random | 1K 130.95/113.67, 8K 153.00/135.43, 25K 140.09/158.57, 70K 168.54/131.95; round 22.7–23.6 ms | `:207-209` |
| B33 | `mtp34_qsa` round cost | 22.2–22.9 ms; natural 99.00–106.80 lowest window | `:255-264` |
| B34 | `mtp34_vector` round cost | 21.6–22.3 ms | `:327-330` |
| B35 | GPU graph times (CUDA events, ~1,866 replays/rank) | target verify **17.68 ms**, draft loop 2.34–2.41 ms, draft extend 1.37–1.38 ms | `:305-312` |
| B36 | CPU graph submission | 131 µs (large) + 37/28 µs ⇒ **≈0.2 ms/round** | `:197-201` |

**B-B5 Docker vs host parity** — mem fraction **0.80**, E5M2 KV, 8,192 chunks, graph bs 1, one request (`…/qwen38_nvfp4_v100_docker_v4_20260908/README.md:91-96`):

| # | Input | host prefill | docker prefill | host decode | docker decode |
|---|---|---|---|---|---|
| B37 nat | 1,000 | 2,952 | 3,269 | 116.81 | 117.55 |
| B38 nat | 8,192 | 5,439 | 5,422 | 117.49 | 120.85 |
| B39 nat | 25,000 | 4,729 | 4,693 | 117.86 | 119.84 |
| B40 nat | 70,000 | 4,630 | 4,611 | 118.89 | 118.11 |
| | | | | | `…/qwen38_nvfp4_v100_docker_v4_20260908/README.md:47-52` |
| B41 | Docker natural request range 113.99–125.68, min window 107.31; host 112.40–124.02, min 105.59 | | | | `:54-57` |
| B42 target-only | 1,000 / 25,000 | 2,784 / 4,889 | 2,704 / 4,872 | **74.284 / 73.555** | **74.265 / 73.548** (−0.027% / −0.010%) | `:59-65` |
| B43 random MTP | accepted/round 2.791 (1K), 3.246 (8K), 3.072 (25K), 2.822 (70K); round 21.533 / 21.750 / 21.773 / 22.008 ms | | | | `:68-73` |
| B44 | host slowest 70K random = 109.20 tok/s (min window 61.60) | | | | `:80` |
| B45 | multimodal recheck: 1K 116.81→116.62 (−0.16%), 25K 117.86→119.74 (+1.60%) | | | | `…/qwen38_nvfp4_v100_multimodal_20260909/README.md:30-33` |

**B-B6 concurrency with MTP** — `--mem-fraction-static 0.84`, `--max-total-tokens 540672`, `--max-running-requests 4`, `--max-mamba-cache-size 24`, 2,048 chunks, graph bs 1 2 4, E5M2 (`…/qwen38_nvfp4_v100_full_context_mtp_concurrency_20260925/README.md:10-29`):

| # | Reqs | in/out each | Aggregate out tok/s | Source |
|---|---|---|---|---|
| B46 | 4 | 1,024/512 | **199.66** | `:41` |
| B47 | 4 | 8,192/1,024 | **120.02** | `:42` |
| B48 | 4 | 65,536/512 | **25.25** | `:43` |
| B49 | 2 | 261,120/512 | **5.83** (523,184 live KV tokens; ~172 aggregate once both decoding) | `:44`, `:46-49` |
| B50 | accepted length across those rows | 2.82–3.83 | | `:49` |
| B51 | startup | `max_num_reqs=4`, 24 Mamba slots, **540,672 E5M2 KV slots**, **1.80 GB/GPU draft weights** | | `:31-33` |

**B-B7 front page** (`sglang-sxm2/README.md`): Qwen3.8-Flash-Next **~113 tok/s one stream (102–140), MTP ~2.4 tok/step; 121 total @2 streams, 134 @3** (`:47-58`).

### 1.4 C — `sglang-V100` measured record

**C-C1 FP16 vs E5M2 KV, MTP 3/4** — 4×V100 SXM2 NVLink, TP4, 1 req, **mem-fraction 0.83**, ctx 262,144, three repeats, saved token IDs at 1,000/8,192/25,000/70,000, 512 outputs (`…/benchmark/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:37-49`):

| # | Input | prose E5M2 / FP16 | code-edit E5M2 / FP16 |
|---|---|---|---|
| C1 | 1,000 | 107.8 / **119.0** | 172.4 / **185.8** |
| C2 | 8,192 | 110.7 / **115.2** | 174.3 / **183.5** |
| C3 | 25,000 | 108.6 / **113.2** | 173.9 / **182.7** |
| C4 | 70,000 | 104.1 / **111.4** | 170.9 / **179.8** |
| | | | `…/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:113-121` |
| C5 | FP16 KV reduces the pool **554,576 → 277,280 tokens** at 0.83 | | `:122-124` |
| C6 | target-only 0.84 gate needs 0.84, not 0.83 (0.83 → only 259,328 slots) | | `:176-179` |
| C7 | target-only range **72.9–74.9 tok/s** | | `:174` |

**C-C2 MTP baseline + phase profile + prompt-lookup** — FP16 compute, **FP8 E5M2 KV**, 262,144 ctx, one request, MTP 3/4, multimodal on, 512 outputs, 126 GiB host RAM (`…/benchmark/qwen38_nvfp4_v100_spec_20261102/README.md:16-34`):

| # | Measurement | Value | Source |
|---|---|---|---|
| C8 | prose decode @1K/8K/25K/70K | 107.6 / 112.2 / 107.7 / 110.6 | `:82-87` |
| C9 | code-edit decode @1K/8K/25K/70K | 172.3 / 174.1 / 175.4 / 168.8 | `:82-87` |
| C10 | tokens/cycle prose vs code | 2.33–2.46 vs 3.76–3.84 | `:82-87` |
| C11 | observed cycle | **21.7–22.3 ms** | `:89` |
| C12 | target-only | 74.68 / 73.85 / 73.55 / 72.80 (prose, 1K→70K) | `:199-207` |
| C13 | cycle phases (8K code, lookup off, 34 rounds) | full **21.81**, draft 2.69, verify **17.61**, target forward 17.29, state commit 0.184, draft extend 1.51 ms | `:142-146` |
| C14 | traced decode window | 489.40 ms window, 366.71 busy, 122.69 idle (**25.1%**) | `:153-156` |
| C15 | kernel sums in that window | MoE 72.56, GDN 24.37, **all-reduce 20.68**, QSA compute 14.48, QSA metadata 0.305, copies 24.57 ms | `:157-159` |
| C16 | profiler overhead | 64 code tokens / 18 rounds → 26.65 ms cycle | `:153-155` |

**C-C3 balanced v5.5 eight-request profile (the current C recommendation)**

| # | Measurement | Value | Source |
|---|---|---|---|
| C17 | Fixed text, 8 clients | v5.4 4-req **253.1** → v5.5 8-req **360.4 tok/s (+42.4%)** | `…/qwen38_concurrent_core_audit_20261104/README.md:3-7`, `:149` |
| C18 | SPEED natural-EOS, 8 clients | 227.5 → **307.9 (+35.3%)** | `:5`, `:151` |
| C19 | Fixed text 4 clients / SPEED 4 clients | 253.4→255.6 (+0.9%) / 228.1→229.5 (+0.6%) | `:148`, `:150` |
| C20 | Four independent 25K prompts | 53.3 → **61.8 (+15.8%)** | `:152` |
| C21 | Clients 1 / 2 / 4 | 116.1 / 155.8 / 255.8 tok/s; power 167.3 / 160.8 / 183.0 W per GPU | `:30-32` |
| C22 | Single-client follow-up | **116.1–117.3 end-to-end**, **121.2–121.4 decode** tok/s | `:143-144` |
| C23 | DCGM 4 vs 8 agents | SM 69.5→78.5%, occupancy 17.7→22.8%, tensor 1.7→2.6%, DRAM 36.2→42.8%, power 184.5→210.7 W; all clocks 1,530 MHz, no throttle | `:166-177` |
| C24 | Config | 8 requests, graphs 1–8, 48 Mamba slots, **mem-fraction 0.86**, **KV pool 277,280 tokens** | `:183-189` |
| C25 | queue test | 48 requests in **233.52 s**, ≥2,952.7 MiB free on GPU 0 | `:138-144` |

**C-C4 MTP width (C1–C4 screen)** — v5.5 baseline `23c879b38c`, FP16 KV, full target+draft vocab, 262,144 ctx, **277,280-token pool**, 48 Mamba, 2K chunks, 300 W limits (`…/qwen38_c14_strata_20261104/README.md:17-33`):

| # | Workload / clients | v5.5 | compact | wide8 (MTP-7/8) |
|---|---|---|---|---|
| C26 | Prose C1 | 114.8 | 115.1 | **84.3** |
| C27 | Prose C2 / C3 / C4 | 166.0 / 198.9 / 255.1 | 166.8 / 198.9 / 256.2 | 134.0 / 158.6 / 199.2 |
| C28 | Copy/edit C1 / C2 / C4 | 101.1 / 124.2 / 149.7 | 102.7 / 126.7 / 153.4 | **112.6** / **142.3** / 150.0 |
| | | | | `…/qwen38_c14_strata_20261104/README.md:53-61` |
| C29 | Restored v5.5 control (flags off) | Prose C1/C2/C3/C4 115.6/166.2/198.6/252.0; copy C1/C2/C4 101.0/126.5/153.3 | | `:72-80` |
| C30 | C1 per-stream decode | prose 119.0 / 119.2 / **87.2**; copy 181.4 / 182.0 / **218.1** | | `:88-92` |
| C31 | tokens per verification, C1 | copy **4.00** (MTP-3/4) vs **7.78** (MTP-7/8); prose 2.56 vs 3.04 | | `:93-96` |
| C32 | decode power | 169.7 W/GPU baseline vs 162.5 W wide8 | | `:98-104` |
| C33 | C8 fixed text | 360.5 / 365.0 (compact) vs 361.3 (restored) | | `:106-107` |
| C34 | SPEED | compact 119.7 @C1, 234.2 @C4; wide8 97.3 / 207.0; restored control 120.8 / 233.3 | | `:108-115` |
| C35 | wide8 headline | 216.5 / 219.7 vs **181.4** MTP-3/4 copy/edit (+20% decode, +11% end-to-end after 8K prefill) | | `:3-7` |

**C-C5 prefill/2K-chunk campaign (v5.1 → core optimization)**

| # | Measurement | Value | Source |
|---|---|---|---|
| C36 | normal request timing | 1K: TTFT 0.30 s, 118.9 tok/s, cycle 21.37 ms; 8K: 2.14 s, 114.2, 21.63; 70K: 19.70 s, 113.5, 22.18; 160K: ~46.9 s, ~109, ~22.9 | `…/qwen38_nvfp4_v100_core_audit_20261103/README.md:36-41` |
| C37 | 512-token copy-edit | ~182 tok/s, ~4 output tokens/cycle | `:50-54` |
| C38 | cycle phases @70K (53 cycles) | draft 3.23, verify forward 17.58, commit 0.19, other 0.15, draft extend 1.56, **cycle 22.73 ms** | `:83-92` |
| C39 | CPU submission | 7.57 ms, overlaps GPU (not additive) | `:94-96` |
| C40 | 70K prefill attribution | MoE 39.1%, QSA 23.0%, HC 16.7%, GDN 12.9%, collectives 6.3%, PLE 1.0% | `:58-69` |
| C41 | 8K prefill attribution | MoE 42.4%, QSA 18.0%, HC 18.0%, GDN 13.8% | `:77` |
| C42 | decode kernel categories | dense/dot 26.9%, MoE 17.4%, HC 15.5%, norm/pointwise 12.2%, GDN 7.6%, copies 5.0%, QSA 4.8%, collectives 4.6% | `:98-102` |
| C43 | DCGM 70K | prefill: engine 92.8%, SM 86.5%, occ 29.4%, tensor 16.8%, DRAM 26.5%; decode: 99.0 / 58.2 / 20.4 / **0.1** / 31.3 | `:115-121` |
| C44 | power 70K | prefill 244–273 W, decode 164–186 W; some prefill samples software-capped; decode none | `:131-133` |
| C45 | calibration | 1 GiB copy 787–790 GB/s; 8,192² FP16 GEMM 91.8–97.0 TFLOP/s; pinned H2D ~12.45 GB/s; GPU0→peer ~48.4 GB/s | `:137-142` |
| C46 | NVLink traffic 70K | ~5.6 GB/s/GPU/dir prefill, ~0.49 GB/s decode | `:149-152` |
| C47 | core optimization E2E | one median request per case: **103.78 → 98.95 s (−4.66%)**; prose 8K +3.7%, 25K +5.6%, 70K +5.9%, 160K +4.6%, code 8K +4.6%, 1K **−1.4%** | `…/qwen38_nvfp4_v100_core_optimization_20261103/README.md:14-28` |
| C48 | decode side of that round | SPEED geomean decode **+0.4% greedy / +0.7% sampled** | `:32-35` |
| C49 | 2K MoE geometry | 40 configs, selected **1.153×** geometric, ≥1.114× per layer, production 1.13–1.18×; full-model ≈ **4–5% lower long-ctx TTFT** | `:39-57` |
| C50 | shared gate 2/4 rows | **8.45 → 2.88 µs** | `:59-65` |
| C51 | grouped Page4 QSA prefill | planning+attention **2.29–2.64×**; rel L2 1.5e-4–1.8e-4; shape `[2048, 6, 256]` only | `:67-83` |
| C52 | cache/memory gates | 32,256 cached tokens reused; min free 2,100.7 MiB GPU0 / 2,570.4 others | `:101-103` |

**C-C6 v2 Docker concurrency (historical)** — E5M2 KV, 8,192 chunks, `--max-running-requests 16`, `--max-mamba-cache-size 80`, graphs `1 2 4 8 16`, mem-fraction 0.85 (0.82 for the c16 matrix) (`…/benchmark/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md:48-67`):

| # | Concurrency | target tok/s | MTP tok/s | delta | MTP TTFT | MTP TPOT | accept |
|---|---|---|---|---|---|---|---|
| C53 | 1 | 55.12 | **74.82** | +35.7% | 1,637.65 ms | 11.76 ms | 3.375 |
| C54 | 4 | 137.26 | 149.25 | +8.7% | 5,049.17 ms | 20.04 ms | 3.089 |
| C55 | 8 | 172.98 | 194.13 | +12.2% | 8,329.74 ms | 28.94 ms | 3.130 |
| C56 | 16 | **239.36** | 183.87 | **−23.2%** | 14,524.25 ms | 55.64 ms | 3.048 |
| | | | | | | | `…/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md:35-40` |
| C57 | 1K→25K single request | target 60.09 (416.03 s, TPOT 16.62) vs MTP **88.07** (283.86 s, TPOT 11.33, accept **3.493**) | | | | | `:21-27` |
| C58 | KV slots | 0.85 → 536,512; 0.82 → 221,616 | | | | | `:61-67` |

**C-C7 4-request full-context profile** — mem 0.84, `--max-total-tokens 540672`, mrr 4, Mamba 24, 2K chunks, graph bs 1 2 4 (`…/qwen38_nvfp4_v100_full_context_mtp_concurrency_20260925/README.md:10-29`): identical numbers to B46–B51 (c4 1k 199.66, c4 8k 120.02, c4 64k 25.25, c2 261k 5.83, accept 2.82–3.83).

**C-C8 vision** — 32×768×1024 JPEGs, 24,675 prompt tokens, 4 outputs (`…/qwen38_vision_v100_20261104/README.md:16-25`):

| # | Configuration | run 1 | run 2 | mean |
|---|---|---|---|---|
| C59 | Triton vision attention | 80.03 s | 79.94 s | **79.98 s** |
| C60 | SM70 Flash-V100 vision attention | 12.31 s | 11.92 s | **12.12 s** (6.60×) |
| C61 | isolated 3-image attn (3,072 patches, head-dim 72) | | | 238.56 → **3.29 ms**, rel L2 0.00038 | `:69-73` |
| C62 | text side, 4-client / 8-client fixed, SPEED 4/8 | baseline 253.15/248.49/221.22/224.49; selected 250.44/249.70/217.34/226.00; 6-req MTP-1/2 232.84/249.90/196.34/235.41 | | `:40-44` |

**C-C9 Docker v5 packaging** — 8 requests, FP16 KV, 2K prefill, 277,280 tokens, 48 Mamba, CPU MM preprocessing, native vision (`…/qwen38_docker_v5_20261105/README.md:8-11`):

| # | Clients | host | docker | diff |
|---|---|---|---|---|
| C63 | C1 | 115.4 | 112.9 | −2.1% |
| C64 | C4 | 255.6 | 250.4 | −2.0% |
| C65 | C8 | 360.9 | 356.5 | −1.2% |
| | | | | `…/qwen38_docker_v5_20261105/README.md:60-64` |
| C66 | 20-case natural-EOS SPEED @C8 | 306.8 aggregate | | `:66-67` |
| C67 | readiness | ~19 min (cached weights + first-use kernel compile); 32-image request 13.60 s; 48/48 docs 266.69 s, ≥2,720.7 MiB free | | `:40-51` |

### 1.5 Cross-model / cross-fork numbers that are easy to mix up (do not use for Qwen)

- GLM-5.3-Flash (8×V100, different model): MTP steps 2/3/4 → 117 / ~123 / **34** tok/s, i.e. 5-row verify "falls off the 4-row kernels" (`sglang-sm70/docs/v100/GLM-5.3-Flash.md:146`); `--max-running-requests 1` because "decode and verify kernels give each row the same result as batch 1 only while the whole launch has at most 4 rows" (`:149`).
- DeepSeek-V4.1-Flash (8×V100): 11.1 tok/s coding, ~7 prose, 2.8 sustained padding, cold prefill 117 tok/s (45.3 s TTFT), warm 558 tok/s (`sglang-sm70/docs/v100/DeepSeek-V4.1-Flash.md:53-61`).
- Qwen3.6/3.5/27B benchmarks in C/B `benchmark/qwen38_27b_*` are a **different model** (27B dense/hybrid, fp8) — e.g. 58.21 tok/s decode at 1,024 in target-only (`sglang-V100/benchmark/qwen38_27b_fp8_target_e5m2_v100_20260822/README.md:52-55`).

### 1.6 The "~80 tok/s sxm2 reference" is not in these repos

No file in A, B or C states ~80 tok/s for this model. Nearest documented sxm2-lineage numbers, all concurrency-1:

| Candidate for the "80" | Value | Why it might be it | Source |
|---|---|---|---|
| Early MTP-3/4, E5M2 KV, 8K chunks, 256 out | **76.37** @25K, 68.59 @1K | earliest MTP benchmark in the lineage | `sglang-sxm2/benchmark/qwen38_flash_next_nvfp4_mtp_v100_20260829/README.md:50-53` |
| v2 Docker, 8,192 in / 1,024 out, c1 | **74.82** (target 55.12) | "MTP at c1" figure quoted on C's model page | `sglang-V100/benchmark/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md:37` |
| v2 Docker, 1,000 in / 25,000 out, c1 | **88.07** | long-decode MTP | `:24` |
| Target-only, optimized | **73.5–74.9** | matches your current 74 | `sglang-sxm2/benchmark/qwen38_nvfp4_v100_mtp_20260908/README.md:25-28` |
| PCIe V100 contributor rows, MTP-3/4 | 92–97 (NCCL) / 106 (custom AR) | different box | `sglang-V100/docs/v100/models/qwen38-flash-next-nvfp4.md:172-175` |

**Conclusion:** treat "80" as unattributed. The reproducible sxm2-lineage numbers are 113–119 natural / 122–176 random at accept 2.5–2.9 with MTP, and 73.5–74.3 target-only — both with **E5M2 KV and 8,192-token chunks**, i.e. neither matches A's shipping config (FP16 KV, 4,096 chunks, hicache).

### 1.7 Live-box cross-check (outside the three repos — read this before acting on §5/§8)

A sibling analysis in the same output directory measured the running server on this box on 2026-11-11 and reports the 74-vs-80 premise is **stale** (`sm70-stack/perf-architecture-analysis.md:9-25`, not one of the three repos, so treat as context, not provenance):

| Live measurement | Value |
|---|---|
| `bench-baseline.json`, two-point method, 32→160 tokens | **107.63 tok/s** decode (105.7 / 107.6 / 108.6), TTFT 432 ms |
| same method while a second sweep was booting | 69.5 tok/s two-point, 60.2 incl. prefill |
| in-server `gen throughput`, `decode_log_interval=40` | **bimodal 62 ↔ 118 tok/s**, step to step |
| live spec accept length | **2.81 / 4** (rate 0.703), histogram over 57 verify steps `[0:13, 1:9, 2:10, 3:25]` |

Consequences for this report: (a) acceptance on the live box is **2.81**, i.e. the good regime, so A's 73.7 is *not* an acceptance collapse — it is more likely protocol/noise or the wrong tree being served (§5.12); (b) the box's own step-to-step spread (62↔118) exceeds the entire 74→80 gap, so **any A/B without ≥3 reps, median, idle box and a fixed two-point protocol will measure noise**; (c) my §5/§8 conclusions about config drift remain valid as *config* findings, but the specific "A ≈ target-only" arithmetic in §6.3 should be re-tested on a pinned protocol before you act on it.

---

## 2. Recommended settings with evidence

### 2.1 The three shipped recipes, side by side

| Knob | A (`sglang-sm70` @sm70/stack) | B (`sglang-sxm2`) | C (`sglang-V100`) |
|---|---|---|---|
| wrapper | `serve_…_v100.sh mtp` | same file, byte-identical | `serve_qwen38.sh` → `…_throughput.sh` → `…_mtp.sh` → `…_v100.sh` |
| `--max-running-requests` | **3** (`…/serve_qwen38_flash_next_nvfp4_v100.sh:141`) | 3 (same file) | **8** (`sglang-V100/scripts/serve_qwen38_flash_next_nvfp4_v100_throughput.sh:14`); 1 in the older single-request profile (`…_v100.sh:41`) |
| `--cuda-graph-*` | `--cuda-graph-max-bs-decode 3`, `--cuda-graph-bs-decode 1 2 3` (`:223-224`) | same | `--cuda-graph-bs 1 … 8` (`throughput.sh:15`); `1 2 4 8 16` at mrr 16 (`qsa_prefill_fix…:58-59`) |
| `--mem-fraction-static` | 0.86 (`:131`) | 0.86 | **0.86** (`throughput.sh:16`), **0.83** (mtp profile `…_mtp.sh:38`), 0.80 (base `…_v100.sh:39`), 0.82–0.85 (v2 docker) |
| `--kv-cache-dtype` | `auto` → FP16 (`:108`) | `auto` → FP16 | `auto` in mtp profile (`…_mtp.sh:13`); **base default `fp8_e5m2`** (`…_v100.sh:6`) |
| prefill chunk | **4096 / 4096** (`:143`, `:149`) | same | **2048 / 2048** (`…_mtp.sh:39-40`); 8192 in base (`…_v100.sh:42`) |
| `--max-total-tokens` | **absent** | absent | **277280 pinned** (`throughput.sh:17`); 540672 (4-req profile) |
| `--max-mamba-cache-size` | 20 (`:142`) | 20 | **48** (`throughput.sh:18`); 24 at mrr 4 |
| hier. cache | `--enable-hierarchical-cache --hicache-size 8 --hicache-mem-layout page_first --hicache-storage-backend file --hicache-write-policy write_back` (`:159-208`) | same | **absent from all Qwen profiles** |
| `--enable-multimodal` | **absent** | absent | **present** (`…_mtp.sh:41`) |
| `--mm-attention-backend` | absent | absent | `fa2_v100` (`throughput.sh:20`); `triton_attn` = prior path (`vision…:91`) |
| `--prefill-max-requests` | absent | absent | **1** (`throughput.sh:19`) |
| MTP flags | EAGLE, topk 1, 3 steps, 4 draft (`:259-268`) | same | same (`…_mtp.sh:42-47`) |
| kernel env defaults | DENSE_GEMV, QWEN_FUSIONS, QSA_DENSE_PREFILL_MAX_TOKENS=8192, FORCE_FP16 (`:67-72`) | same | same (`…_v100.sh:14-22`) **plus** `SM70_MTP_SHARED_GATE=1`, `SM70_MOE_PREFILL_2K=1`, `SM70_QSA_GROUPED_PREFILL=1` (`…_mtp.sh:25-27`) **plus** `BATCHED_MOE=1`, `BATCHED_MOE32=1`, `BATCHED_FUSIONS=1` (`throughput.sh:9-11`) **plus** `MM_PREPROCESS_ON_CPU=1` (`…_mtp.sh:18`) |
| explicitly-zeroed | — | — | `QSA_FUSED_GRAPH_METADATA=0`, `PROMPT_LOOKUP=0`, `MTP_WIDE_MOE=0` (`…_mtp.sh:22-24`); `MTP_LOCAL_TOP1=0`, `HC_COMBINE_NORM=0`, `AR_PUSH16=0` (`scripts/serve_qwen38.sh:52`) |

### 2.2 Recommendations ranked by evidence strength

**R1 — Concurrency ceiling 8 (aggregate) beats 3, if your metric is agent throughput.**
C: 8-client fixed text 253.1 → **360.4 tok/s (+42.4%)**, SPEED 227.5 → **307.9 (+35.3%)**, four independent 25K prompts +15.8%, while 4-client and 1-client are unchanged within screen noise (`sglang-V100/benchmark/qwen38_concurrent_core_audit_20261104/README.md:146-152`). Config: 8 requests, graphs 1–8, 48 Mamba slots, mem 0.86, 277,280-token pool (`:183-189`).
A's opposite choice (3) rests on a *per-stream* peak at 3 and a 4th stream "adding no throughput" (`sglang-sm70/scripts/serve_qwen38_flash_next_nvfp4_v100.sh:133-141`) — note A's very next sentence concedes a 4th stream *does* add aggregate on random text. C's 8-client data is aggregate, A's is single-run per-stream: the two are not measuring the same thing (§5.3).

**R2 — KV dtype FP16 (`auto`), not `fp8_e5m2`.**
C's controlled 3-repeat A/B: FP16 wins on **every** prompt and length — prose +10.4%/+4.1%/+4.2%/+7.0%, code +7.8%/+5.3%/+5.1%/+5.2% (`sglang-V100/benchmark/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:113-121`). Cost: pool 554,576 → 277,280 tokens at 0.83 (`:122-124`). A ships `auto` with the same rationale — the sm70 decode kernel reads only the selected top-k, so 2 bytes/element costs no decode speed, and FP16 accepts better (`sglang-sm70/docs/v100/Qwen3.8-Flash-Next.md:150`). **Caveat:** B's *entire* MTP measurement record used E5M2 (`sglang-sxm2/benchmark/qwen38_nvfp4_v100_mtp_20260908/README.md:96-100`), so B's 113–119 numbers are an E5M2 baseline — expect more, not less, with FP16.

**R3 — MTP 3 steps / 4 draft tokens / topk 1 is the chosen tree shape everywhere.**
Chosen in B (`…/qwen38_nvfp4_v100_mtp_20260908/README.md:3-8`), in A (`…/serve_…_v100.sh:259-268`), in C (`…/scripts/serve_qwen38_flash_next_nvfp4_v100_mtp.sh:42-47`). Alternatives measured and rejected: 2 draft tokens (`mtp12`: round 18.8–19.3 ms but 90.7–95.0 tok/s natural, `:138-142`), 2 steps/3 tokens (`mtp23_native`: 109–125 natural but 3/4 response-equivalence failures, `:353-368`), 7 steps/8 tokens (copy/edit +15–20% but prose −27…−33%, `sglang-V100/benchmark/qwen38_c14_strata_20261104/README.md:53-61`, `:88-96`).

**R4 — 2,048-token prefill chunk is the memory-safe value; 4,096 is untested at C's gates; 8,192 OOMs.**
- 8K chunk OOM: PLE `conv1d` asked for another **160 MiB** and failed with 110.69 MiB free on GPU 0 (`sglang-V100/benchmark/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:203-210`); the same 160 MiB is one `[8192, 10240]` FP16 buffer, "and the current path materializes several" (`…/qwen38_nvfp4_v100_core_audit_20261103/README.md:194-199`).
- A's independent 8K failure: "134 MiB alloc failed" → `--max-prefill-tokens 4096` (`sglang-sm70/scripts/serve_qwen38_flash_next_nvfp4_v100.sh:144-149`).
- C's 4K **dispatch** experiment failed a prefix-cache gate, not OOM: one prefix hit fell 30,720 → 28,672 tokens because chunk boundaries space checkpoints further apart; reverted (`…/qwen38_concurrent_core_audit_20261104/README.md:107-113`).
- 2K is what preserves "intermediate prefix checkpoints" that C relies on *instead of* hicache (`sglang-V100/docs/v100/models/qwen38-flash-next-nvfp4.md:130-134`).

**R5 — Enable the prefill/decode kernel toggles C ships on and A leaves opt-in.**

| Env var | Isolated measured win | Full-model measured win | Evidence |
|---|---|---|---|
| `SGLANG_SM70_MOE_PREFILL_2K=1` | 40-config screen, **1.153×** geometric, ≥1.114× per layer, production 1.13–1.18× | ≈**4–5% lower long-context TTFT** | `…/qwen38_nvfp4_v100_core_optimization_20261103/README.md:39-57` |
| `SGLANG_SM70_MTP_SHARED_GATE=1` | gate 2/4-row **8.45 → 2.88 µs**; 8/12/16-row 9.0–9.2 → **3.0 µs** | small; overlaps routed experts | `:59-65`, `…/qwen38_concurrent_core_audit_20261104/README.md:83-93` |
| `SGLANG_SM70_QSA_GROUPED_PREFILL=1` | planning+attention **2.29–2.64×** | incremental, "smaller at longer contexts" | `…/qwen38_nvfp4_v100_core_optimization_20261103/README.md:67-83` |
| `SGLANG_SM70_BATCHED_FUSIONS=1` | gate 8/12/16 rows → 3.0 µs; GDN 8-row fusion **60.69 → 45.13 µs (1.34×)** | needed only above bs 4 | `…/qwen38_concurrent_core_audit_20261104/README.md:83-97` |
| `SGLANG_SM70_BATCHED_MOE=1` / `…_MOE32=1` | widens NVFP4 MoE + top-k to 16 / 32 rows; "changes the number of independent route blocks launched, without growing per-thread accumulators" | required for the 8-request profile | `:115-120` |
| `SGLANG_MM_PREPROCESS_ON_CPU=1` | keeps pixel/JPEG work off TP0 | — | `sglang-V100/scripts/serve_qwen38_flash_next_nvfp4_v100_mtp.sh:17-18` |
| `SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=8192` | dense FA-v100 beats sparse QSA gather below 8K | — | `sglang-sm70/docs/v100/Qwen3.8-Flash-Next.md:98`, `sglang-V100/scripts/serve_qwen38_flash_next_nvfp4_v100.sh:20-22` |
| `SGLANG_SM70_DENSE_GEMV`, `SGLANG_SM70_QWEN_FUSIONS`, `SGLANG_SM70_FORCE_FP16` | batch-one Volta projections; QKV 32.55 → 26.98 µs, shared gate/up 6.21 → 4.02 µs; fusions 6.90 → 4.10 µs and 8.46 → 2.94 µs | decode +4.8–5.2% in the first pass | `sglang-sm70/scripts/serve_…_v100.sh:67-72`, `sglang-sxm2/benchmark/qwen38_nvfp4_v100_70tps_20260907/README.md:101-112` |

**R6 — Hierarchical cache is A-only, and its own comments say it costs GPU headroom.**
A ships host+disk hicache: `--hicache-size 8` per rank, total `N × 2 × 4` ranks = 64 GB; N=16 → 128 GB left only ~2 GB free and was "thrash-prone" (`sglang-sm70/docs/v100/Qwen3.8-Flash-Next.md:154`, `…/serve_…_v100.sh:154-158`). `page_first` is mandatory for the Mamba host pool (`:199-206`). C's profiles have **no** hicache and still get prefix reuse from 2K-chunk checkpoints (§R4). A's mem-fraction comment blames "the ~2.5G headroom left after the KV pool + hierarchical-cache GPU overhead" for the 8K-chunk OOM (`…/serve_…_v100.sh:144-148`).

**R7 — `write_back` is mandatory on this class of storage.**
`write_through` collapsed the disk READBACK path to **344 tok/s (3 tok/s at 55k tokens)** because disk writes dirtied page cache on DRAM-less SATA SSDs (30 MB/s cold `dd`, 678 MB/s 5 s after) (`…/serve_…_v100.sh:168-186`, `sglang-sm70/docs/v100/Qwen3.8-Flash-Next.md:156`). `write_through_selective` is explicitly not an alternative: agentic sessions re-hit the same prefix, so the `hit_count>=2` threshold fires within one turn (`…/serve_…_v100.sh:184-186`).

**R8 — Disk-tier metadata cache on**: 92–119 ms → **20 ms** pre-prefill wait at 50–68k files (`…/serve_…_v100.sh:81-87`; `docs/v100/Qwen3.8-Flash-Next.md:155`).

**R9 — `--disable-prefill-cuda-graph`** on V100: upstream enables breakable prefill graphs for this model; A's capture "does not fit next to the KV pool … the draft's capture ran out of memory at 4096 tokens" (`sglang-sm70/docs/v100/Qwen3.8-Flash-Next.md:158`, `…/serve_…_v100.sh:225-230`). No doc measures the TTFT gain that would justify shrinking the pool instead.

**R10 — `--sleep-on-idle`**: without it each rank busy-spins a core because the idle loop runs a tp-group `all_reduce` per iteration via `check_hicache_events()`; ~7.5 cores total at DP=2 with GPUs at 0% (`…/serve_…_v100.sh:209-219`; `docs/v100/Qwen3.8-Flash-Next.md:159`). GLM page adds: "Decode speed is the same either way" (`docs/v100/GLM-5.3-Flash.md:155`).

**R11 — `--warmups sampling`**: builds FlashInfer sampling kernels at startup; otherwise the first sampled request waits ~90 s cold (`docs/v100/Qwen3.8-Flash-Next.md:160`, `…/serve_…_v100.sh:256-257`). GLM page uses `prefix_reuse,sampling` (`docs/v100/GLM-5.3-Flash.md:131`).

**R12 — `--enable-cache-report`**: pure reporting (gates `usage.prompt_tokens_details.cached_tokens` / `cache_read_input_tokens`); the count is tracked regardless (`…/serve_…_v100.sh:233-239`; `docs/v100/Qwen3.8-Flash-Next.md:161`).

**R13 — all-reduce stays on the one-shot push algorithm.**
- `SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage` in A (`…/serve_…_v100.sh:62`) and C (`…/scripts/serve_qwen38_flash_next_nvfp4_v100.sh:11`).
- 50-config 4-GPU probe at 2 and 4 rows: existing one-shot push wins, "no all-reduce setting changed" (`sglang-sxm2/benchmark/qwen38_nvfp4_v100_mtp_20260908/README.md:86-87`); 3.35 µs vs ≥3.53 µs for pull configs (`…/qwen38_nvfp4_v100_70tps_20260907/README.md:150-152`).
- `SGLANG_SM70_AR_PUSH16=1` saves only 0.2–0.6 µs on small messages and gave no server gain (`…/qwen38_c14_strata_20261104/README.md:152-155`), so C ships it **0** (`sglang-V100/scripts/serve_qwen38.sh:52`).
- All-reduce is 20.68 ms of summed kernel time in a 489 ms traced window (`…/qwen38_nvfp4_v100_spec_20261102/README.md:157-159`) and 4.6% of decode kernel duration (`…/qwen38_nvfp4_v100_core_audit_20261103/README.md:98-102`) — not a top-3 lever.

**R14 — attention/linear-attn backends: only one configuration is documented as measured.**
`--attention-backend tilelang_fa_v100`, `--linear-attn-prefill-backend tilelang`, `--linear-attn-decode-backend triton` in A (`…/serve_…_v100.sh:98-100`), C base (`…/serve_qwen38_flash_next_nvfp4_v100.sh:32-34`) and the v2 docker sizing (`…/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md:50-52`). No fork documents a throughput comparison against another `--attention-backend` value for this model — see §7.

**R15 — `--enable-metrics`**: low overhead, surfaces accept length/queue depth/prefix hit-miss; A deliberately does **not** add `--enable-metrics-for-all-schedulers` because per-rank instrumentation adds per-step overhead at DP1 (`…/serve_…_v100.sh:240-251`).

---

## 3. Rejected / neutral settings (with the number that rejected them)

### 3.1 Rejected, with measurements

| Setting | Measured outcome | Verdict | Source |
|---|---|---|---|
| `--max-running-requests 4` (A's own framing) | aggregate 108.8 vs 116.7 @3; TTFT 656 ms → 2.5 s | rejected **for per-stream latency**, not for aggregate | `sglang-sm70/scripts/serve_…_v100.sh:133-141` |
| `--max-running-requests 16` (target-only) | target 239.36 vs MTP 183.87 (**−23.2%**); TPOT 55.64 ms | rejected: speculation + slow tail lose | `sglang-V100/benchmark/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md:40-44` |
| `--mem-fraction-static 0.88` | OOM-crash at ~2 min under 32k × np=8; ~0.6 GiB free | rejected | `…/serve_…_v100.sh:117-130`, `docs/v100/Qwen3.8-Flash-Next.md:151` |
| `--mem-fraction-static 0.83` for 262K target-only | only 259,328 KV slots → gate failed | rejected; use **0.84** | `sglang-V100/benchmark/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:176-179` |
| prefill chunk 8,192 | PLE conv1d 160 MiB alloc failed; concurrency-16 warmup died at 0.85 | rejected | `…/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:203-210`, `…/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md:61-67` |
| prefill chunk 4,096 (C's dispatch change) | prefix hit 30,720 → 28,672 tokens → shared-document cache gate failed | rejected, patch retained | `…/qwen38_concurrent_core_audit_20261104/README.md:107-113` |
| `--hicache-size 16` | 128 GB host tier, ~2 GB free, "near thrashing" | rejected | `…/serve_…_v100.sh:154-158` |
| `--hicache-size 5` | cleared the 694k device pool by only ~23% and asserted once then passed → latent crash-loop | rejected as racy | `…/serve_…_v100.sh:160-166` |
| `--hicache-write-policy write_through` | readback 344 tok/s (3 tok/s @55k) | rejected | `…/serve_…_v100.sh:168-186` |
| `--hicache-mem-layout layer_first` | MambaPoolHost accepts only `page_first`/`page_first_direct` | not an option | `…/serve_…_v100.sh:199-206` |
| `--enable-prefill-cuda-graph` (i.e. not disabling) | draft capture OOM at 4096 tokens | rejected | `…/serve_…_v100.sh:225-230` |
| MTP 2 draft tokens (`mtp12`) | 18.8–19.3 ms round but 90.7–95.0 tok/s natural; 3/4 short responses differ | not selected | `sglang-sxm2/benchmark/qwen38_nvfp4_v100_mtp_20260908/README.md:138-142` |
| MTP 2 steps/3 tokens (`mtp23_native`) | 19.0–19.7 ms round; 109–125 natural; 3/4 answers differ | isolated patch, outside runtime | `:353-368` |
| MTP 7 steps/8 tokens | +15.2% vs 4-token on copy/edit but **prose −32.9%**; SPEED −19.4%/−11.3% | opt-in specialist only | `sglang-V100/benchmark/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:128-133`, `…/qwen38_c14_strata_20261104/README.md:106-115` |
| MTP 6-token | screened at 8K, no balanced improvement | not selected | `…/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:130-133` |
| Adaptive MTP width `[3, 7]`, EMA 0.2, interval 5 | acceptance collapsed to **1.21** output tokens/verification; 7→3 transition raised a CUDA indexing assertion, reproduced on pristine v5.5 | **rejected** for serving | `…/qwen38_c14_strata_20261104/README.md:117-131` |
| 6-request MTP-1/2 profile (with faster vision) | 4-client −7…−10%, 8-client SPEED +4.2%, no fixed-length gain | not selected | `…/qwen38_vision_v100_20261104/README.md:44-51` |
| `SGLANG_SM70_PROMPT_LOOKUP=1` | code −7…−9% at 1K–25K (accept 3.76–3.83 → 3.32–3.46; 0.7–0.8 ms cheaper cycle did not pay); 25K prose 86.8 → 76.7; 70K prose 110.0 → 68.8, code 113.5 → 57.9 | **do not enable** | `…/qwen38_nvfp4_v100_spec_20261102/README.md:100-117` |
| FR-Spec / 64K hot-vocab map | reduced acceptance on the CJK screen | not adopted | `…/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:79-83` |
| FR-Spec 131,328-token map | +1.6% code, −0.9% prose; screen −3.8…+1.5% greedy | keep full vocab | `:134-137` |
| `SGLANG_SM70_QSA_FUSED_GRAPH_METADATA=1` | shipped **0**; the QSA port "does not isolate its speed effect from restart and runtime variation"; the 120→140 tok/s stretch "has **not** been demonstrated" | opt-in only, off | `…/qwen38_nvfp4_v100_spec_20261102/README.md:125-128`, `sglang-V100/scripts/serve_qwen38_flash_next_nvfp4_v100_mtp.sh:22` |
| Shared-expert fusion for MTP (`mtp34_fused`) | isolated 9.3 → 3.0 µs and 7.7 → 6.1 µs, but serving round **23.6–24.4 ms vs 23.5–24.3 ms** (no change) and changed token IDs in 2/4 checks | **removed from production** | `…/qwen38_nvfp4_v100_mtp_20260908/README.md:165-178` |
| 18-variant MoE unroll/CTA probe | 4-row 75.2 → 73.5 µs, too small | no change selected | `:156-163` |
| Grouped Marlin at 2 rows | 156.4 vs 47.3 µs; wider-K can't handle the 160-wide down projection | excluded | `:161-163` |
| Wider expert weight loads | best 4-row gate/up 71.35 µs vs **67.71 µs** at the existing width | not selected | `:375-380` |
| Padding 4-row cuBLAS to 8/16/32 rows | "fails to improve any of the three measured projection shapes" | not selected | `:376-380` |
| 8-row wide HC mixing | graph replay 39.2 / 52.5 µs vs **34.5 / 38.2 µs** | not integrated | `…/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:105-109` |
| 8-row wide projection | **9 of 10 shapes slower**; best saved ~2.5 µs on one 640×2560 | rejected | `:191-195` |
| Canonical QSA index ordering (sort in expansion kernel) | preserved text only 1 of 2 tasks at 8K, neither at 25K/70K; **+0.1 ms/round** | removed | `…/qwen38_nvfp4_v100_mtp_20260908/README.md:276-288` |
| Expert-weight reuse across verification rows | only ~1.0–1.95× grouping; saves ≤5.6 µs even with all ten shared | rejected | `:299-303`, `…/qwen38_c14_strata_20261104/README.md:187` |
| FP16 WMMA projections / cuBLAS algo selection / HC prefetch / wider scalar tiling | 314 shape records "mostly lost"; one 1.18× case; HC-up 15–17% lead at 12/16 rows only; 3.8% isolated two-row gain | not integrated | `…/qwen38_c14_strata_20261104/README.md:184-186` |
| Wider dense/HC scalar kernels | 16-row dense needs 254 regs/thread, HC up to 182 (V100 limit 255) → mostly slower | out of serving | `…/qwen38_concurrent_core_audit_20261104/README.md:99-105` |
| Rejected: MoE split-K, single-CTA HC combine, K-as-constant GEMV, cooperative-grid HC, 512-thread HC down | split-K slower + changed rounding; coop grid saved 0.37 µs; 512-thread HC improved isolation but regressed serving (8K-out window 69.83 tok/s) | all reverted | `…/qwen38_nvfp4_v100_70tps_20260907/README.md:143-152` |
| `SGLANG_SM70_MTP_LOCAL_TOP1`, `SGLANG_SM70_HC_COMBINE_NORM`, `SGLANG_SM70_AR_PUSH16` | compact profile prose ±0.5% (within noise), copy/edit 1.6–2.5% (not confirmed by restored control) | ship **0** | `…/qwen38_c14_strata_20261104/README.md:53-86`, `sglang-V100/scripts/serve_qwen38.sh:52` |
| Grouped-QSA **decode adapter** (1Cat #387 Page4) | prototype rejected on its microbenchmark incl. planner cost; timings lost in host restart | not integrated | `…/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:90-103` |
| Upstream #41166 grouped small copies | tuple kernel fails to compile on pinned Triton 3.5.1 for unit strides | removed | `…/qwen38_nvfp4_v100_spec_20261102/README.md:62-73` |
| Upstream #41175 NEXTN verify/draft prep | deferred: worker/layout interfaces differ | deferred | `:64-67` |
| Upstream #40972 QSA KV prep | not applicable (targets BF16/E4M3 + TRT-LLM; this path is FP16/E5M2) | n/a | `:68` |
| `NCCL_P2P_LEVEL=NVL` | on PCIe-only cards sends traffic through host memory; B's lineage measured it as a slowdown and now leaves NCCL alone | C base still sets it (`sglang-V100/scripts/serve_qwen38_flash_next_nvfp4_v100.sh:10`); A/B leave it unset (`sglang-sm70/scripts/serve_…_v100.sh:60-64`) | `sglang-sxm2/README.md:184`, `sglang-V100/docs/v100/models/qwen38-flash-next-nvfp4.md:155-167` |
| `--enable-metrics-for-all-schedulers` | per-rank instrumentation adds per-step overhead at DP1 | deliberately omitted | `sglang-sm70/scripts/serve_…_v100.sh:249-250` |
| `--enable-health-endpoint-generation` | set **0** (avoid generation on health probes) | ship 0 | `…/serve_…_v100.sh:73`, `sglang-V100/scripts/serve_…_v100.sh:19` |

### 3.2 Documented as neutral / no-gain

- HC gate fusion (~3 µs isolated) — enabled by default but "full-model effect is much smaller because this work overlaps routed experts" (`…/qwen38_nvfp4_v100_mtp_20260908/README.md:290-297`; `…/qwen38_nvfp4_v100_core_optimization_20261103/README.md:63-65`).
- Router sort 4.77 → 4.27 µs isolated, no serving gain (`…/qwen38_nvfp4_v100_70tps_20260907/README.md:129-132`); the router/ PLE follow-up "did not establish an additional serving gain" (`:152-154`).
- CPU graph submission ≈0.2 ms/round — cannot explain the gap to 120 tok/s (`…/qwen38_nvfp4_v100_mtp_20260908/README.md:197-201`).
- State-commit 0.19 ms = <1% of cycle — "little remaining budget" (`…/qwen38_nvfp4_v100_core_audit_20261103/README.md:83-96`, `:201-203`).
- CPU preprocessing thread policy: 28-thread default unchanged, identical feature hashes (`…/qwen38_vision_v100_20261104/README.md:71-73`).
- PLE: ~1% of prefill GPU work; "no evidence that it dominates steady decode" (`…/qwen38_c14_strata_20261104/README.md:192`, `…/qwen38_nvfp4_v100_core_audit_20261103/README.md:194-199`).
- Prefill-side core optimization round: +4.7% latency, **+0.4%/+0.7% decode** — i.e. prefill work does not move decode (`…/qwen38_nvfp4_v100_core_optimization_20261103/README.md:23-35`).

---

## 4. Sweep tables and the scripts that generate them

### 4.1 Sweep tables present as tables in docs

| Sweep | Rows | Where |
|---|---|---|
| Target-only dev points × prefill × decode (25K/2K-out) | 8 configs | `sglang-sxm2/benchmark/qwen38_nvfp4_v100_70tps_20260907/README.md:19-28` |
| Seed × prefill × decode | 3 | `:38-42` |
| main vs candidate, 3 shapes × (prefill, decode) | 3 | `:70-74`; earlier pass `…/qwen38_nvfp4_v100_20260907/README.md:26-30` |
| MTP candidate × 4 lengths × (random, natural, lowest window) | 4 tables (`mtp34_gdn`, `mtp34_qsa`, `mtp34_vector`, `mtp23_native`) | `…/qwen38_nvfp4_v100_mtp_20260908/README.md:130-135`, `:215-220`, `:259-264`, `:332-337`, `:358-363` |
| QSA accumulator: verify rows 1/2/4 × before/after µs | 3 | `:246-250` |
| Native HC / GDN / QKVZBA isolated pairs | several | `:79-87`, `:180-189` |
| MTP round cost × candidate | 5 (38.9 → 21.6 → 23.5–24.3 → 22.7 → 22.2 → 21.6 ms) | `:20-23`, `:137-139`, `:207-209`, `:255-257`, `:327-330` |
| Host vs Docker × 4 lengths × (prefill, decode) + random accept/round | 3 tables | `…/qwen38_nvfp4_v100_docker_v4_20260908/README.md:47-73` |
| Concurrency 1/4/8/16 × target/MTP × TTFT/TPOT/accept | 4 | `sglang-V100/benchmark/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md:35-40` |
| E5M2 vs FP16 KV × 4 lengths × 2 prompt types | 8 | `…/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:113-121` |
| prompt-lookup off/on × 4 lengths × 2 prompt types (interleaved, order reversed) | 8 | `…/qwen38_nvfp4_v100_spec_20261102/README.md:100-105` |
| v5.5 vs compact vs wide8 × 7 workload rows; restored control × 6 | 2 tables | `…/qwen38_c14_strata_20261104/README.md:53-61`, `:72-80` |
| C1 per-stream × 3 profiles; tokens/verification | 2 | `:88-96` |
| 1/2/4 clients × aggregate × power; 4 vs 8 agents × DCGM | 2 | `…/qwen38_concurrent_core_audit_20261104/README.md:26-32`, `:166-175` |
| v5.4 vs v5.5 × 5 workloads | 5 | `:146-152` |
| 70K prefill attribution %; decode kernel category %; DCGM prefill/decode; calibration | 4 | `…/qwen38_nvfp4_v100_core_audit_20261103/README.md:58-69`, `:98-102`, `:115-121`, `:137-142` |
| 6 cases × (baseline TTFT/request, candidate TTFT/request) | 6 | `…/qwen38_nvfp4_v100_core_optimization_20261103/README.md:14-21` |
| 3 profiles × 4 text/SPEED cells | 3 | `…/qwen38_vision_v100_20261104/README.md:40-44` |
| Host/Docker C1/C4/C8 | 3 | `…/qwen38_docker_v5_20261105/README.md:60-64` |
| PCIe: NCCL vs custom-AR × 4 workloads | 2 | `sglang-V100/docs/v100/models/qwen38-flash-next-nvfp4.md:172-175` |
| v2 Docker concurrency 1/4/8/16 × target/MTP | 4 | `:204-209` |
| MTP knob table (GLM cross-model) steps 2/3/4 | 1 line | `sglang-sm70/docs/v100/GLM-5.3-Flash.md:146` |

### 4.2 Raw sweep data files (jsonl/json/csv, not rendered as tables)

- B: `benchmark/qwen38_nvfp4_v100_20260907/{moe_tuning.jsonl, gemv_tuning.jsonl, marlin_tuning.jsonl, kernel_experiments.json, measurements.jsonl, final_kernel_benchmark.jsonl}`; `…/qwen38_nvfp4_v100_70tps_20260907/{gemv_tuning.jsonl, kernel_experiments.json, decode_profile_summary.json, short_measurements.jsonl}`; `…/qwen38_nvfp4_v100_mtp_20260908/{moe_unroll_tuning.jsonl, moe_scale_timing.jsonl, moe_wide_gate_timing.jsonl, hc_gate_timing.jsonl, hc_norm_timing.jsonl, hc_tensorcore_padding_timing.jsonl, qsa_tile_timing.jsonl, topk_tile_timing.jsonl, three_rows_timing.jsonl, qsa_production_timing.jsonl, allreduce_batch_probe.log, graph_events_summary.json, graph_submit_summary.json}` (`…/qwen38_nvfp4_v100_mtp_20260908/README.md:89-92`, `:271-274`, `:311-312`, `:379-380`).
- C: `benchmark/qwen38_c14_strata_20261104/{summary.json, metadata.json, raw/*.json.gz}`; `…/qwen38_concurrent_core_audit_20261104/{summary.json, raw/dcgm_*.jsonl.gz, raw/mtp3_c8_dcgm_sm.jsonl.gz}`; `…/qwen38_nvfp4_v100_kv_qsa_20261102/{matrix.json, wide_hc_micro.json, wide_projection_micro.json, profiles/*, resumed_20261103/*}`; `…/qwen38_nvfp4_v100_core_optimization_20261103/{summary.json, metadata.json, raw/}`; `…/qwen38_nvfp4_v100_spec_20261102/{candidate_summary.json, candidate_lookup_hits.json, profile_summary.json}`.
- A: **no Qwen sweep data at all.** A's only sweep script is `scripts/dsv41_sm70_marlin_decode_sweep.py` (DeepSeek-V4.1, different model) plus `dsv41_v100_{custom_ar_microbench,decode_baseline,nsys_buckets,engram_ablate,verify_profile}.py` — all DSV4.1.

### 4.3 Sweep generators (scripts)

| Script | What it sweeps | Fork |
|---|---|---|
| `benchmark/qwen38_spec/bench.py sweep` | **process-owned server restarts** from a JSON config: `{"order": ["a","b","b","a"], "variants": {"a": {"command": [...], "env": {...}}}}`; refuses to start over a listening server; `required_log_markers` abort a silent fallback | C (`…/qwen38_spec/README.md:85-109`) |
| `benchmark/qwen38_spec/round.py` | resumable multi-stage round from `matrix.json`; archives each stage; optional `cases`/`repeats`/`speed`/`multimodal`/`profile` bounds | C (`:111-130`) |
| `benchmark/qwen38_spec/compare.py` | paired per-prompt median decode ratios + geometric-mean domain summaries, gated on matching input hashes/temps/output limits | C (`:128-135`) |
| `benchmark/qwen38_spec/bench.py run --lookup-ab` | interleaved prompt-lookup A/B, order reversed per repeat | C (`:71-83`) |
| `benchmark/qwen38_spec/{tune_moe_prefill.py, screen_core_candidates.py, screen_grouped_qsa.py, prefill_ab.py, summarize_core_optimization.py}` | 40-config MoE geometry; core candidates; grouped-QSA; interleaved prefill A/B via a test-only control file | C (`…/qwen38_nvfp4_v100_core_optimization_20261103/README.md:144-148`) |
| `benchmark/qwen38_spec/core_audit.py` + `analyze_core_audit.py` + `plot_core_audit.py` + `hardware_calibration.py` | timing / DCGM / CPU-event / GPU-trace passes on a dedicated endpoint | C (`…/qwen38_spec/README.md:224-236`) |
| `benchmark/qwen38_spec/probe` (PYTHONPATH) + `collect_profile.py`, `analyze_profile.py` | rank-zero CUDA events for draft/verify/target/commit/extend/round + CPU submission; `SGLANG_SPEC_PROBE_DIR`, `_ON_PROFILE`, `_ROUNDS` (default 512) | C (`:137-186`) |
| `benchmark/qwen38_spec/{check.py, check_queue.py, check_cached_document.py}` | correctness gates; 4 clients × 6 growing docs; real long prefix hit vs cold replay | C (`:188-216`, `:244-246`) |
| `benchmark/qwen38_nvfp4_v100_mtp_20260908/{bench.py, bench_natural.py}` | random-token vs natural prose/code sweep at 1K/8K/25K/70K, per-request acceptance retained | B (`…/qwen38_nvfp4_v100_mtp_20260908/README.md:115-120`) |
| `benchmark/qwen38_nvfp4_v100_mtp_20260908/probe_*.py` | `probe_moe_unroll`, `probe_moe_scales`, `probe_moe_wide_gate`, `probe_hc_tensorcore_padding`, `probe_qkvzba_fused`, `probe_batch_fusions`, `build_moe_*` | B (`:154-163`, `:316-320`) |
| `benchmark/qwen38_nvfp4_v100_70tps_20260907/bench_sustained.py` | 25K/2K-out sustained decode, `--output-len`, `--repeats`, `--seed` | B (`…/qwen38_nvfp4_v100_70tps_20260907/README.md:194-200`) |
| `benchmark/qwen38_nvfp4_v100_20260907/bench_{serving,kernels}.py` | serving A/B by label (refuses an existing label); kernel entry-point microbench | B (`…/qwen38_nvfp4_v100_20260907/README.md:185-198`) |
| `benchmark/qwen38_nvfp4_v100_docker_v4_20260908/{validate.py, summarize.py}` | docker-vs-host parity across 4 runs | B/C (`…/qwen38_nvfp4_v100_docker_v4_20260908/README.md:180-197`) |
| `benchmark/qwen38_c14_strata_20261104/{collect_results.py, plot_results.py}` | collection/plots for the C1–C4 strata screen | C |
| `benchmark/qwen38_vision_v100_20261104/image_bench.py` | 32-image page benchmark with cache flush | C (`…/qwen38_vision_v100_20261104/README.md:97-101`) |
| `scripts/dsv41_sm70_marlin_decode_sweep.py` | Marlin decode sweep — **DeepSeek-V4.1 only** | A |

---

## 5. A vs B vs C divergences (candidate regressions)

### 5.1 A ships B's measured doc verbatim but has no measurement of its own at that level

`docs/v100/Qwen3.8-Flash-Next.md` is **byte-identical** in A@`sm70/stack` and B (same MD5), including the 2026-10-02 date, 112.8 tok/s, 332,736-token pool, 47 forwards/s and the 3-stream/4-octet TTFT tables (`sglang-sm70/docs/v100/Qwen3.8-Flash-Next.md:41-77`). A's only *own* Qwen numbers are 84 / 75 / 73.7 tok/s from 128-token `/generate` spot checks (`sglang-sm70/SM70.md:5`, `:37`, `:59`). **A therefore publishes 112.8 while its own status page reports 73.7–84; the older 84 was measured with degenerate output** (`SM70.md:7-12`), which inflates acceptance.
→ Treat A's 112.8 as *B's measurement, inherited*, and A's 73.7 as the only number A currently reproduces.

### 5.2 A's front page dropped every measured number

B's `README.md` carries the perf table and per-model decode lines (**~113 tok/s one stream, 102–140, MTP ~2.4 tok/step, 121/134 at 2/3 streams** — `sglang-sxm2/README.md:47-58`, plus the agent-loop TTFT table 9 s / 40 s / 62 s cold and 0.34–0.61 s per tool turn, `:51-55`). A's `README.md` is upstream's README plus a V100 build section that only links to `docs/v100/*` (`sglang-sm70/README.md:25`, `:71-113`; no `tok/s` line anywhere). Losing the front-page numbers also loses the "sxm2 reference vs this fork" comparison that the 112.8 claim depends on. Minor: B's own Performance header says "Measured on 8× V100-SXM2-32GB" while the Qwen section is "on four cards" (`sglang-sxm2/README.md:32`, `:47-49`) — the 113 figure is a 4-card number despite the header.

### 5.3 Concurrency ceiling: A=3, C=8 — and A's own justification contradicts C's measurement

- A: 3 requests, graphs 1–3, "aggregate decode peaks here. A fourth stream adds no throughput and stretches TTFT" (`sglang-sm70/scripts/serve_…_v100.sh:133-141`, `docs/v100/Qwen3.8-Flash-Next.md:152`).
- C: 8 requests, graphs 1–8, 48 Mamba slots: **8-client 253.1 → 360.4 tok/s**, 4-client unchanged (`sglang-V100/benchmark/qwen38_concurrent_core_audit_20261104/README.md:146-152`, config `:183-189`).
- Both can be true because they measure different things: A reports aggregate at np=3 (116.7) vs np=4 (108.8) on six short chat prompts with per-stream rates 67/53/43/38; C reports *whole-batch aggregate including prefill and queueing* at 8 clients with per-stream rates lower but total higher ("each individual stream decodes more slowly … but requests spend less time queued", `:155-158`).
- Note A's *doc* only goes to 3 streams (2 → 121.0, 3 → 133.6; `docs/v100/Qwen3.8-Flash-Next.md:57`), so the 108.8-at-4 figure exists **only** inside a code comment and is a single unreplicated measurement. Neither fork documents a 4-client-per-stream vs 8-client-aggregate comparison on the same workload.
- **A also cannot benefit from its own 8-row kernels**: with graphs captured only for bs 1–3 and `BATCHED_*` unset, `SGLANG_SM70_BATCHED_FUSIONS` / `BATCHED_MOE32` (8/12/16 rows, `…/qwen38_concurrent_core_audit_20261104/README.md:83-97`, `:115-120`) are dead code at A's ceiling.

### 5.4 KV dtype: three different stories

| Fork | Doc claim | Launcher reality | Measurements were taken with |
|---|---|---|---|
| A | FP16 "costs no decode speed and accepts better than `fp8_e5m2`" (`docs/v100/Qwen3.8-Flash-Next.md:150`) | `--kv-cache-dtype auto` (`scripts/serve_…_v100.sh:108`) | A's 3 spot checks: dtype unstated; the mem-fraction comment implies a 1-byte pool (`:123-125`: 402,912 tokens) |
| B | same text (identical file) | identical script | **E5M2 everywhere** (`benchmark/qwen38_nvfp4_v100_mtp_20260908/README.md:96-100`, `…/qwen38_nvfp4_v100_docker_v4_20260908/README.md:91-94`, `…/qwen38_flash_next_nvfp4_mtp_v100_20260829/README.md:12`) |
| C | FP16 is the recommendation (`benchmark/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:154-170`) | `auto` in the MTP profile, `fp8_e5m2` default in the base profile (`scripts/serve_…_v100.sh:6`, `…_mtp.sh:13`) | both, with a controlled A/B (`:113-121`) |

Net: B's 113–119 tok/s headline is an **E5M2** number, so A shipping FP16 (correctly) should be *faster*, not 40% slower. The dtype does not explain A's gap.

### 5.5 Pool sizing is unreproducible in A (three different numbers in one fork)

- `332,736 tokens` — `sglang-sm70/docs/v100/Qwen3.8-Flash-Next.md:43`
- `402,912 → 352,864 tokens` — `scripts/serve_…_v100.sh:123-125`
- `694k tokens/rank` — `scripts/serve_…_v100.sh:163`, `:190`
- C, for comparison, **pins** it: `--max-total-tokens 277280` (`sglang-V100/scripts/…_throughput.sh:17`) or `540672` (`…/qwen38_nvfp4_v100_full_context_mtp_concurrency_20260925/README.md:16`), and reports 277,280 / 554,576 / 540,672 / 536,512 / 221,616 for specific configs (`…/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:122-124`, `…/qwen38_flash_next_qsa_prefill_fix_v100_20260830/README.md:61-67`).

### 5.6 hicache: A-only feature, and A's own numbers show it costing GPU headroom

A enables host(64 GB)+disk tiers (`docs/v100/Qwen3.8-Flash-Next.md:154-156`, `scripts/serve_…_v100.sh:159-208`). C has **none** in any Qwen profile (`sglang-V100/scripts/serve_qwen38_flash_next_nvfp4_v100_throughput.sh:1-21`, `…_mtp.sh:36-47`; docs describe prefix reuse via "intermediate prefix checkpoints", `docs/v100/models/qwen38-flash-next-nvfp4.md:130-134`). A blames hicache GPU overhead for its 8K-chunk OOM (`scripts/serve_…_v100.sh:144-148`) and notes the 8-size sizing was measured under `layer_first` and "must be re-validated" (`:204-205`). C's counter-measurement: 4K chunks lost prefix density (30,720 → 28,672 cached tokens) even *without* a host tier (`…/qwen38_concurrent_core_audit_20261104/README.md:107-113`).

### 5.7 Env-var defaults flipped between C and A

| Var | C | A | Consequence |
|---|---|---|---|
| `SGLANG_SM70_MTP_SHARED_GATE` | **1** (`…_mtp.sh:25`) | opt-in, "all default off" (`docs/v100/Qwen3.8-Flash-Next.md:169-171`, `:175`) | A leaves an 8.45→2.88 µs (and 9.0→3.0 µs at 8/12/16 rows) win off |
| `SGLANG_SM70_MOE_PREFILL_2K` | **1** (`…_mtp.sh:26`) | opt-in (`:180`) | A leaves a 1.153× MoE / 4–5% TTFT win off — and A's chunk is 4096, so the 2K-only guard wouldn't fire anyway |
| `SGLANG_SM70_QSA_GROUPED_PREFILL` | **1** (`…_mtp.sh:27`) | not mentioned at all in A's docs/launcher | a 2.29–2.64× isolated prefill win is absent from A |
| `SGLANG_SM70_BATCHED_MOE{,_32}`, `BATCHED_FUSIONS` | **1** (`…_throughput.sh:9-11`) | opt-in (`docs:176-179`) | dead at A's bs≤3 |
| `SGLANG_SM70_MTP_WIDE_MOE` | **0** explicit (`…_mtp.sh:24`) | opt-in (`docs:177`) | agreement (off) |
| `SGLANG_SM70_QSA_FUSED_GRAPH_METADATA` | **0** explicit (`…_mtp.sh:22`) | absent | agreement (off), C is explicit |
| `SGLANG_SM70_PROMPT_LOOKUP` | **0** explicit (`…_mtp.sh:23`) | absent | agreement (off) |
| `SGLANG_MM_PREPROCESS_ON_CPU` | **1** (`…_mtp.sh:18`) | opt-in toggle only (`docs:181`) | A does not default it on |
| `SGLANG_SM70_TURBOMIND_OPS_PATH` | required + load-marker gate (`…_mtp.sh:19`, `:30-33`; `…/qwen38_spec/README.md:103-109`) | not referenced in the launcher | C guards against a silent FP16-MoE fallback; A has no such guard |

### 5.8 A dropped multimodal pinning

C pins `--enable-multimodal` + `--mm-attention-backend fa2_v100` (`…_mtp.sh:41`, `…_throughput.sh:20`) because the native vision backend is **6.60×** faster end-to-end on a 32-image request (79.98 s → 12.12 s, `…/qwen38_vision_v100_20261104/README.md:3-7`, `:22-25`), and `triton_attn` is the old path (`:91`). A passes neither flag (`scripts/serve_…_v100.sh:90-258`), so on A the vision backend is whatever upstream defaults to — while A's own doc claims image input works on both APIs (`docs/v100/Qwen3.8-Flash-Next.md:39`).

### 5.9 A dropped the experiment harness and all Qwen sweep data

B and C carry `benchmark/qwen38_*` — verified counts: **B = 14** `qwen38_*` dirs (`git -C sglang-sxm2 ls-tree -r --name-only HEAD -- benchmark | cut -d/ -f2 | grep ^qwen38 | sort -u`), **C = 27**. A = **0** (`git -C sglang-sm70 ls-tree -r --name-only sm70/stack -- benchmark | grep qwen38` → no output). A's `scripts/` has only `dsv41_*` tooling (DeepSeek-V4.1). Consequence: A's 73.7 tok/s cannot be decomposed (no accept histogram, no round time, no per-phase timing), whereas C/B can attribute 77% of a cycle to target verification (`…/qwen38_nvfp4_v100_core_audit_20261103/README.md:83-92`).

### 5.10 C has one thing A/B fixed (not a regression)

C's base launcher defaults `NCCL_P2P_LEVEL=NVL` (`sglang-V100/scripts/serve_qwen38_flash_next_nvfp4_v100.sh:10`); B's lineage documents that forcing `NVL` routes PCIe-only cards through host memory and now leaves it to NCCL, after a measurement to that effect (`sglang-sxm2/README.md:184`, `sglang-sm70/scripts/serve_…_v100.sh:60-64`). A/B are correct here; if you port C's launcher, drop that default.

### 5.11 Internal inconsistencies inside A (verify before trusting)

| Claim A | Conflicts with |
|---|---|
| `--linear-attn-decode-backend tilelang` listed as an environment fact | the launcher uses `triton` (`scripts/serve_…_v100.sh:100`) — `SM70.md:20` vs `:100` |
| "A fourth stream adds no throughput" | the 108.8@np4 figure appears **only** in that same comment, is unreplicated, and contradicts C's `+42.4%` at 8 clients (`…/qwen38_concurrent_core_audit_20261104/README.md:149`); A's doc stops at 3 streams (`docs/v100/Qwen3.8-Flash-Next.md:57`) |
| 332,736-token pool | 402,912/352,864 (§5.5) |
| 112.8 tok/s | 73.7–84 tok/s (`SM70.md`) |
| "`SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage`" | A also says never set `NCCL_P2P_LEVEL` on an NVLink mesh (`:60-64`) yet C's PCIe table shows the custom AR opt-in is *required* on PCIe (`docs/v100/models/…:155-167`) — no A-side PCIe measurement |

### 5.12 A's checkout is upstream-only: `main` has none of the SM70 layer

Verified on the local clone:

| Probe | `main` | `sm70/stack` |
|---|---|---|
| `SGLANG_SM70_*` vars in `python/` | **0** | **43** |
| files with `sm70` in the path under `python/` | **0** | **81** |
| `docs/v100/` | absent | 6 files |
| `SM70.md` | 4094 B / 66 lines | 1560 B / 23 lines |
| `git diff --stat main sm70/stack -- python` | — | **339 files, +52,570 / −1,429** |

Commands: `git -C sglang-sm70 grep -ohE 'SGLANG_SM70_[A-Z0-9_]+' main -- python | sort -u | wc -l`; `git -C sglang-sm70 ls-tree -r --name-only main -- python | grep -c sm70`; `git -C sglang-sm70 diff --stat main sm70/stack -- python`.

Implications:
- Serving from the checkout runs **upstream SGLang with no Volta kernels**: A's launcher would still export `SGLANG_SM70_FORCE_FP16=1`, `SGLANG_SM70_DENSE_GEMV=1`, `SGLANG_SM70_QWEN_FUSIONS=1` (`scripts/serve_…_v100.sh:67-72`), but nothing reads them, and `--attention-backend tilelang_fa_v100` / `--linear-attn-*` would have no SM70 implementation behind them. Any throughput or correctness number attributed to "fork A" must state which ref it came from.
- On `sm70/stack` the knobs are real and wired to production modules, not just tests: `environ.py` plus `kernels/ops/moe/moe_wna16_marlin.py` (`MOE_PREFILL_2K`), `kernels/ops/gemm/sm70_qwen_fusions.py` (`MTP_SHARED_GATE`), `kernels/ops/moe/sm70_nvfp4_moe_decode.py` (`BATCHED_MOE32`), `srt/layers/attention/vision.py` (`VISION_ATTENTION`), `srt/layers/attention/tilelang_fa_v100/_decode_cuda.py` (`QSA_GROUPED_PREFILL`). So §5.7 is a *defaults* regression, not a missing-code regression — flipping the six vars is a config change, not a port.
- This is the most likely explanation for the sibling report's "`sglang-sm70` contains **zero** `SGLANG_SM70_*` env knobs" (`sm70-stack/perf-architecture-analysis.md:23`): true for the checkout (`main`), false for `sm70/stack`.

---

## 6. MTP / acceptance notes

### 6.1 Tree shape: what each shape actually buys

| Shape | Cycle / round | Acceptance (natural) | Decode natural | Decode copy/edit | Verdict |
|---|---|---|---|---|---|
| 2 steps / 3 draft (`mtp23_native`) | **19.0–19.7 ms** | — | 109.35–124.79 | — | faster cycle, 3/4 answers differ → isolated patch (`…/qwen38_nvfp4_v100_mtp_20260908/README.md:353-368`) |
| 2 draft tokens (`mtp12_gdn`) | **18.8–19.3 ms** | too few | **90.7–95.0** | — | lost acceptance → not selected (`:138-142`) |
| **3 steps / 4 draft (`mtp34_*`)** | 21.6–24.3 ms | **2.5–2.7** tok/round | **108–118** | 172–186 | **selected everywhere** (`:3-8`, `:332-337`; `…/qwen38_c14_strata_20261104/README.md:93-96`) |
| 7 steps / 8 draft (`wide8`) | — | 3.04 prose / 7.78 copy | **87.2** (−27%) | **218.1** (+20%) | workload-specific only (`…/qwen38_c14_strata_20261104/README.md:3-13`, `:88-96`) |
| adaptive `[3,7]` | — | **1.21** (collapsed) | ~56–105 | ~56 | **rejected**, plus CUDA assertion (`:117-131`) |
| `--speculative-eagle-topk` >1 | — | — | — | — | **not documented anywhere** (§7) |

### 6.2 Acceptance on real vs random text (the single most misread number in this corpus)

- Real prose/code: **2.5–2.7** tokens per round at 3/4 (`sglang-sxm2/benchmark/qwen38_nvfp4_v100_mtp_20260908/README.md:137-139`); 2.33–2.46 prose / 3.76–3.84 copy-heavy code (`sglang-V100/benchmark/qwen38_nvfp4_v100_spec_20261102/README.md:82-87`); mean **2.47** over 1,015 A-side samples (`sglang-sm70/scripts/serve_…_v100.sh:246-248`).
- Random-token corpus: **3.98 of 4** vs 2.6–2.7 on real text (`…/serve_…_v100.sh:138-140`); random decode 122–176 vs natural 108–118 (`…/qwen38_nvfp4_v100_mtp_20260908/README.md:10-15`).
- Consequence documented in plain words: random-token sweeps over-report MTP throughput by 15–50%, and **acceptance varies between runs of the *same* configuration** — "long-context outputs/acceptance can differ between runs of unchanged configurations; the cause has not been isolated" (`:142-145`), and "the highest rates do not establish consistent 120 tok/s" (`:209-210`).
- A canonises the same warning: "Measure with realistic text before changing this: a random-token benchmark reports aggregate still climbing at four" (`…/serve_…_v100.sh:138-140`).

### 6.3 Where the MTP time goes (so you know what to fix)

| Phase | ms | Share of cycle | Source |
|---|---|---|---|
| target verification forward | **17.58** | **≈77%** | `sglang-V100/benchmark/qwen38_nvfp4_v100_core_audit_20261103/README.md:83-92` |
| MTP drafting | 3.23 | 14% | same |
| draft extension | 1.56 | 7% | same |
| recurrent-state commit | 0.19 | <1% | same |
| remaining verify overhead | ~0.15 | <1% | same |
| **complete cycle** | **22.73** | 100% | same |
| (alt. measurement) target verify | 17.68 | | `sglang-sxm2/benchmark/qwen38_nvfp4_v100_mtp_20260908/README.md:305-312` |
| (alt.) draft loop / draft extend | 2.34–2.41 / 1.37–1.38 | | same |
| (alt.) cycle phases @8K | 21.81 total: draft 2.69, verify 17.61, target 17.29, commit 0.184, extend 1.51 | | `…/qwen38_nvfp4_v100_spec_20261102/README.md:142-146` |
| CPU graph submission | ~0.2 total (131 + 37/28 µs) | ~1% | `…/qwen38_nvfp4_v100_mtp_20260908/README.md:197-201` |

Implication all three forks agree on: **decode throughput is a function of (a) target-verify cost and (b) acceptance.** With 4-row verify at ~21.6–22.7 ms and accept 2.5, you get ~110–118 tok/s; at accept 2.1 you get ~99. A's 73.7 tok/s is *arithmetically* consistent with either accept ≈1.6 at a 22 ms cycle or accept 2.8 at a ~34 ms cycle — A documents neither, which is why its number cannot be diagnosed from its own files (§5.9). The live box measured accept **2.81 at ~107.6 tok/s** (§1.7), so the second explanation (a slow cycle / contended or wrong-tree serving) currently has more support than an acceptance collapse.

### 6.4 Verify-width vs kernel-width coupling (the trap in wider MTP)

- 4 draft tokens → 4/5 verify rows, which is exactly the widest existing small-row kernel; "4 steps (5-row verify) falls off the 4-row kernels and dropped to 34" on GLM (`sglang-sm70/docs/v100/GLM-5.3-Flash.md:146`).
- GLM also documents the row-count/bs coupling: `--max-running-requests 1` because row results match batch 1 "only while the whole launch has at most 4 rows" (`:149`). Qwen inherits the same shape: 2/4-row fusions (`docs/v100/Qwen3.8-Flash-Next.md:175`) and 6/8-row MoE behind `SGLANG_SM70_MTP_WIDE_MOE` (`:177`), with 8/12/16-row work behind `BATCHED_FUSIONS`/`BATCHED_MOE32` (`:176-179`).
- Wider verify is only worth it where acceptance is high: 7.78 tokens/verification on copy-heavy code vs 3.04 on prose (`…/qwen38_c14_strata_20261104/README.md:93-96`).

### 6.5 Acceptance-related correctness constraints (do not "fix" acceptance by relaxing these)

- QSA ring overwrote a retained prefix key during **even four-token** verification: reproduced reading `[8, 5, 6, 7]` instead of `[4, 5, 6, 7]`; fixed by preserving the whole verify window plus preceding group members (`sglang-V100/benchmark/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:12-28`).
- Async result copy replaced GPU sources with CPU destinations while the copy stream could still read → hung request and TP ranks disagreeing on completion; fixed by recording the copy stream (`…/qwen38_nvfp4_v100_mtp_20260908/README.md:44-56`).
- Radix-selected sparse indices change order between replays while keeping the same set → FP16 attention output changes; canonical ordering removes it but costs 0.1 ms/round and did **not** stabilise serving (`:276-288`).
- A's radix cache replays an *approximate* linear-attention state, so greedy output is not bit-identical cold vs warm (`sglang-sm70/docs/v100/Qwen3.8-Flash-Next.md:187`).
- A's degenerate-output episode was a **missing dense marlin repack** (silent zero-output stub), not a kernel bug (`sglang-sm70/SM70.md:39-43`; `sglang-sm70/README.md:85`). Any A-side throughput number taken before that fix (84 tok/s included) is invalid.

---

## 7. Explicitly NOT documented (do not assume)

| Topic | Status |
|---|---|
| `--page-size` / page-size sweep | **not documented** in any fork for this model. Only indirect: "SGLang's Page16 token mapping is converted on GPU" and Page4 grouping (`sglang-V100/benchmark/qwen38_nvfp4_v100_core_optimization_20261103/README.md:71-75`); "16,384 pages at its 262K context capacity" / 8,192-page microbenchmark 23.26 → 2.11 µs (`sglang-sxm2/benchmark/qwen38_nvfp4_v100_70tps_20260907/README.md:122-127`) |
| `--speculative-eagle-topk` other than 1 | **no measurement**; every recipe pins 1 (`…/serve_…_v100.sh:265`, `…_mtp.sh:46`) |
| `--attention-backend` alternatives | **no A/B for this model**. Only `tilelang_fa_v100` appears in every launcher; `triton_attn` is documented only for **vision** (`…/qwen38_vision_v100_20261104/README.md:91`) |
| `--linear-attn-decode-backend` alternatives | tilelang is named once as an environment fact (`sglang-sm70/SM70.md:20-21`) but every launcher uses `triton`; no comparison |
| `--enable-dp-attention`, `--dp-size` >1 for Qwen | only inert wiring (`--dp-size ${FLASH_NEXT_DP:-1}`, `…/serve_…_v100.sh:110-114`); never measured |
| `--chunked-prefill-size 8192` with **FP16** KV | never measured (8K rows are all E5M2, `…/qwen38_nvfp4_v100_docker_v4_20260908/README.md:91-94`) |
| hicache on C's profile | never tried — C's prefix reuse is chunk-checkpoint based; no comparison to A's host tier exists in any repo |
| `--mem-fraction-static` between 0.87 and 0.93 for Qwen | no sweep; only 0.80 / 0.82 / 0.83 / 0.84 / 0.85 / 0.86 / 0.88-crash |
| A's 73.7 tok/s decomposition | **no accept length, no round time, no phase timing, no sweep data** in A (`sglang-sm70` has no `benchmark/qwen38_*`) |
| The "~80 tok/s sxm2 reference" | **not present in any of the three repos** (§1.6) |
| MTP-7/8 on ≥4 concurrent requests | only C1/C2/C4 aggregate rows for C4 (`…/qwen38_c14_strata_20261104/README.md:53-61`); no 8-client wide-MTP run |
| FP16-vs-E5M2 at concurrency >1 | not measured (C's KV A/B is concurrency 1, `…/qwen38_nvfp4_v100_kv_qsa_20261102/README.md:37-48`) |

---

## 8. Answer-shaped summary

**Five highest-leverage levers to raise decode tok/s (all evidence-backed):**

1. **Acceptance and verify cost, not kernels.** 77% of a 22.7 ms MTP cycle is target verification; 2.5 accepted tokens/round ⇒ 108–118 tok/s, and every kernel win in two forks moved the cycle only 22.7 → 21.6 ms (≈4%). The levers that actually move tok/s are accepted tokens/round and the 17.6 ms verify forward — not the MoE/GDN micro-kernels (`…/qwen38_nvfp4_v100_core_audit_20261103/README.md:83-92`, `…/qwen38_nvfp4_v100_mtp_20260908/README.md:10-15`, `:342-345`). **Caveat:** the live box already accepts 2.81/4 and can hit 107.6 tok/s (§1.7), so re-measure on a pinned protocol before attributing your 74 to acceptance.
2. **Enable C's default-on prefill/row kernels** (`MTP_SHARED_GATE`, `MOE_PREFILL_2K`, `QSA_GROUPED_PREFILL`, `BATCHED_FUSIONS`, `BATCHED_MOE{,32}`): 1.153× MoE, 2.29–2.64× grouped QSA, 8.45→2.88 µs gate, 60.69→45.13 µs GDN-8row, 4–5% TTFT. A ships all of them off and cannot even use the row-widening ones at bs≤3 (§5.7, §R5).
3. **Raise the concurrency ceiling to 8 with graphs 1–8** if your throughput metric is aggregate agent traffic: 253.1 → 360.4 tok/s (+42.4%) at 8 clients with 4-client unchanged (§R1).
4. **Keep FP16 KV** (`auto`): +10.4%/+4.1%/+4.2%/+7.0% prose and +7.8%/+5.3%/+5.1%/+5.2% code over `fp8_e5m2` on a 3-repeat controlled A/B (§R2). Note B's 113–119 headline is an E5M2 measurement, so FP16 should be *better* than the reference you are comparing against.
5. **2,048-token prefill chunks + prefix checkpoints** (not 4K/8K, not hicache-first): 8K reliably OOMs PLE (160 MiB), 4K lost prefix density 30,720 → 28,672 cached tokens, and 2K is the only value validated through the 48-request document queue (≥2,952.7 MiB free) (§R4).

**Three findings most likely to be regressions in A:**

1. **A's checkout is upstream-only — the SM70 layer is on a branch you are not on.** `main` has 0 `SGLANG_SM70_*` vars and 0 sm70-named files under `python/`; `sm70/stack` has 43 vars, 81 files and a 339-file / +52,570-line diff. Serving from the checkout silently gives you upstream kernels and inert SM70 env vars, which is also the most plausible non-performance explanation for any "A is slow" observation. Verify the ref before any A/B (§5.12, §0.2).
2. **A can't reproduce its own published number.** `docs/v100/Qwen3.8-Flash-Next.md` (byte-identical to B's) claims 112.8 tok/s natural decode at 3 streams; A's own status page reports 73.7 tok/s at 128 tokens, and the only higher figure (84) was measured while output was degenerate. A has zero Qwen sweep data, so the gap is unattributable from A's files (§5.1, §5.9, §A1–A3). Against §1.7's 107.6 tok/s live measurement, the 73.7 is best read as an old/unpinned spot check, not a property of the stack.
3. **A's 3-request ceiling + bs 1–3 graphs silently disables the 8/12/16-row kernels** that C measured as its single biggest serving win (253.1 → 360.4 tok/s at 8 clients); the 108.8-at-4 figure behind A's ceiling exists only in a code comment, unreplicated, with no doc table above 3 streams. Compounding it, A dropped C's default-on prefill kernels (2K MoE 4–5% TTFT, grouped Page4 QSA 2.29–2.64×), replaced C's memory-safe 2K chunks with 4K, added a host-tier hicache whose own comment blames it for eating the prefill headroom, and dropped `--enable-multimodal --mm-attention-backend fa2_v100` (6.60× on a 32-image request) — all config-level regressions, cheap to test (§5.3–5.8).

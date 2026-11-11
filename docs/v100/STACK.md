# SM70 stack map

## Repository layout

This fork carries SM70/V100 support as a **stack of ten slices** on top of upstream
`sgl-project/sglang` (`2c5b680aa6`, 2026-11-11). Each slice is one commit, grouped by subsystem,
with disjoint file sets — so the chain is mechanical to rebase:

| # | Slice | Scope |
|---|---|---|
| 1 | `01-kernels` | `python/sglang/kernels/jit/csrc/sm70_*`, ops wrappers, capability gates |
| 2 | `02-attention` | TileLang paged decode/prefill, QSA sparse attention, FlashAttention-v100 |
| 3 | `03-moe-quant` | marlin_v100, NVFP4 decode/top-k, opt-in FP8/FP16 expert kernels |
| 4 | `04-models` | Qwen4-Exp/MTP, DeepSeek-V4, GLM-5.3, MiniMax adaptations |
| 5 | `05-cache-runner` | QSA pending ring, DSV4.1 budget, HiCache/warmup hooks |
| 6 | `06-speculative` | TP-local draft top-1, verified prompt lookup |
| 7 | `07-env-mm-docs` | env registry, multimodal, `docs/v100/*`, `SM70.md` |
| 8 | `08-serving` | PCIe all-reduce, scheduler, loader hooks |
| 9 | `09-build` | docker, vendored patches, `requirements.txt` (cu128) |
| 10 | `10-tests` | the V100 test suite |

The slice boundaries are file-set based, so a slice's file count moves as fixes land inside it; `git show --stat <slice>` is the source of truth. Credits and third-party licenses, including the
predecessor fork the slices were assembled and rebased from, are in [`NOTICE`](../../NOTICE).

## How to rebase onto a newer upstream

```bash
git fetch upstream main
git rebase --onto upstream/main <last-known-base> sm70/stack
```

The slices are disjoint by path, so conflicts are rare and usually confined to
`python/sglang/srt/managers/scheduler.py`, `python/sglang/kernels/aot/CMakeLists.txt`,
`python/sglang/srt/layers/moe/{topk.py,fused_moe_triton/layer.py}` and `environ.py` (registry field
insertion only). Before pushing, run:

```bash
python -m py_compile $(git diff --name-only <base> HEAD -- '*.py')   # syntax
python scripts/stack/import_check.py upstream/main                    # import resolution (star-imports excluded)
```

The last rebase (eb73a50b1d -> 2c5b680aa6, 99 upstream commits across 679 files) produced exactly
seven conflicted files; all were resolved by keeping upstream's newer structure and re-laying the
SM70 delta on top. Three more breakages were silent and had to be fixed afterwards:

- `python/sglang/srt/managers/schedule_policy.py` — the rebase glued the pre-rebase fragment of
  `add_one_req` into upstream's rewritten body, which derives the input length in
  `_select_prefill_admission` and no longer defines `cand_extend_input_len`: the scheduler died with
  `NameError` on the first prefill. The stale fragment was dropped; SWA accounting lives in
  `PrefillBudget` now.
- `python/sglang/kernels/aot/CMakeLists.txt` — the fork's triton
  `install(DIRECTORY ... triton_kernels)` block pointed at the `repo-triton` `FetchContent` upstream
  had removed, so it was dropped.
- `python/sglang/srt/layers/quantization/sm70_fp16_moe.py` — upstream moved the runner state into
  `UnquantizedFusedMoEMethod.__init__` and dropped `use_triton_kernels`, so the SM70 subclass had to
  call `super().__init__()`.

## Rebuilding the chain from a resolved tree

If a rebase resolves to a single big tree, the slices can be regenerated mechanically:

```bash
pwsh scripts/stack/rebuild_ordered.ps1 -Repo <this clone> -Base upstream/main \
  -Source <resolved-branch> -Target sm70/stack \
  -ExpectPath scripts/serve_qwen38_flash_next_nvfp4_v100.sh   # repeatable; fails if the replay drops a file
```

It replays each slice's file set in order (tolerating paths upstream deleted) and reproduces the
resolved tree exactly.

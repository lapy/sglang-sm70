## Repository layout

This fork carries SM70/V100 support as a **stack of ten slices** on top of upstream
`sgl-project/sglang` (`2c5b680aa6`, 2026-11-11). Each slice is one commit, grouped by subsystem,
with disjoint file sets — so the chain is mechanical to rebase:

| # | Slice | Files | Scope |
|---|---|---|---|
| 1 | `01-kernels` | 120 | `python/sglang/kernels/jit/csrc/sm70_*`, ops wrappers, capability gates |
| 2 | `02-attention` | 48 | TileLang paged decode/prefill, QSA sparse attention, FlashAttention-v100 |
| 3 | `03-moe-quant` | 25 | marlin_v100, NVFP4 decode/top-k, opt-in FP8/FP16 expert kernels |
| 4 | `04-models` | 25 | Qwen4-Exp/MTP, DeepSeek-V4, GLM-5.3, MiniMax adaptations |
| 5 | `05-cache-runner` | 24 | QSA pending ring, DSV4.1 budget, HiCache/warmup hooks |
| 6 | `06-speculative` | 13 | TP-local draft top-1, verified prompt lookup |
| 7 | `07-env-mm-docs` | 80 | env registry, multimodal, `docs/v100/*`, `SM70.md` |
| 8 | `08-serving` | 47 | PCIe all-reduce, scheduler, loader hooks |
| 9 | `09-build` | 16 | docker, vendored patches, `requirements.txt` (cu128) |
| 10 | `10-tests` | 136 | the V100 test suite |

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
python ../sm70-stack/import-check.py upstream/main                    # import resolution (star-imports excluded)
```

The last rebase (eb73a50b1d -> 2c5b680aa6, 99 upstream commits across 679 files) produced exactly
seven conflicted files; all were resolved by keeping upstream's newer structure and re-laying the
SM70 delta on top.

## Rebuilding the chain from a resolved tree

If a rebase resolves to a single big tree, the slices can be regenerated mechanically:

```bash
pwsh sm70-stack/rebuild-ordered.ps1 -Repo <this clone> -Base upstream/main -Source <resolved-branch> -Target sm70/stack
```

It replays each slice's file set in order (tolerating paths upstream deleted) and reproduces the
resolved tree exactly.

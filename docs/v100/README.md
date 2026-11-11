# SM70 (V100 / Volta) documentation

Everything about running SGLang on 32 GB V100s: installation, the model recipes, the performance
work and the stack itself. The fork is **sglang-sm70**; `main` carries all ten SM70 slices on top of
upstream `sgl-project/sglang` at `2c5b680aa6`. Credits and third-party licenses are in
[`NOTICE`](../../NOTICE).

> **Measurements.** Numbers labelled "this fork" were measured on 4× V100-SXM2-32GB at a **150 W cap**
> with the benchmark harness in `PERFORMANCE-TUNING.md` (warm, clocks ≥1500 MHz, medians of five).
> Numbers labelled "predecessor" are quoted from the records of the fork this stack was assembled
> and rebased from (`sglang-sxm2`) and may come from different hardware or power limits. Absolute
> throughput is only comparable within one power configuration.

## Start here

| document | what it covers |
|---|---|
| [`INSTALL.md`](INSTALL.md) | Build from source: CUDA 12.6 toolkit, patched deps, the Marlin extensions, Docker |
| [`Qwen3.8-Flash-Next.md`](Qwen3.8-Flash-Next.md) | The primary validation target: TP4, MTP, QSA, NVFP4, measured performance |
| [`PERFORMANCE-TUNING.md`](PERFORMANCE-TUNING.md) | What we changed and why, before/after, rejected knobs, roadmap, measurement caveat |
| [`STACK.md`](STACK.md) | The ten-slice layout, how to rebase onto a newer upstream, remaining conflict points |

## Other model recipes

| document | setup |
|---|---|
| [`GLM-5.3-Flash.md`](GLM-5.3-Flash.md) | 8× V100, TP8, MTP |
| [`DeepSeek-V4.1-Flash.md`](DeepSeek-V4.1-Flash.md) | 8× V100, DSpark speculative decoding, sticky last-sequence cache |
| [`MiniMax-H3.md`](MiniMax-H3.md) | video generation on V100 |

## Performance research

| document | what it is |
|---|---|
| [`perf/forks-measured-results.md`](perf/forks-measured-results.md) | Performance archaeology: every measured number in the predecessor fork and `sglang-V100`, with citations |
| [`perf/knob-inventory.md`](perf/knob-inventory.md) | Every SM70 tunable: default, effect, how to A/B it |
| [`perf/architecture-analysis.md`](perf/architecture-analysis.md) | Where the 26 ms MTP decode step actually goes on this hardware |

## Scripts

The recipes live in [`scripts/`](../../scripts):

| script | purpose |
|---|---|
| `install_v100.sh` | Installs the cu128 dependency set, patched FlashInfer, `sglang-kernel`, Marlin |
| `setup_v100_marlin.sh` | Builds and installs the dense + MoE `marlin_v100` extensions |
| `serve_qwen38_flash_next_nvfp4_v100.sh` | The Qwen3.8-Flash-Next-NVFP4 recipe (`mtp` / `mtp-alt` / `nomtp` modes) |
| `serve_glm53_flash_nvfp4_v100.sh` | The GLM-5.3-Flash recipe |

## Known caveats

- **Power and clocks.** The reference box caps the V100s at 150 W and drops them to 135 MHz after
  120 s idle with a ramp on wake. Warm up before measuring; a cold first request measures the ramp.
- **`SGLANG_SM70_MTP_LOCAL_TOP1` deadlocks** the scheduler (watchdog, no progress). Leave it off.
- **PLE embedding offload is mandatory** on 4×32 GB; on-device placement OOMs.
- **Environment variables.** The 39 `SGLANG_SM70_*` / `SGLANG_V100_*` knobs are documented in the
  [environment reference](../docs/references/environment_variables.mdx); `PERFORMANCE-TUNING.md` says
  which ones actually move throughput.

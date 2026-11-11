# Optional grouped Page4 QSA dependency

`scripts/setup_v100_grouped_qsa.py` builds only the grouped planner and attention
bindings from the Flash-V100 sources bundled in
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM/tree/d8f42b391af06410ab01bb71657ededf7dce7799/flash-attention-v100).
The installed binary is a local build artifact and is not committed. It is
written to SGLang's SM70 prebuilt directory
(`python/sglang/kernels/prebuilt/_sm70_grouped_qsa_v100.so`, resolved through
`sglang.kernels.sm70_paths` and overridable with `SGLANG_SM70_PREBUILT_DIR`).

The underlying Flash-V100 project is by D. Skryabin / ai-bond. The grouped Page4
work was contributed by yangzhuxinyzx in
[1Cat-vLLM #387](https://github.com/1CatAI/1Cat-vLLM/pull/387).
The BSD-3-Clause and upstream Apache-2.0 license texts are retained here.
SGLang's adapter uses the library through its planner/attention API and keeps
the existing native QSA path available.

The shared library must use the same PyTorch ABI as the serving environment.
Source and output hashes are written to the build directory's `metadata.json`.

Opt in with `SGLANG_SM70_QSA_GROUPED_PREFILL=1`. The adapter is limited to the
measured 2,048-row FP16/Page16 Qwen shape; a missing library, another shape, or
the flag being unset keeps the native QSA prefill.

## Optional vision attention

`--mm-attention-backend fa2_v100` (or `SGLANG_SM70_VISION_ATTENTION=1`, which
only replaces the platform default) uses the separate `flash_attn_v100` Python
package from [ai-bond/flash-attention-v100](https://github.com/ai-bond/flash-attention-v100).
The validated host build uses revision `d89800edf608d85744f3ab6188be5fd0736acf39`
with `vision_host_compat.patch`, which records the existing host's CUDA 12.8,
packaging, TorchBind optional-tensor and paged-layout compatibility changes.
These are separate from the grouped QSA patch above. The vision adapter uses
non-paged, bidirectional variable-length attention.

Qwen3.8 has 72-wide vision heads. The adapter zero-pads them to the supported
128-wide kernel, retains the original attention scale, and crops the output.
It requires SM70, FP16 and eager vision execution. Language CUDA graphs remain
enabled. Numerical tests cover ragged images, custom scaling, noncontiguous
inputs and bidirectional attention without mixing images.

Build in the serving environment, with its existing PyTorch installation:

```bash
conda activate sglang-v100
q38_repo="$PWD"  # Run from the SGLang checkout.
git clone https://github.com/ai-bond/flash-attention-v100.git flash-attention-v100-vision
git -C flash-attention-v100-vision checkout --detach d89800edf608d85744f3ab6188be5fd0736acf39
git -C flash-attention-v100-vision apply \
  "$q38_repo/python/sglang/kernels/third_party/flash_v100/vision_host_compat.patch"
MAX_JOBS=2 NVCC_THREADS=2 python -m pip install --no-deps --no-build-isolation \
  ./flash-attention-v100-vision
PYTHONPATH=python python -m pytest -q test/manual/layers/test_sm70_vision_attention.py
```

The performance evidence records the measured binary and interface hashes.
This round reused that installed library; it did not rebuild a Docker image.
The in-tree Triton vision path remains available with
`--mm-attention-backend triton_attn`, and it is what
`_determine_attention_backend` keeps when the package is absent.

## Building on the pinned CUDA 12.6 toolkit

The recipe above assumes CUDA 12.8 + torch 2.9.1. This stack pins **CUDA 12.6**, because that
nvcc is the last one that still emits sm_70 offline, and torch 2.8/2.9. Two changes are needed,
one per patch ([`sm70-cuda126-floor.patch`](sm70-cuda126-floor.patch) and
[`sm70-tanh-guard.patch`](sm70-tanh-guard.patch)):

1. `setup.py` refuses any CUDA below 12.8. The extension only ever builds
   `arch=compute_70,code=sm_70`, so 12.6 is sufficient and the floor moves to 12.6.
2. `include/mat_mul.h` calls `__tanhf`, a fast-tanh that exists in the device ISA only from
   sm_75. The call sits in the softcap branch (`IS_SOFTCAP`), which is instantiated even when
   softcap is off because the name is non-dependent, so CUDA 12.6 headers reject it with
   "identifier __tanhf is undefined in device code". The patch routes it through a
   `sm70_tanhf` helper that falls back to device `tanhf` below sm_75 (Qwen3.8 uses softcap 0,
   so the branch is dead at runtime).

`scripts/install_v100.sh` runs all of this for you via
[`scripts/setup_fa_v100_vision.sh`](../../../../../scripts/setup_fa_v100_vision.sh) (skip with
`SGLANG_V100_SKIP_FA_V100=1`); the Dockerfile does the same. To do it by hand, build in the
serving venv (V100-only arch keeps it to a few minutes). The four patches must be
applied **in this order** -- `vision_host_compat` and `torch291` both touch `pyproject.toml`, so
applying `torch291` first makes the host-compat patch fail at `pyproject.toml:17`:

```bash
TP=$PWD/python/sglang/kernels/third_party/flash_v100
git clone https://github.com/ai-bond/flash-attention-v100.git flash-attention-v100-vision
git -C flash-attention-v100-vision checkout --detach d89800edf608d85744f3ab6188be5fd0736acf39
for p in "$TP/vision_host_compat.patch" \
         "$TP/../../../../../patches/flash-attention-v100-torch291.patch" \
         "$TP/sm70-cuda126-floor.patch" \
         "$TP/sm70-tanh-guard.patch" ; do
  git -C flash-attention-v100-vision apply "$p" || exit 1
done
TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=2 NVCC_THREADS=2 \
  python -m pip install --no-deps --no-build-isolation ./flash-attention-v100-vision
```

| patch | what it does | why we need it |
|---|---|---|
| `vision_host_compat.patch` | TorchBind empty-tensor handling, 3D `[N,H,D]` paged KV,
  license field, CUDA floor 12.9 -> 12.8 | inherited verbatim from the sglang-V100 port |
| `patches/flash-attention-v100-torch291.patch` | build requirement `torch 2.10` -> `2.9.1` |
  inherited verbatim; that is how the library became buildable on a V100 host at all |
| `sm70-cuda126-floor.patch` | CUDA floor 12.8 -> 12.6 | our pinned toolkit |
| `sm70-tanh-guard.patch` | routes `__tanhf` through an sm_75-guarded helper | ours; see below |

`grouped_masked_kv.patch` is carried but **not applied**: it masks K/V loads on the grouped
Page4 decode path, and our grouped build comes from the copy bundled in 1Cat-vLLM via
[`scripts/setup_v100_grouped_qsa.py`](../../../../../scripts/setup_v100_grouped_qsa.py), not from
this tree. Apply it there if that path is ever enabled.

Verified on this stack: `test/manual/layers/test_sm70_vision_attention.py` 9/9 pass, and a
standalone numeric check against a float reference gives `max_abs_err <= 0.001` in FP16.
The kernel supports `D` in {32, 64, 128, 256} only, which is why the 72-wide Qwen vision heads
are zero-padded to 128 by the adapter above.

**Prebuilt wheels do not apply here.** The published wheels (26.06 and 26.10) both pin
`torch @ .../cu129/torch-2.10.0+cu129` in their metadata; importing them against our torch
2.8/2.9 fails with `undefined symbol: c10_cuda_check_implementation`. They do work on a
torch 2.10 + cu129 install (verified on a V100: arch list includes sm_70), so a wheel is an
option only for a host that has moved to torch 2.10.

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One-block builder + installer for the SM70 (V100) Marlin kernels.
#
# Builds https://github.com/zhinianqin/marlin_v100 against the *currently
# active* Python env's torch (must match the torch SGLang will run with),
# then installs the resulting dense and MoE extensions into SGLang's SM70
# prebuilt dir so they are auto-detected at runtime on SM70 (no env var needed).
#
# Usage (from any directory):
#   conda activate sglang-v100            # or whichever env runs sglang
#   bash scripts/setup_v100_marlin.sh
#
# Re-run is safe: it rebuilds incrementally and re-installs the .so.
#
# Knobs (env vars, all optional):
#   MARLIN_V100_REPO   path to an existing marlin_v100 checkout (default: ~/marlin_v100)
#   MARLIN_V100_REF    pinned git revision (default: known-good SM70 base)
#   CUTLASS_DIR        path to a CUTLASS checkout with include/{cute/cutlass}
#   CUDAHOSTCXX        CUDA host compiler (default: auto-detect g++-12)
#   MARLIN_V100_INSTALL_DIR install artifacts here instead of importing sglang
#   MARLIN_V100_SKIP_SMOKE  set to 1 to validate in a later cached Docker layer
#   MARLIN_V100_SKIP_BF16_COMPAT set to 1 with CUDA 12.8+ toolkit headers
#   MAX_JOBS, NVCC_THREADS  parallelism handed to the build

set -euo pipefail

log() { printf '\033[1;34m[setup_v100_marlin]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup_v100_marlin WARN]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[setup_v100_marlin ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

PYTHON="${PYTHON:-python}"
"$PYTHON" -c 'import sys, torch; assert sys.version_info[:2] == (3,12) or True' \
  || die "PYTHON ('$PYTHON') is not usable."
"$PYTHON" -c 'import torch' || die "torch is not importable in the active env; activate the sglang env first."
TORCH_VER="$("$PYTHON" -c 'import torch; print(torch.__version__)')"
CUDA_VER="$("$PYTHON" -c 'import torch; print(torch.version.cuda)')"
log "active python: $("$(which "$PYTHON")" -c 'import sys; print(sys.executable)')"
log "torch=${TORCH_VER}  cuda=${CUDA_VER}"

# --- locate / clone marlin_v100 -------------------------------------------------
REPO="${MARLIN_V100_REPO:-$HOME/marlin_v100}"
MARLIN_V100_REF="${MARLIN_V100_REF:-6d72a49939701d26b15b617a4cd2423174adb2d1}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/patches"
SM70_PATCHES=(
  "$PATCH_DIR/marlin-v100-qwen-sm70-tuning.patch"
  "$PATCH_DIR/marlin-v100-qwen38-nvfp4-tuning.patch"
)
if [[ "${MARLIN_V100_SKIP_BF16_COMPAT:-0}" != 1 ]]; then
  SM70_PATCHES+=("$PATCH_DIR/marlin-v100-sm70.patch")
fi
if [[ ! -d "$REPO/.git" ]]; then
  log "cloning marlin_v100 to $REPO"
  git clone https://github.com/zhinianqin/marlin_v100.git "$REPO"
  git -C "$REPO" checkout --detach "$MARLIN_V100_REF"
else
  log "using existing marlin_v100 at $REPO"
  if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$MARLIN_V100_REF" ]]; then
    git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || \
      die "$REPO has local changes on another revision; move it aside or set MARLIN_V100_REPO."
    git -C "$REPO" fetch origin "$MARLIN_V100_REF"
    git -C "$REPO" checkout --detach "$MARLIN_V100_REF"
  fi
fi

# Tree of MARLIN_V100_REF with the first $1 SM70 patches applied, built in a
# scratch index so the checkout is not touched.
patched_tree() {
  local index patch status=0
  index="$(mktemp)"
  GIT_INDEX_FILE="$index" git -C "$REPO" read-tree "$MARLIN_V100_REF" || status=1
  for patch in "${SM70_PATCHES[@]:0:$1}"; do
    (( status )) || GIT_INDEX_FILE="$index" git -C "$REPO" apply --cached "$patch" || status=1
  done
  (( status )) || GIT_INDEX_FILE="$index" git -C "$REPO" write-tree || status=1
  rm -f "$index"
  return "$status"
}

# Do the checkout's copies of the patched files equal those in tree $1?
# A file missing on both sides counts as equal.
checkout_matches() {
  local path want have
  for path in "${PATCHED_PATHS[@]}"; do
    want="$(git -C "$REPO" rev-parse -q --verify "$1:$path" || true)"
    have=""
    if [[ -f "$REPO/$path" ]]; then have="$(git -C "$REPO" hash-object -- "$path")"; fi
    [[ "$want" == "$have" ]] || return 1
  done
}

for SM70_PATCH in "${SM70_PATCHES[@]}"; do
  [[ -f "$SM70_PATCH" ]] || die "missing SM70 compatibility patch: $SM70_PATCH"
done
# The patches stack (later ones edit lines that earlier ones add), so no single
# patch can be checked on its own. Find how many the checkout already carries.
mapfile -t PATCHED_PATHS < <(git -C "$REPO" apply --numstat "${SM70_PATCHES[@]}" | cut -f3 | sort -u)
APPLIED=""
for (( n = ${#SM70_PATCHES[@]}; n >= 0; n-- )); do
  tree="$(patched_tree "$n")" \
    || die "SM70 compatibility patches do not apply cleanly to $MARLIN_V100_REF"
  if checkout_matches "$tree"; then APPLIED="$n"; break; fi
done
[[ -n "$APPLIED" ]] \
  || die "$REPO has local changes in the patched files; move it aside or set MARLIN_V100_REPO."
for SM70_PATCH in "${SM70_PATCHES[@]:0:$APPLIED}"; do
  log "already applied: $(basename "$SM70_PATCH")"
done
for SM70_PATCH in "${SM70_PATCHES[@]:$APPLIED}"; do
  git -C "$REPO" apply "$SM70_PATCH"
  log "applied: $(basename "$SM70_PATCH")"
done

# --- toolchain: CUDA, host compiler, CUTLASS -----------------------------------
[[ -n "${CUDA_HOME:-}" ]] || CUDA_HOME="/usr/local/cuda-${CUDA_VER}"
[[ -d "$CUDA_HOME" ]] || CUDA_HOME="/usr/local/cuda"
[[ -d "$CUDA_HOME/bin" ]] || die "CUDA_HOME ($CUDA_HOME) has no bin/; set CUDA_HOME manually."
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
log "CUDA_HOME=$CUDA_HOME  nvcc=$(nvcc --version | tail -1 | awk '{print $2}')"

# gcc-13 hits the well-known _Float128 / glibc error in CUDA's <mathcalls.h>;
# gcc-12 is the known-good host compiler for sm70 on this toolchain.
if [[ -z "${CUDAHOSTCXX:-}" ]]; then
  for cand in /usr/bin/g++-12 /usr/bin/g++-11 /usr/bin/g++-10; do
    [[ -x "$cand" ]] && CUDAHOSTCXX="$cand" && break
  done
  [[ -n "${CUDAHOSTCXX:-}" ]] || die "no g++-12/11/10 found; install g++-12 or set CUDAHOSTCXX."
fi
export CUDAHOSTCXX
log "CUDAHOSTCXX=$CUDAHOSTCXX"

if [[ -z "${CUTLASS_DIR:-}" ]]; then
  for cand in \
    "$HOME/GooseLLM/.deps/cutlass-src" \
    "$HOME/tilelang/3rdparty/cutlass" \
    "$HOME/cutlass" ; do
    if [[ -f "$cand/include/cute/tensor.hpp" && -f "$cand/include/cutlass/cutlass.h" ]]; then
      CUTLASS_DIR="$cand"; break
    fi
  done
  [[ -n "${CUTLASS_DIR:-}" ]] || die "CUTLASS not found; set CUTLASS_DIR to a checkout with include/{cute,cutlass}."
fi
export CUTLASS_DIR
log "CUTLASS_DIR=$CUTLASS_DIR"

# --- make marlin_v100's build.sh use the active env's python --------------------
mkdir -p "$REPO/.venv/bin"
ACTIVE_PYTHON="$("$(which "$PYTHON")" -c 'import sys; print(sys.executable)')"
# A symlink to a venv interpreter is not sufficient: Python discovers the
# environment from pyvenv.cfg beside the symlink and silently falls back to the
# system prefix. A forwarding launcher preserves both venv and Conda prefixes.
# Remove an existing venv symlink first: shell redirection follows symlinks and
# would otherwise overwrite the active environment's Python executable.
rm -f "$REPO/.venv/bin/python"
printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$ACTIVE_PYTHON" > "$REPO/.venv/bin/python"
chmod +x "$REPO/.venv/bin/python"

# CMake stores absolute Torch paths. If this checkout was previously built
# from another Conda environment, discard only the stale CMake build tree so
# the extension is actually linked against the active environment.
ACTIVE_PREFIX="$("$PYTHON" -c 'import sys; print(sys.prefix)')"
CMAKE_CACHE="$(find "$REPO/build" -name CMakeCache.txt -print -quit 2>/dev/null || true)"
if [[ -n "$CMAKE_CACHE" ]] && ! grep -Fq "$ACTIVE_PREFIX" "$CMAKE_CACHE"; then
  STALE_BUILD_DIR="$(dirname "$CMAKE_CACHE")"
  warn "removing stale CMake cache from another Python env: $STALE_BUILD_DIR"
  rm -rf "$STALE_BUILD_DIR"
fi

# --- build ----------------------------------------------------------------------
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
# keep ptxas quiet unless explicitly requested (cuts log noise massively)
export CMAKE_ARGS="${CMAKE_ARGS:-}"
log "building (MAX_JOBS=$MAX_JOBS) ... this takes ~15-40 min"
( cd "$REPO" && bash "$REPO/build.sh" )

# --- locate the built artifacts -------------------------------------------------
# The runtime loaders match these exact basenames; any other name is invisible
# to them and the SM70 Marlin path silently falls back to SGLang's zero-output
# JIT stub (dense quantized linears become zeros and the model echoes its prompt).
SO_MOE="$REPO/vllm/_moe_C.abi3.so"
SO_DENSE="$REPO/vllm/_C.abi3.so"
[[ -f "$SO_MOE" ]] || die "build finished but $SO_MOE was not produced."
[[ -f "$SO_DENSE" ]] || die "build finished but $SO_DENSE was not produced."
log "built MoE extension:   $SO_MOE"
log "built dense extension: $SO_DENSE"

# --- install into SGLang's SM70 prebuilt dir ------------------------------------
# RFC #29630 retired sglang.jit_kernel; sglang.kernels.sm70_paths is now the one
# place that knows where the prebuilt .so live, and the runtime loaders read it too.
if [[ -n "${MARLIN_V100_INSTALL_DIR:-}" ]]; then
  PKG_DIR="$MARLIN_V100_INSTALL_DIR"
else
  PKG_DIR="$("$PYTHON" - <<'PYPATH' || die "could not locate the SGLang SM70 prebuilt dir."
try:
    from sglang.kernels.sm70_paths import sm70_prebuilt_dir
except Exception as e:
    raise SystemExit(f"cannot import sglang.kernels.sm70_paths: {e}")
print(sm70_prebuilt_dir())
PYPATH
)"
fi
mkdir -p "$PKG_DIR"

# Both exports are required: the dense one supplies the SM70 repack/GEMM used
# by dense quantized linears, the MoE one the routed-expert kernel.
DEST_MOE="$PKG_DIR/_sm70_marlin_v100_moe.abi3.so"
DEST_DENSE="$PKG_DIR/_sm70_marlin_v100_dense.abi3.so"
cp -f "$SO_MOE" "$DEST_MOE"
cp -f "$SO_DENSE" "$DEST_DENSE"
log "installed -> $DEST_MOE"
log "installed -> $DEST_DENSE"

# --- runtime smoke test ---------------------------------------------------------
# Verify through SGLang's own loader, not just by filename: the dense loader
# globs _sm70_marlin_v100_dense*.so and the MoE loader globs
# _sm70_marlin_v100_moe*.so, and a mismatch is silent. The in-tree repack is a
# zero-output stub below sm_80, so a missed .so zeroes every dense quantized
# linear and the model echoes its prompt instead of answering.
if [[ "${MARLIN_V100_SKIP_SMOKE:-0}" != 1 ]]; then
  log "smoke test: MoE op registration + loader visibility of both extensions"
  "$PYTHON" - <<PY || die "smoke test failed: SGLang's SM70 loader cannot resolve both extensions installed under $PKG_DIR (expected $DEST_MOE and $DEST_DENSE). A missed dense repack silently zeroes every dense quantized linear; fix the file names before deploying."
import torch
from sglang.srt.layers.quantization.marlin_utils import _sm70_marlin_v100_repack_ops
torch.ops.load_library("$DEST_MOE")
assert hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm"), "MoE op not registered"
gptq_repack, awq_repack = _sm70_marlin_v100_repack_ops()
assert gptq_repack is not None, "dense gptq_marlin_repack not found by SGLang's loader"
assert awq_repack is not None, "dense awq_marlin_repack not found by SGLang's loader"
print("ok: MoE op registered; dense gptq/awq repack resolved by SGLang's loader")
PY
else
  log "smoke test deferred to a separate cached build layer"
fi

log "done. SGLang on SM70 will now auto-detect and use this kernel for AWQ/GPTQ/compressed-tensors MoE."
log "tuning knobs (optional): SM70_MARLIN_MOE_CTA_GEOMETRY, SM70_MARLIN_MOE_SPLIT_K, SM70_MARLIN_MOE_METADATA_CACHE"

#!/usr/bin/env bash
# Install the optional FlashAttention-V100 vision kernels by default; the runtime
# environment variable SGLANG_SM70_VISION_ATTENTION (default off) decides whether they are used.
#
#   SGLANG_V100_SKIP_FA_V100=1     skip the build entirely
#   SGLANG_V100_FA_V100_REF=<sha>  override the pinned revision
#   SGLANG_V100_FA_V100_URL=<url>  override the repository (mirror)
set -u
V=${SGLANG_V100_DEPS_DIR:-$HOME/.cache/sglang-v100-sources}
P=${SGLANG_V100_PYTHON:-python}
REPO_ROOT=${REPO_ROOT:-$PWD}
TP=$REPO_ROOT/python/sglang/kernels/third_party/flash_v100
REF=${SGLANG_V100_FA_V100_REF:-d89800edf608d85744f3ab6188be5fd0736acf39}
URL=${SGLANG_V100_FA_V100_URL:-https://github.com/ai-bond/flash-attention-v100.git}
DIR=$V/flash-attention-v100-vision

log() { printf '\n\033[1;34m[install_v100]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[install_v100] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

if [ "${SGLANG_V100_SKIP_FA_V100:-0}" = "1" ]; then
  log "Skipping FlashAttention-V100 (SGLANG_V100_SKIP_FA_V100=1); vision stays on triton_attn"
  exit 0
fi

if "$P" -c "import flash_attn_v100" >/dev/null 2>&1; then
  log "FlashAttention-V100 already importable; leaving it alone"
  exit 0
fi

log "Installing FlashAttention-V100 vision kernels (runtime-gated, costs a few minutes)"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch -q --all || true
else
  mkdir -p "$(dirname "$DIR")"
  git clone -q "$URL" "$DIR" || die "clone failed: $URL"
fi
git -C "$DIR" checkout -q --detach "$REF" || die "cannot check out $REF"

# Order matters: vision_host_compat and torch291 both touch pyproject.toml.
for p in "$TP/vision_host_compat.patch" \
         "$REPO_ROOT/patches/flash-attention-v100-torch291.patch" \
         "$TP/sm70-cuda126-floor.patch" \
         "$TP/sm70-tanh-guard.patch" ; do
  [ -f "$p" ] || die "missing patch: $p"
  if git -C "$DIR" apply --check "$p" 2>/dev/null; then
    git -C "$DIR" apply "$p" || die "apply failed: $p"
    printf '  applied %s\n' "$(basename "$p")"
  elif git -C "$DIR" apply --reverse --check "$p" 2>/dev/null; then
    printf '  already applied %s\n' "$(basename "$p")"
  else
    die "cannot apply $p (revision or tree changed?)"
  fi
done

TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS="${MAX_JOBS:-2}" NVCC_THREADS="${NVCC_THREADS:-2}" \
  "$P" -m pip install --no-deps --no-build-isolation "$DIR" \
  || die "build failed; see python/sglang/kernels/third_party/flash_v100/README.md"

"$P" -c "from flash_attn_v100 import flash_attn_varlen_func" \
  || die "flash_attn_v100 installed but does not import"

log "FlashAttention-V100 installed. Vision attention still uses triton_attn by default."
printf '  Activate it per deployment with SGLANG_SM70_VISION_ATTENTION=1 (or --mm-attention-backend fa2_v100).\n'
printf '  It wins on multi-image/document and video requests; single small images are slightly faster on triton.\n'
printf '  See docs/v100/PERFORMANCE-TUNING.md#vision-attention-is-regime-dependent\n'

"""Pre-load SM70 Triton specializations that first-request warmup otherwise
device-loads after serving starts, when CUDA-free is often <0.5 GiB.

``cuModuleLoadData`` needs free device memory *outside* the torch caching
allocator. Call this from ModelRunner.prewarm_sampling (after graphs, before
``mark_serving_started``): empty the cache, launch dummy grids that match the
np=1 / page_size=256 / context-len pool specializations, then empty again.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import get_available_gpu_memory, next_power_of_2
import inspect

logger = logging.getLogger(__name__)


def maybe_prewarm_sm70_runtime_kernels(
    *,
    device: str,
    page_size: int,
    pool_len: int,
    model: Any,
) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        if not get_platform().is_sm70:
            return
    except Exception:
        return

    page_size = int(page_size) or 1
    pool_len = int(pool_len) or 1
    _empty_cache()
    logger.info(
        "SM70 runtime-kernel prewarm: CUDA-free %.2f GiB (page_size=%d pool_len=%d)",
        _cuda_free_gb(),
        page_size,
        pool_len,
    )
    try:
        _prewarm_allocator_kernels(device=device, page_size=page_size, pool_len=pool_len)
        _prewarm_engram_commit(device=device, model=model)
        torch.cuda.synchronize()
    except Exception:
        logger.warning("SM70 runtime-kernel prewarm failed", exc_info=True)
    finally:
        _empty_cache()
        logger.info(
            "SM70 runtime-kernel prewarm done: CUDA-free %.2f GiB", _cuda_free_gb()
        )


def _empty_cache() -> None:
    torch.cuda.empty_cache()


def _cuda_free_gb() -> float:
    try:
        return float(
            get_available_gpu_memory(
                "cuda", torch.cuda.current_device(), empty_cache=False
            )
        )
    except Exception:
        return -1.0



def _prewarm_launch(kernel, grid, **kwargs) -> None:
    """Launch a Triton kernel for warmup, binding only the arguments it declares.

    These kernels are upstream code whose signatures move between rebases:
    `alloc_extend_kernel` lost its `bs_upper` argument in 2c5b680aa6 (the batch now comes
    from `program_id`), and passing one extra positional argument makes Triton fail at bind
    time with "dynamic_func() takes N positional arguments but N+1 were given". Binding by
    name means a warmup that merely warms cannot break the engine when a parameter is added
    or removed; a *required* argument that disappeared still raises, just with a clear error.
    """
    fn = getattr(kernel, "fn", kernel)
    try:
        accepted = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        accepted = set(kwargs)
    bound = {name: value for name, value in kwargs.items() if name in accepted}
    kernel[grid](**bound)


def _prewarm_allocator_kernels(*, device: str, page_size: int, pool_len: int) -> None:
    from sglang.kernels.ops.memory.allocator import alloc_extend_kernel
    from sglang.kernels.ops.memory.common import get_last_loc_triton
    from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs
    from sglang.srt.mem_cache.allocation import assign_req_to_token_pool

    bs = 1
    bs_upper = next_power_of_2(bs)
    prefix_lens = torch.zeros((bs,), dtype=torch.int64, device=device)
    seq_lens = torch.full((bs,), page_size, dtype=torch.int64, device=device)
    last_loc = torch.full((bs,), -1, dtype=torch.int64, device=device)
    free_pages = torch.arange(16, dtype=torch.int64, device=device)
    out_indices = torch.empty((page_size,), dtype=torch.int64, device=device)
    _prewarm_launch(
        alloc_extend_kernel,
        (bs,),
        pre_lens_ptr=prefix_lens,
        seq_lens_ptr=seq_lens,
        last_loc_ptr=last_loc,
        free_page_ptr=free_pages,
        out_indices=out_indices,
        bs_upper=bs_upper,
        page_size=page_size,
    )

    dummy_pool = torch.zeros((2, pool_len), dtype=torch.int32, device=device)
    req_pool_indices = torch.zeros((bs,), dtype=torch.int32, device=device)
    prefix_one = torch.ones((bs,), dtype=torch.int64, device=device)
    get_last_loc_triton(dummy_pool, req_pool_indices, prefix_one)

    start_offset = torch.zeros((bs,), dtype=torch.int64, device=device)
    end_offset = torch.full((bs,), 8, dtype=torch.int64, device=device)
    out_cache_loc = torch.arange(8, dtype=torch.int64, device=device)
    _prewarm_launch(
        assign_req_to_token_pool,
        (bs,),
        req_pool_indices=req_pool_indices,
        req_to_token=dummy_pool,
        start_offset=start_offset,
        end_offset=end_offset,
        out_cache_loc=out_cache_loc,
        pool_len=pool_len,
        bs_upper=bs_upper,
    )
    _prewarm_launch(
        assign_extend_cache_locs,
        (bs,),
        req_pool_indices=req_pool_indices,
        req_to_token=dummy_pool,
        start_offset=start_offset,
        end_offset=end_offset,
        out_cache_loc=out_cache_loc,
        pool_len=pool_len,
        bs_upper=bs_upper,
    )


def _prewarm_engram_commit(*, device: str, model: Any) -> None:
    from sglang.kernels.ops.embeddings.engram_hash import engram_commit_history

    hasher = _find_engram_hasher(model)
    if hasher is None:
        return
    history = getattr(hasher, "history", None)
    if history is None or history.ndim != 2 or history.shape[1] == 0:
        return
    width = int(history.shape[1])
    dummy_history = torch.zeros((2, width), dtype=history.dtype, device=device)
    verify_ids = torch.zeros((1, width), dtype=torch.int32, device=device)
    req_slots = torch.zeros((1,), dtype=torch.int32, device=device)
    commit_lens = torch.ones((1,), dtype=torch.int32, device=device)
    engram_commit_history(dummy_history, verify_ids, req_slots, commit_lens)


def _find_engram_hasher(model: Any) -> Optional[Any]:
    hasher = getattr(model, "engram_hasher", None)
    if hasher is not None:
        return hasher
    inner = getattr(model, "model", None)
    return getattr(inner, "engram_hasher", None)

"""The sm70 warmup must survive upstream kernel signature drift.

`_prewarm_launch` binds warmup arguments by name, so a kernel that gains or loses a
parameter does not break engine startup. This is a regression test for a real failure:
on upstream 2c5b680aa6 `alloc_extend_kernel` dropped its `bs_upper` argument, the warmup
still passed it positionally, and every boot logged

    SM70 runtime-kernel prewarm failed
    TypeError: dynamic_func() takes 6 positional arguments but 7 were given
"""

from sglang.srt.model_executor import sm70_warmup


class _FakeKernel:
    """Stands in for a triton JITFunction: records how it was called."""

    def __init__(self, params):
        self.calls = []
        self.fn = self._make_fn(params)

    @staticmethod
    def _make_fn(params):
        namespace = {}
        arglist = ", ".join(params)
        exec(f"def kern({arglist}):\n    return locals()\n", namespace)  # noqa: S102
        return namespace["kern"]

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append((grid, kwargs))
            return kwargs

        return launch


def test_filters_arguments_the_kernel_no_longer_takes():
    kernel = _FakeKernel(["pre_lens_ptr", "seq_lens_ptr", "last_loc_ptr",
                          "free_page_ptr", "out_indices", "page_size"])
    sm70_warmup._prewarm_launch(
        kernel,
        (1,),
        pre_lens_ptr="a",
        seq_lens_ptr="b",
        last_loc_ptr="c",
        free_page_ptr="d",
        out_indices="e",
        bs_upper=1,          # dropped upstream; must not reach the kernel
        page_size=16,
    )
    assert len(kernel.calls) == 1
    grid, kwargs = kernel.calls[0]
    assert grid == (1,)
    assert "bs_upper" not in kwargs
    assert kwargs["page_size"] == 16


def test_passes_everything_when_the_signature_matches():
    kernel = _FakeKernel(["req_pool_indices", "req_to_token", "start_offset",
                          "end_offset", "out_cache_loc", "pool_len", "bs_upper"])
    sm70_warmup._prewarm_launch(
        kernel,
        (2,),
        req_pool_indices=1,
        req_to_token=2,
        start_offset=3,
        end_offset=4,
        out_cache_loc=5,
        pool_len=6,
        bs_upper=7,
    )
    grid, kwargs = kernel.calls[0]
    assert grid == (2,)
    assert kwargs["bs_upper"] == 7


def test_falls_back_when_the_signature_cannot_be_inspected():
    class _Opaque:
        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def launch(**kwargs):
                self.calls.append(kwargs)

            return launch

    kernel = _Opaque()
    sm70_warmup._prewarm_launch(kernel, (1,), a=1, b=2)
    assert kernel.calls == [{"a": 1, "b": 2}]

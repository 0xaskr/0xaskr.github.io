---
title: 'Tokamax Source Code Walkthrough: From Pallas Kernels to Autotuning'
description: "A walkthrough of Tokamax's GPU/TPU kernels, Config selection, tuning caches, benchmarking, and JAX transformations, using LayerNorm as a running example with complete examples and source links."
pubDate: '2026/9/10'
tags: ["Tokamax", "JAX", "Pallas", "GPU", "TPU", "Autotuning"]
---

## Reading Conventions and Source Revision

- **Running example**: Parts 1–2 follow the Normalization operator, then compare it with Attention, Ragged Dot, and the GPU/TPU backends.
- **Code paths**: All paths are relative to the `tokamax/` package directory.
- **`# sym:X`**: Identifies the corresponding symbol in the source.
- **Prerequisites**: Familiarity with JAX arrays, `jit`, `grad`, and `vmap`.
- **Source revision**: The entire article is pinned to main commit [`f21f67cc921f6ad56785a084ba30dc3d85278e9a`][src-revision].

---

## Source Tree Overview

```text
tokamax/                                  # Package root
├── __init__.py                           # Computation APIs, Op, tuning, and timing
├── autotuning.py                         # get_bound_args / JSON specification utilities
├── benchmarking.py                       # Benchmark helper exports
├── config.py                             # Two public configuration options
├── benchmarks/                           # Benchmark entry points
├── data/autotuning/                      # 72 bundled tuning JSON files
├── experimental/utils/tuning/tpu/        # TPU tuning utilities
└── _src/
    ├── ops/
    │   ├── op.py                         # Op / BoundArguments / configuration decisions
    │   ├── normalization/
    │   │   ├── api.py                    # layer_norm dispatch
    │   │   ├── base.py                   # Normalization / VJP reference implementations
    │   │   ├── pallas_triton.py          # Forward Op, Fusion adapter, and kernel
    │   │   ├── pallas_triton_vjp.py      # Backward Op and kernel
    │   │   ├── pallas_triton_config.py   # Config / key / heuristics
    │   │   ├── pallas_triton_vjp_config.py
    │   │   └── arg_specs.py              # Tuning and test samples
    │   ├── attention/                    # Triton, Mosaic GPU/TPU, XLA, cuDNN
    │   ├── ragged_dot/                   # Grouped matmul and quantized variants
    │   ├── gated_linear_unit/            # GLU
    │   ├── linear_softmax_cross_entropy_loss/
    │   ├── triangle_multiplication/
    │   ├── flex_attention/
    │   ├── ragged_gather/
    │   ├── ragged_scatter/
    │   ├── ragged_gather_reduce/
    │   ├── causal_conv1d_gated_delta_rule/
    │   └── experimental/
    │       ├── kda/                      # Kimi Delta Attention
    │       ├── mla/                      # Multi-head Latent Attention
    │       ├── gmm_v2/                   # GMM / TGMM v2
    │       ├── fused_moe_rs/             # Experimental fused MoE path
    │       └── tpu/                      # Splash / Ring Attention, TopK
    ├── autotuning/
    │   ├── api.py                        # autotune / AutotuningResult
    │   ├── autotuner.py                  # Candidate compilation, measurement, and results
    │   ├── cache.py                      # Lazy JSON loading
    │   └── arg_spec.py                   # ArgSpec data model
    ├── pallas/
    │   ├── block.py                      # BlockRef / pallas_call wrapper
    │   └── grid.py                       # Program ID grouping and reordering
    ├── benchmarking.py                   # standardize / compile / timer
    ├── hlo_utils.py                      # Operator information extraction from StableHLO
    ├── hlo_utils_common.py               # Kernel information types and deduplication
    ├── batching.py                       # vmap information capture
    ├── precision.py                      # Precision normalization
    ├── quantization.py                   # Qwix QArray / AsQArray adapters
    ├── config.py                         # absl.flags + JAX user context
    ├── shape.py                          # Padding, broadcasting, symbolic shapes
    ├── ad.py                             # VJP construction from residuals
    ├── gpu_utils.py                      # GPU device capability checks
    ├── pydantic.py                       # JAX type and Op serialization
    ├── numerics.py                       # Initialization and numerical comparison
    ├── utils.py                          # Exact division, splitting, and reconstruction
    ├── mosaic_gpu.py                     # GPU layout, quantization, synchronization
    ├── mosaic_tpu.py                     # TPU quantization tiling and pipeline utilities
    └── version.py                        # Version and build identifier
```

---

<details>
<summary>Expand the contents: three parts, fifteen chapters</summary>

- **Part 1: Kernel Layer — GPU / TPU Kernels**
  - [Chapter 1: The Structure of an Operator Directory](#chapter-1-the-structure-of-an-operator-directory)
  - [Chapter 2: BlockSpec, Grid, and Shape Canonicalization](#chapter-2-blockspec-grid-and-shape-canonicalization)
  - [Chapter 3: block.pallas_call — Extending Pallas](#chapter-3-blockpallas_call--extending-pallas)
  - [Chapter 4: Kernels for Other Operators](#chapter-4-kernels-for-other-operators)
- **Part 2: Autotuning System**
  - [Chapter 5: Config — Defining Tunable Parameters](#chapter-5-config--defining-tunable-parameters)
  - [Chapter 6: Cache Keys and the Autotuning Cache](#chapter-6-cache-keys-and-the-autotuning-cache)
  - [Chapter 7: Benchmark-Driven Autotuning](#chapter-7-benchmark-driven-autotuning)
  - [Chapter 8: Heuristics — Default Configurations](#chapter-8-heuristics--default-configurations)
  - [Chapter 9: The default_config Decision Chain](#chapter-9-the-default_config-decision-chain)
- **Part 3: Infrastructure**
  - [Chapter 10: The Op Base Class System](#chapter-10-the-op-base-class-system)
  - [Chapter 11: The Benchmarking System](#chapter-11-the-benchmarking-system)
  - [Chapter 12: Batching / vmap Utilities](#chapter-12-batching--vmap-utilities)
  - [Chapter 13: Precision and Quantization](#chapter-13-precision-and-quantization)
  - [Chapter 14: Configuration and Utility Libraries](#chapter-14-configuration-and-utility-libraries)
  - [Chapter 15: Key Design Patterns](#chapter-15-key-design-patterns)
- [Appendix](#appendix)

</details>

---

## Part 1: Kernel Layer — GPU / TPU Kernels

We start by following `layer_norm` from the common API into the kernel. Normalization keeps its reduction, tiling, and boundary handling in a small enough area to show how a JAX call becomes a Pallas kernel. We then compare it with Attention, Ragged Dot, and the GPU/TPU backends.

---

### Chapter 1: The Structure of an Operator Directory

Take `_src/ops/normalization/` as an example:

| File | Role | Description |
|---|---|---|
| `api.py` | Dispatch layer | Exposes `layer_norm` and selects an implementation in candidate order |
| `base.py` | Reference implementation + operator base class | Parameter validation, pure JAX forward computation, and the VJP contract |
| `pallas_triton.py` | Triton forward backend | `PallasTritonNormalization` and `_normalization_kernel` |
| `pallas_triton_vjp.py` | Triton backward backend | A separate VJP Op and kernel |
| `pallas_triton_config.py` | Forward configuration | Config, cache keys, and heuristics |
| `pallas_triton_vjp_config.py` | Backward configuration | Config and keys specific to the backward pass |
| `arg_specs.py` | Tuning samples | Typical input shapes, dtypes, and switches |
| `*_test.py` / `test_base.py` | Tests | Numerical, transformation, and backend behavior checks |

**The pattern**: This directory separates the kernel, operator contract, and tuning strategy. Normalization has separate configuration files for its forward and backward passes. Other operators sometimes keep Config in the backend file or move it into `*_common.py`. [Normalization directory][src-normalization-tree].

#### The Four-Layer Call Chain

From the user call to device computation, the path crosses four layers:

```text
tokamax.layer_norm(x, scale, offset)
  │
  ▼
① api.py
  IMPLEMENTATIONS → try PallasTritonNormalization / Normalization in order
  │
  ▼
② op.py
  bind → device/shape checks → batch/VJP wrappers → internal fwd selects Config
  │
  ▼
③ pallas_triton.py::_fwd
  canonicalize_shape_3d → Fusion → BlockSpec → grid → block.pallas_call
  │
  ▼
④ _normalization_kernel
  BlockRef.load → mean / var / rstddev → scale / offset → BlockRef.store
```

Under `jax.jit`, Python decisions such as backend and Config selection occur mainly during tracing/lowering. Each execution of the compiled program uses the kernel configuration chosen at that point. [Op call implementation][src-op-call].

---

**① Dispatch layer, `api.py` — backend registration and ordered fallback**

```python
# _src/ops/normalization/api.py
# sym:IMPLEMENTATIONS

_IMPLEMENTATIONS = dict(xla=base.Normalization())
_DEFAULT_IMPLEMENTATIONS = ('xla',)

try:
  from tokamax._src.ops.normalization import pallas_triton  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

  _IMPLEMENTATIONS['triton'] = pallas_triton.PallasTritonNormalization()
  _DEFAULT_IMPLEMENTATIONS = ('triton',) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass


IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS
```

After registration, the public entry point normalizes `implementation` into a sequence of candidates. The following excerpt keeps the key branches, omitting parameter documentation and the detailed call arguments:

```python
# _src/ops/normalization/api.py
# sym:layer_norm (structural excerpt)

def layer_norm(x, scale, offset, *, implementation=None, **kwargs):
    if implementation is None:
        implementation = _DEFAULT_IMPLEMENTATIONS
    elif isinstance(implementation, str):
        implementation = (implementation,)
    elif not implementation:
        raise ValueError("`implementation` must not be an empty sequence.")

    errors = []
    for impl in implementation:
        if impl == "triton" and not gpu_utils.has_triton_support():
            errors.append(NotImplementedError("Triton not supported"))
            continue
        if impl not in IMPLEMENTATIONS:
            raise ValueError(f"Unknown implementation: {impl}")
        try:
            return IMPLEMENTATIONS[impl](
                x=x, scale=scale, offset=offset, **kwargs
            )
        except NotImplementedError as error:
            errors.append(error)
    raise ExceptionGroup("all implementations failed", errors)
```

A few details matter:

- If Triton can be imported, the default candidates are `("triton", "xla")`. Explicitly specifying `"triton"` tries only that implementation.
- The first successful candidate wins. There is no step that benchmarks every backend and then picks the fastest.
- Only `NotImplementedError` is caught to continue the fallback sequence. Unknown implementation names and input validation errors follow their own exception paths.
- The full public signature includes `axis`, `epsilon`, `scale_offset`, and `subtract_mean`, but no `config=` parameter. [Complete API][src-normalization-api].

---

**② Op base class, `op.py` — binding arguments, selecting Config, recording metadata, and connecting VJP**

This is a structural excerpt. Chapter 10 covers the omitted array splitting, batch capture, and VJP branches.

```python
# _src/ops/op.py
# sym:Op.__call__ (structural excerpt)

def __call__(self, *args, return_residuals=False, **kwargs):
    ba = self.bind(*args, **kwargs)
    ...

    @self._capture_batched_args
    def fwd(*arrays, batched_args, fwd_res=True):
        ...
        config = ba.default_config
        json_op = self.replace(config=config, vjp=None)
        json_ba = BoundArguments(json_op, _abstractify(dict(ba.arguments)))
        json_data = str(BOUND_ARGS_ADAPTER.dump_json(json_ba), "utf-8")

        with _tokamax_metadata(json_data):
            out, residuals = self._fwd(*args, config=config, **kwargs)
        ...

    ...
    f = jax.custom_vjp(f)
    f.defvjp(fwd, bwd, optimize_remat=True)
    return f(*arrays)
```

`BoundArguments` pairs the operator with the arguments for this call, while `default_config` determines its execution configuration. Metadata makes it possible to recover the workload after the model has been lowered. There is also a shortcut: with no custom VJP and batch capture disabled, the operator is called directly, without the final `custom_vjp` wrapper. [Complete call branches][src-op-call].

---

**③ Backend implementation, `pallas_triton.py` — canonicalizing shapes, tiling, and launching the kernel**

The following excerpt shows the main tiling and launch logic in `_fwd`:

```python
# _src/ops/normalization/pallas_triton.py
# sym:PallasTritonNormalization._fwd

orig_x_shape = x.shape
# Canonicalize to 3D, where the second axis is the reduced axis.
x_shape = pallas_triton_config.canonicalize_shape_3d(orig_x_shape, axis)
x_shape_ty = jax.ShapeDtypeStruct(x_shape, x.dtype)

if callable(x):
  x = fuser.Fusion(lambda x=x: x().reshape(x_shape), ((), {}), x_shape_ty)
else:
  x = x.reshape(x_shape)  # Reshape outside to allow input-output aliasing.
  x = fuser.Fusion(lambda x=x: x, ((), {}), x_shape_ty)

param_bcast = lambda p: None if p is None else p[:, None]

stat_shape = jax.ShapeDtypeStruct((x.shape[0], 1, x.shape[2]), jnp.float32)
out_shape = (
    x,
    stat_shape if return_residuals and subtract_mean else None,
    stat_shape if return_residuals else None,
)

block_m = config.block_m
block_n = 1 if config.block_n is None else config.block_n
block_a = pl.next_power_of_2(x.shape[1])

x_spec = pl.BlockSpec((block_m, block_a, block_n), lambda i, j: (i, 0, j))
param_spec = pl.BlockSpec((block_a, 1), lambda i, j: (0, 0))
stat_spec = pl.BlockSpec((block_m, 1, block_n), lambda i, j: (i, 0, j))
grid = (pl.cdiv(x.shape[0], block_m), pl.cdiv(x.shape[2], block_n))

x_fn, x_values, x_prefetch = fuser.get_fusion_values(x)
if x_prefetch:
  raise NotImplementedError('Prefetch not supported.')
x_fn_spec_puller = fuser.pull_block_spec(x_fn, x_spec, grid_len=len(grid))
x_fn, (x_value_specs,), _ = x_fn_spec_puller(x_values)

kernel = functools.partial(
    _normalization_kernel,
    x_fn=x_fn,
    epsilon=epsilon,
    scale_offset=scale_offset,
    subtract_mean=subtract_mean,
)

name = 'pallas_layer_norm' if subtract_mean else 'pallas_rms_norm'
if return_residuals:
  name += '_fwd_res'

y, mean, rstddev = block.pallas_call(
    kernel,
    name=name,
    out_shape=out_shape,
    grid=grid,
    in_specs=(x_value_specs, param_spec, param_spec),
    out_specs=(x_spec, stat_spec, stat_spec),
    filter_specs=True,
    input_output_aliases={0: 0} if input_output_alias else {},
    compiler_params=plgpu.CompilerParams(num_warps=config.num_warps),
)(x_values, param_bcast(scale), param_bcast(offset))
```

Here is what happens:

- The input is first reshaped to `(M, K, N)`, where K is the reduction axis.
- `fuser.Fusion` wraps the local input computation. `pull_block_spec` propagates the output block's requirements back to the actual inputs.
- The block indices for `scale` and `offset` do not depend on the program ID. Every program uses the same parameters along the reduction axis.
- `return_residuals` determines whether to output the mean and reciprocal standard deviation. Default aliasing is also constrained by callable inputs.
- `input_output_aliases` describes buffer relationships inside the call. JAX still manages Python-level array semantics and copying. [Forward wrapper and aliasing conditions][src-normalization-triton].

---

**④ Kernel layer — normalizing the current data block**

```python
# _src/ops/normalization/pallas_triton.py
# sym:_normalization_kernel

def _normalization_kernel(
    x_value_refs,
    scale_ref,
    offset_ref,
    y_ref,
    mean_ref,
    rstddev_ref,
    *,
    x_fn,
    epsilon,
    scale_offset,
    subtract_mean,
):
  """Normalization kernel."""
  pids = (pl.program_id(0), pl.program_id(1))
  x = x_fn(pids, (), jax.tree.map(lambda x: x.load(), x_value_refs))
  x = x.astype(jnp.promote_types(x.dtype, jnp.float32))

  axis_len = y_ref.full_shape[1]
  if subtract_mean:
    mean = jnp.sum(x, axis=1, keepdims=True) / axis_len
    if mean_ref is not None:
      mean_ref.store(mean)
    x -= mean
    # # Zero invalid values (when axis is not a power of two).
    if x.shape[1] != axis_len:
      x *= (jnp.indices(x.shape, sparse=True)[1] < axis_len).astype(x.dtype)

  var = jnp.sum(jnp.square(x), axis=1, keepdims=True) / axis_len
  rstddev = jax.lax.rsqrt(var + epsilon)
  if rstddev_ref is not None:
    rstddev_ref.store(rstddev)
  x *= rstddev

  if scale_ref is not None:
    x *= scale_ref.load(bounds_check=False).astype(x.dtype) + scale_offset
  if offset_ref is not None:
    x += offset_ref.load().astype(x.dtype)

  y_ref.store(x.astype(y_ref.dtype))
```

`axis_len` comes from the original full shape. Even when the reduction axis is padded, the mean and variance are divided by the actual K, not the padded length.

LayerNorm needs an additional padding correction: invalid elements load as 0, but subtracting the mean turns them into `-mean`. They must be zeroed again before summing the squares. Setting `subtract_mean=False` takes the RMSNorm path. [Kernel computation][src-normalization-kernel].

---

### Chapter 2: BlockSpec, Grid, and Shape Canonicalization

#### 2.1 canonicalize_shape_3d: A Common Shape Convention

```python
# _src/ops/normalization/pallas_triton_config.py
# sym:canonicalize_shape_3d / canonicalize_shape

def canonicalize_shape_3d(
    shape: Sequence[int], axis: int
) -> tuple[int, int, int]:
  return (math.prod(shape[:axis]), shape[axis], math.prod(shape[axis:][1:]))


def canonicalize_shape(
    shape: Sequence[int], axis: int
) -> tuple[int, int] | tuple[int, int, int]:
  if len(shape[axis:]) > 1:
    return canonicalize_shape_3d(shape, axis)
  return (math.prod(shape[:axis]), shape[axis])
```

| Symbol | Meaning | `(2, 3, 17), axis=-1` |
|---|---|---|
| M | Product of dimensions before the reduction axis | 6 |
| K | Length of the reduction axis | 17 |
| N | Product of dimensions after the reduction axis | 1 |

`_fwd` uses the three-dimensional shape `(6, 17, 1)`. Config heuristics and cache keys can use `(6, 17)`, dropping the trailing 1. The two representations serve different layers. [Shape canonicalization][src-normalization-config].

#### 2.2 BlockSpec: Declaring the Tiling

```python
# _src/ops/normalization/pallas_triton.py
# sym:PallasTritonNormalization._fwd

x_spec = pl.BlockSpec(
    (block_m, block_a, block_n),
    lambda i, j: (i, 0, j),
)
```

| Element | Meaning | Value here |
|---|---|---|
| `block_shape` | Shape of the current block | `(block_m, block_a, block_n)` |
| `index_map` | Program coordinates → block indices | `lambda i, j: (i, 0, j)` |

Ordinary blocked indexing multiplies each block index by the corresponding block size. The starting element is therefore `(i * block_m, 0, j * block_n)`. The reduction axis is not split across multiple programs. [BlockSpec conventions][jax-blockspec].

#### 2.3 Grid: Defining the Parallel Work

```python
# _src/ops/normalization/pallas_triton.py
# sym:PallasTritonNormalization._fwd

grid = (
    pl.cdiv(x.shape[0], block_m),
    pl.cdiv(x.shape[2], block_n),
)
```

For `(M, K, N)=(6,17,1)` and `block_m=4`:

```text
block_shape = (4, 32, 1)
grid        = (2, 1)

program (0, 0): M positions 0..3, K positions 0..31
program (1, 0): M positions 4..7, K positions 0..31
                M positions 6..7 and K positions 17..31 are masked
```

The product of the grid dimensions is only the number of logical invocations. On a GPU, simultaneous residency also depends on SM, register, and shared-memory limits. On a TPU, sequential grid execution, pipelines, and `dimension_semantics` must be considered to determine which axes can actually execute in parallel. [TPU execution semantics][jax-tpu].

#### 2.4 How Config Controls Tiling

```python
# _src/ops/normalization/pallas_triton.py
# sym:PallasTritonNormalization._fwd

y, mean, rstddev = block.pallas_call(
    kernel,
    name=name,
    out_shape=out_shape,
    grid=grid,
    in_specs=(x_value_specs, param_spec, param_spec),
    out_specs=(x_spec, stat_spec, stat_spec),
    filter_specs=True,
    input_output_aliases={0: 0} if input_output_alias else {},
    compiler_params=plgpu.CompilerParams(num_warps=config.num_warps),
)(x_values, param_bcast(scale), param_bcast(offset))

y = y.reshape(orig_x_shape)
stat_shape = list(orig_x_shape)
stat_shape[axis] = 1
if mean is not None:
  mean = mean.reshape(stat_shape)
if rstddev is not None:
  rstddev = rstddev.reshape(stat_shape)

return y, (mean, rstddev) if return_residuals else None
```

| Parameter | Source | Role |
|---|---|---|
| `block_m` | Config | Block size along M |
| `block_n` | Config; `None` becomes 1 in the execution layer | Block size along N |
| `block_a` | `next_power_of_2(K)` | Covers the reduction axis |
| `num_warps` | Config → CompilerParams | Triton warp parameter |

Alignment constraints vary by backend. Triton imposes operation-shape constraints, Mosaic GPU has memory-access alignment requirements, and TPU blocks follow their own rules. Common TPU constraints involve 8/128 alignment of the last two dimensions, with exceptions such as covering a full dimension. When writing a kernel, inspect the restrictions in its specific Config and implementation. [Backend tiling rules][jax-blockspec].

> The key point so far: the backend interprets Config as tiling and compiler parameters, and the kernel computes in that execution shape. Part 2 explains where Config comes from.

---

### Chapter 3: block.pallas_call — Extending Pallas

Normalization's Triton path wraps the native Pallas call with `_src/pallas/block.py`, adding full-shape and boundary information to each Ref.

#### 3.1 The BlockRef Concept

```python
# _src/pallas/block.py
# sym:BlockRef (field excerpt)

@dataclasses.dataclass(frozen=True, slots=True)
class BlockRef:
    ref: Any
    full_shape: tuple[int, ...]
    spec: pl.BlockSpec
```

Inside the kernel, `load()` and `store()` use this information to determine the valid region of the current block. `full_shape` lets reductions use the actual axis length. `at[...]` goes through `BlockRefIndexer`, preserving these associations. [BlockRef][src-block].

#### 3.2 Automatic Bounds Checking

```python
# _src/pallas/block.py
# sym:BlockRef.bounds_checked

@property
def bounds_checked(self) -> tuple[bool, ...]:
  """Indicates which dimensions require bounds checking."""
  assert self.spec.block_shape is not None
  if any(isinstance(bs, pl.Element) for bs in self.spec.block_shape):
    return (True,) * self.ndim
  bounds = self.bounds
  return tuple((bound % dim != 0) for bound, dim in _zip(bounds, self.shape))
```

Ordinary blocked indexing checks whether the dimensions divide evenly. Using `pl.Element` takes the corresponding checking path. `inbounds_masks` then combines block starts with indices inside the block to construct a mask.

When a mask is present and the caller has not provided `other`, `BlockRef.load` fills invalid positions with 0. Stores suppress out-of-bounds writes. This is a property of the Tokamax wrapper; native Pallas padding does not offer the same guarantee about values. [Boundary handling][src-block], [Pallas padding][jax-blockspec].

#### 3.3 Cache Optimization

```python
# _src/pallas/block.py
# sym:BlockRef.load

uses_element_indexing = any(
    isinstance(bd, pl.Element) for bd in self.spec.block_shape
)
if "eviction_policy" not in kwargs and uses_element_indexing:
  # If the index mapping function is a function of every program ID, then
  # it means that block is only read by one program ID. It is not likely to
  # be beneficial to keep the values in the cache. Here, we simply check if
  # the index mapping contains every program ID.
  # TODO: Can we detect more complicated functions of the program IDs?
  pids = _pids()
  start_idxs = tuple(map(id, self.spec.index_map(*pids)))
  if all(pid in start_idxs for pid in map(id, pids)):
    kwargs["eviction_policy"] = "evict_first"
```

This hint is attempted only for `pl.Element` indexing when the caller has not supplied an eviction policy. The implementation checks whether `index_map` directly returns every program ID object, using this as a rough heuristic for data reuse. Actual cache residency still depends on the program and device. [Load cache hints][src-block].

#### 3.4 The mock.patch Implementation

```python
# _src/pallas/block.py
# sym:pallas_call.wrapped_kernel (excerpt)

with (
    _PL_LOAD_STORE_PATCH_LOCK,
    mock.patch.object(plgpu, "load", wrapped_load),
    mock.patch.object(plgpu, "store", wrapped_store),
):
    return kernel(*in_refs, *out_refs)
```

While tracing the kernel, the wrapper replaces Triton load/store with versions that accept BlockRef and then unwrap the underlying Ref. The lock protects modifications to shared Python attributes; it does not serialize programs on the device. Other backends may use native `pl.pallas_call` directly or use different wrappers. [Call wrapper][src-block].

#### 3.5 Grid Scheduling Optimization

```python
# _src/pallas/grid.py
# sym:_get_group_size_cost

def _get_group_size_cost(
    group_size_m: int,
    grid_m: int,
    grid_n: int,
    block_m_cost: int,
    block_n_cost: int,
) -> int:
  """Returns the total cost for the given group size."""
  num_live_groups = pl.cdiv(jax.devices()[0].core_count, group_size_m)
  num_live_blocks_m = pl.cdiv(num_live_groups, grid_n) * group_size_m
  num_live_blocks_m = min(num_live_blocks_m, grid_m)
  num_live_blocks_n = min(num_live_groups, grid_n)
  return num_live_blocks_m * block_m_cost + num_live_blocks_n * block_n_cost
```

`get_cheapest_grid_pids` tries group sizes along both M and N, choosing a program ID reordering from estimated active block counts and data costs. It changes access order while preserving the logical computation domain. The cost model's choice still needs validation on the actual workload. [Grid utilities][src-grid].

---

### Chapter 4: Kernels for Other Operators

#### 4.1 Flash Attention (Triton)

Within a Q block, Attention iterates over the required KV range while maintaining online softmax statistics and an output accumulator:

```text
Q block → KV block → QK logits → update softmax statistics → accumulate PV
                         ↑                                      │
                         └──── iterate over remaining KV ───────┘
```

The Triton Config at this revision is:

```python
# _src/ops/attention/pallas_triton.py
# sym:Config

@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_q: pydantic.PositiveInt
  block_k: pydantic.PositiveInt
  num_stages: pydantic.PositiveInt
  num_warps: pydantic_lib.PowerOfTwo
  block_d: pydantic.PositiveInt | None = None
  block_d_out: pydantic.PositiveInt | None = None
  split_k: pydantic.PositiveInt = 1
  pack_mask: bool = False
```

`block_q/block_k`, head-dimension tiling, `split_k`, and mask handling affect different execution paths. `use_base2` is an Op field. Q/K/V block mappings and forward/backward statistics are more involved than in Normalization. [Triton Attention][src-attention-triton].

The public input shapes are `Q: (*B,T,N,H)`, `K: (*B,S,K,H)`, and `V: (*B,S,K,h)`. Q and KV can have different head counts, and the output head dimension can differ as well. Each backend further checks which combinations it supports. [Attention API][src-attention-api].

#### 4.2 Mosaic GPU (SM80/SM90/SM100)

| Operator | Architecture dispatch at this revision | Main files |
|---|---|---|
| Attention | SM90, SM100 | `pallas_mosaic_gpu.py` + two architecture-specific kernels |
| GLU | SM80, SM90, SM100 | `pallas_mosaic_gpu.py` + three architecture-specific kernels |
| Ragged Dot | Depends on architecture, quantization, and layout | `pallas_mosaic_gpu.py` + multiple kernel variants |

SM80 is still an experimental path. The common GPU capability check alone does not establish support for every operator. [GPU utilities][src-gpu-utils], [Attention dispatch][src-attention-gpu], [GLU dispatch][src-glu-gpu].

For example, the SM100 Attention source explicitly arranges:

```text
copy_gmem_to_smem → multistage K/V buffers → produced barrier
                                                 │
                                     QK/PV: tcgen05_mma
                                                 │
                                     consumed barrier → reuse buffers
```

Tunable choices on this path also include pipeline slots, shared-memory usage, and how work is divided between computation stages. [SM100 kernel][src-attention-sm100].

#### 4.3 Mosaic TPU — The TPU Backend

TPU shares the Op and configuration interfaces, but its backend implements kernel layouts, memory spaces, and pipelines. Attention's forward pass adjusts Q/K/V layouts and calls Splash. Its Config includes:

```python
# _src/ops/attention/pallas_mosaic_tpu.py
# sym:Config

@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_q: Annotated[int, pydantic.Field(multiple_of=common.NUM_LANES, gt=0)]
  block_kv: Annotated[int, pydantic.Field(multiple_of=common.NUM_LANES, gt=0)]
  block_kv_compute: Annotated[
      int, pydantic.Field(multiple_of=common.NUM_LANES, gt=0)
  ]
  q_layout: splash.QKVLayout
  k_layout: splash.QKVLayout
  v_layout: splash.QKVLayout
  use_experimental_scheduler: bool
  use_base2_exp: bool = True

  def __post_init__(self):
    if self.block_kv % self.block_kv_compute:
      raise ValueError(
          f"{self.block_kv=} must be a multiple of {self.block_kv_compute=}."
      )
```

`_get_autotuning_configs` enumerates tiles, layouts, and schedulers, then filters unsuitable combinations. TPU tuning is therefore not limited to manual table lookups. [TPU Attention][src-attention-tpu].

`custom_buffered_pallas_call` connects a scalar-prefetch grid to `pltpu.emit_pipeline`. Indices and dynamic grid information pass through SMEM, while buffer configurations arrange data movement. Generation-specific parameters in the utility guide layout and tiling choices. [TPU utilities][src-mosaic-tpu].

#### 4.4 The Ragged Dot Kernel Variants

| File/path | Purpose |
|---|---|
| `pallas_triton.py` | Triton grouped matrix multiplication |
| `pallas_mosaic_gpu_kernel_sm80.py` | SM80 path |
| `pallas_mosaic_gpu_kernel_sm90.py` / `*_sm90_quant.py` | Regular and quantized SM90 paths |
| `pallas_mosaic_gpu_kernel_sm100.py` / `*_sm100_quant.py` | Regular and quantized SM100 paths |
| `*_sm100_fp8_quant.py` / `*_sm100_i8_quant.py` | FP8 / INT8 variants |
| `*_sm100_quant_post_scale.py` | Quantized variant with post-scaling |
| `pallas_mosaic_tpu.py` / `pallas_mosaic_tpu_kernel.py` | Original TPU GMM/TGMM paths |
| `pallas_mosaic_tpu_v2.py` + `experimental/gmm_v2/` | TPU v2 path |

GPU variants are selected by the actual dispatch code. TPU v2 allows `tile_m/tile_k/tile_n` to be left unspecified so that the lower-level implementation can calculate tiles under constraints such as VMEM capacity. `implementation="mosaic"` still maps to `mosaic_gpu` or `mosaic_tpu`; v2 has its own entry point. [GPU dispatch][src-ragged-gpu], [TPU v2][src-ragged-tpu-v2], [public dispatch][src-ragged-api].

#### 4.5 The Fused GLU Kernel

GLU has the following mathematical structure:

```text
gate = x @ W_gate
up   = x @ W_up
out  = activation(gate) * up
```

Mosaic GPU brings the activation and multiplication after the matrix products into the kernel, avoiding intermediate tensor materialization. However, requesting `return_residuals=True` redirects `_fwd` to the base-class path, so training and forward-only execution may take different code paths.

One common trap: HBM traffic must be worked out from the inputs, weights, outputs, residuals, and compiled program. Counting Python expressions does not tell you how many round trips to device memory occur. [GLU implementation][src-glu-gpu].

#### 4.6 A Common Kernel Pattern

1. **Op contract**: Validates inputs and connects configuration, VJP, and context.
2. **Config**: Describes the tunable execution shape of this implementation.
3. **Tiling and layout**: Map logical inputs to each kernel invocation.
4. **Kernel**: Computes block-local results or participates in a longer pipeline or communication sequence.
5. **Tuning and timing**: Measure candidates and reuse the results in later compilations.

| Aspect | GPU (Triton) | GPU (Mosaic) | TPU (Mosaic) |
|---|---|---|---|
| Tiling constraints | Operation shapes and kernel-specific limits | Memory alignment, layout, and architecture limits | TPU block rules and operator constraints |
| Common configuration | Blocks, warps, stages | Tiles, stages, shared memory, and division of computation | Tiles, layouts, buffers, schedulers |
| Tuning sources | Candidate measurements + heuristics | Candidate measurements + heuristics | Candidate measurements, heuristics, some LUTs, or lower-level tiling decisions |
| Grid meaning | Logical program space | Depends on the execution model | Combined with sequential execution, pipelines, and multicore semantics |

This revision also includes directories for Ragged Gather/Scatter/Reduce, KDA, MLA, TPU TopK, fused MoE, and fused causal-conv/Gated Delta Rule. Internal or experimental status should be checked separately from top-level exports. [Additional operator directories][src-ops-tree].

---

## Part 2: Autotuning System

Part 1 left one question open: who chooses Config for the kernel? The entry point is `BoundArguments.default_config`. We first look at the selection logic, then work through Config, Cache, Benchmark, Heuristics, and the final decision chain.

### Autotuning Priority Overview

```text
Explicit op.config?
  ├─ Yes → use it directly
  └─ No → evaluate heuristics
           ├─ NullConfig → no tunable parameters; return
           └─ Tunable Op → query autotuning cache
                            ├─ Nonempty hit → fastest_config
                            └─ Miss → select one fallback branch
                                       ├─ "autotune"   → measure candidates
                                       ├─ "heuristics" → use heuristic Config
                                       └─ "error"      → raise an error
```

After a cache miss, `autotuning_cache_miss_fallback` determines which branch to take. A failed tuning run does not automatically fall back to heuristics here. `get_config` evaluates heuristics before looking in the cache to recognize the special `NullConfig` case, which has no tunable parameters. For a tunable Op, cached data takes precedence over the heuristic result. Chapter 9 covers the complete order. [Decision logic][src-op-config].

---

### Chapter 5: Config — Defining Tunable Parameters

#### 5.1 The pydantic Dataclass Pattern

Normalization defines its configuration as follows:

```python
# _src/ops/normalization/pallas_triton_config.py
# sym:Config

@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_m: pydantic_lib.PowerOfTwo
  block_n: pydantic_lib.PowerOfTwo | None
  num_warps: pydantic_lib.PowerOfTwo
```

| Modifier/type | Effect | Significance for tuning |
|---|---|---|
| `frozen=True` | Makes instances immutable | Allows them to serve as elements of candidate sets and keys in result mappings |
| `kw_only=True` | Requires construction with keyword arguments | Makes the field being set explicit |
| `slots=True` | Fixes attribute storage | Avoids a per-instance dynamic attribute dictionary |
| `PowerOfTwo` | Validates powers of two | Enforces this implementation's parameter constraints |

`block_m` controls the M direction, and `block_n` controls N. For a reduction over the last axis, `block_n=None`, which becomes 1 in the execution layer. K is covered by `block_a=next_power_of_2(K)`. [Config definition][src-normalization-config].

#### 5.2 How to Use Config

**① Public API: let the framework choose the configuration**

Ordinary calls do not pass Config. For the CPU examples, first pin the source revision used in this article and the JAX version used for validation:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install 'jax[cpu]==0.11.1' \
  'git+https://github.com/openxla/tokamax.git@f21f67cc921f6ad56785a084ba30dc3d85278e9a'
```

This complete CPU example uses a reduction length of 17, which is not a power of two, and checks the forward pass, gradients, and vmap:

```python
import jax
import jax.numpy as jnp
import numpy as np
import tokamax

x = jax.random.normal(jax.random.key(0), (2, 3, 17), dtype=jnp.float32)
scale = jnp.linspace(0.5, 1.5, 17)
offset = jnp.linspace(-0.1, 0.1, 17)

def normalize(a):
    return tokamax.layer_norm(a, scale, offset, implementation="xla")

def reference(a):
    centered = a - jnp.mean(a, axis=-1, keepdims=True)
    variance = jnp.mean(centered ** 2, axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(variance + 1e-6) * scale + offset

y = jax.jit(normalize)(x)
np.testing.assert_allclose(y, reference(x), rtol=2e-5, atol=2e-5)
actual_grad = jax.grad(lambda a: jnp.sum(normalize(a) ** 2))(x)
expected_grad = jax.grad(lambda a: jnp.sum(reference(a) ** 2))(x)
np.testing.assert_allclose(actual_grad, expected_grad, rtol=3e-4, atol=3e-4)
np.testing.assert_allclose(jax.vmap(normalize)(x), y, rtol=2e-5, atol=2e-5)
print("forward / grad / vmap: PASS", y.shape)
```

Selecting `xla` runs the reference path. On a supported NVIDIA GPU, you can switch to `implementation="triton"` and repeat the numerical comparison in that environment. [Public call][src-normalization-api].

**② Source-level debugging: construct the backend Op manually**

The public `layer_norm` function has no `config=` parameter. To configure it manually, construct the backend Op directly. The following complete example checks configuration construction and argument binding on a CPU without executing the GPU kernel:

```python
import jax
import jax.numpy as jnp
from tokamax._src.ops.normalization import pallas_triton
from tokamax._src.ops.normalization import pallas_triton_config

config = pallas_triton_config.Config(
    block_m=4, block_n=None, num_warps=4
)
op = pallas_triton.PallasTritonNormalization(config=config)
abstract_x = jax.ShapeDtypeStruct((8, 1024), jnp.float32)
bound = op.bind(abstract_x, scale=None, offset=None)
assert bound.default_config == config
print(bound.default_config)
```

| Field | Effect on execution |
|---|---|
| `block_m=4` | One program covers 4 positions along M |
| `block_n=None` | This example reduces the last axis, so the execution-layer block size along N is 1 |
| `num_warps=4` | Passed to the Triton compiler parameters |

An explicit configuration takes precedence over the cache. Valid field values, successful compilation on a device, and reasonable performance are three separate questions. [Configuration selection][src-op-config].

**③ Performance experiments: autotune, then freeze Config**

This complete GPU example requires a supported NVIDIA GPU and a matching JAX GPU backend. It measures only the two supplied candidates, then checks the output of the selected configuration. The example has not been run on a GPU for this article.

```python
import jax
import jax.numpy as jnp
import numpy as np
import tokamax
from tokamax._src.ops.normalization import pallas_triton
from tokamax._src.ops.normalization import pallas_triton_config

if jax.default_backend() != "gpu":
    raise RuntimeError("This example requires a supported NVIDIA GPU and a JAX GPU backend")

x = jax.random.normal(jax.random.key(0), (128, 1024), dtype=jnp.float32)
scale = jnp.ones((1024,), dtype=jnp.float32)
offset = jnp.zeros_like(scale)
op = pallas_triton.PallasTritonNormalization()
bound = op.bind(x, scale, offset)
candidates = {
    pallas_triton_config.Config(block_m=1, block_n=None, num_warps=4),
    pallas_triton_config.Config(block_m=4, block_n=None, num_warps=4),
}

data = bound.autotune(configs=candidates, cache_results=False)
best_config = data.fastest_config
fast_op = op.replace(config=best_config)
y = jax.jit(fast_op)(x, scale, offset)
y_ref = tokamax.layer_norm(x, scale, offset, implementation="xla")
np.testing.assert_allclose(y, y_ref, rtol=1e-4, atol=1e-4)
print("best among supplied candidates:", best_config)
```

`fast_op` fixes only the forward configuration for this workload; the VJP Op has its own configuration. To broaden the search, use the backend's default candidate set. Reliable tuning results also require numerical checks and a stable measurement environment. [Single-Op tuning][src-op-autotune].

#### 5.3 Comparing Config Complexity Across Operators

| Operator/implementation | Configuration | Characteristics |
|---|---|---|
| Normalization Triton | `block_m, block_n, num_warps` | Processes the reduction axis as a whole |
| Attention Triton | `block_q, block_k, num_stages, num_warps, block_d, split_k`, etc. | Adds loops, head-dimension tiling, and KV partitioning |
| Attention Mosaic GPU | Separate SM90 / SM100 Config types | Architecture determines the candidate type |
| Attention Mosaic TPU | Q/KV tiles, QKV layouts, scheduler | Adapts the Splash path |
| Ragged Dot TPU v2 | `tile_m, tile_k, tile_n, bucket_base` | Can delegate tiling to the lower-level implementation by default |
| XLA Op with no tunable parameters | `NullConfig` | Uses the reference computation directly |

Fields with the same name are not necessarily interchangeable between implementations. Config's semantics are defined by the `_fwd` and kernel that consume it. [Attention Config][src-attention-triton], [GPU Config dispatch][src-attention-gpu], [TPU Config][src-attention-tpu], [Ragged Dot v2][src-ragged-tpu-v2].

#### 5.4 Serializing Config

pydantic handles field validation and JSON encoding. The Config in this chapter can be serialized through TypeAdapter. This snippet continues from the `config` object above:

```python
import pydantic

adapter = pydantic.TypeAdapter(type(config))
encoded = adapter.dump_json(config)
decoded = adapter.validate_json(encoded)
assert decoded == config
```

A configuration object, the package's bundled cache, and an `AutotuningResult` file are three different formats. To save the device, workloads, and timing results from an entire tuning run, use the result serialization interface in Chapter 7. [Type adapters][src-pydantic].

---

### Chapter 6: Cache Keys and the Autotuning Cache

#### 6.1 Cache Key Design

Normalization condenses the abstract input specification into a hashable mapping:

```python
# _src/ops/normalization/pallas_triton_config.py
# sym:get_key

def get_key(
    x: jax.Array,
    scale: jax.Array | None,
    offset: jax.Array | None,
    *,
    axis: int,
    return_residuals: bool = False,
    **kwargs,
) -> Key:
  """Returns the lookup key for the given args."""
  # TODO: Cap the shape at a given size?
  key = immutabledict.immutabledict(
      x=jax.ShapeDtypeStruct(canonicalize_shape(x.shape, axis), x.dtype),
      scale=_maybe_shape(scale),
      offset=_maybe_shape(offset),
      has_epsilon=kwargs.pop('epsilon') != 0.0,
      has_scale_offset=kwargs.pop('scale_offset') != 0.0,
      subtract_mean=kwargs.pop('subtract_mean'),
      return_residuals=return_residuals,
  )
  assert not kwargs, f'Unhandled kwargs: {kwargs}'
  return key
```

A few design choices stand out:

1. **Use shape/dtype**: Concrete array values are not part of the key.
2. **Group switches into classes**: Epsilon and scale offset distinguish only zero from nonzero. These are the tuning equivalence classes chosen by this implementation.
3. **Include the forward mode**: Subtracting the mean or returning residuals changes the workload.
4. **Catch omitted parameters**: The final `assert not kwargs` catches new parameters that have not been included in the key.

The cache is also separated by Op and device kind. Each backend decides how batch information enters the key. Normalization's override still contains a TODO about using batched arguments. [Key function][src-normalization-config], [backend override][src-normalization-triton].

#### 6.2 AutotuningCache: Lazy JSON Loading

```python
# _src/autotuning/cache.py
# sym:AutotuningCache.__missing__

def __missing__(self, device_kind: DeviceKind) -> DeviceAutotuningCache:
  self[device_kind] = (cache := self._load_cache(device_kind))
  return cache
```

The first access to a device kind loads its JSON files. The device string is lowercased, with spaces replaced by underscores. The filename is the **concrete Op class name converted to snake_case**:

```text
data/autotuning/{device_kind}/{op_class_name}.json

data/autotuning/nvidia_h100_80gb_hbm3/pallas_triton_normalization.json
data/autotuning/nvidia_b200/pallas_mosaic_gpu_ragged_dot.json
```

`PallasTritonNormalization` does not read `normalization.json` just because it belongs to the normalization family. After loading, the code calls `op.bind(**k)` to regenerate the cache key using the current implementation. When multiple paths contain the same key, later data overwrites earlier data. [Lazy loading implementation][src-cache].

#### 6.3 The Global Cache and Overlays

At this revision, overlays use ContextVar while also maintaining a JAX user context:

```python
# _src/ops/op.py
# sym:AUTOTUNING_CACHE_OVERLAY_STACK

_AUTOTUNING_CACHE: dict[Op, AutotuningCache] = {}

AUTOTUNING_CACHE_OVERLAY_STACK = contextvars.ContextVar(
    "AUTOTUNING_CACHE_OVERLAY_STACK", default=()
)
AUTOTUNING_CACHE_OVERLAY_JAX_CONFIG = jax.make_user_context(())
```

Lookup starts at the innermost overlay, then moves to the process cache and its lazily loaded bundled JSON data. Entering an `AutotuningResult` scope adds the result object's identity to the JAX context, so scope changes participate in distinguishing JIT cache entries. [Overlay lookup][src-op-cache-lookup], [scope management][src-autotuning-result].

| Operation | Actual effect |
|---|---|
| `bound.autotune(cache_results=True)` | Writes to the process cache |
| `tokamax.autotune(...)` | Uses `cache_results=False` internally and returns a result object |
| `with result:` | Temporarily enables the result as an overlay |
| `result.dump(fp)` / `result.dumps()` | Explicit persistence/serialization |

The internal `_src.config.ignore_autotuning_cache` option skips cache lookup, including overlays. It is not re-exported by the public `tokamax.config` module. [Public configuration entry point][src-public-config].

---

### Chapter 7: Benchmark-Driven Autotuning

When a cache miss occurs and the `autotune` strategy is selected, the system measures the backend's candidate configurations. You can also supply candidates explicitly and start an experiment, as in Chapter 5.

#### 7.1 AutotuningData

`AutotuningData` stores `Config → BenchmarkData | Exception`. The best configuration is selected by the median execution time among successful results:

```python
# _src/autotuning/autotuner.py
# sym:AutotuningData.fastest_config

@property
def fastest_config(self) -> K:
  valid_benchmarks = tuple(
      it for it in self.items() if isinstance(it[1], BenchmarkData)
  )
  try:
    key_fn = lambda x: x[1].median_evaluation_time_ms
    return min(valid_benchmarks, key=key_fn)[0]
  except ValueError as e:
    if self:
      exceptions = cast(tuple[Exception, ...], tuple(self.values()))
      raise ExceptionGroup("All configs failed", exceptions) from e
    raise ValueError("Autotuning data is empty") from e
```

| Method | Purpose |
|---|---|
| `fastest_config` | The successful candidate with the lowest median time |
| `prune()` | Keeps only the fastest candidate |
| `prune_errors()` | Removes exceptions while retaining all successful candidates |

Empty data and an all-failed candidate set raise different exceptions. Also, `dumps(prune_errors=True)` only removes errors; it does not keep just the fastest configuration. [Result model][src-autotuner].

#### 7.2 Autotuner: Parallel Compilation and Synchronous Measurement

The default executor configuration is:

```python
# _src/autotuning/autotuner.py
# sym:Autotuner (field excerpt)

@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Autotuner:
    compile_executor_fn: Callable[[], futures.Executor] | None = (
        futures.ThreadPoolExecutor
    )
    executor_fn: Callable[[], futures.Executor] = _SyncExecutor
```

`_SyncExecutor` executes the function directly when it is submitted:

```python
# _src/autotuning/autotuner.py
# sym:_SyncExecutor

class _SyncExecutor(futures.Executor):
  """A "no-op" `Executor` that runs submitted functions synchronously."""

  def submit(self, fn, /, *args, **kwargs):
    future = futures.Future()
    try:
      future.set_result(fn(*args, **kwargs))
    except Exception as e:  # pylint: disable=broad-exception-caught
      future.set_exception(e)
    return future
```

The two-stage flow is:

```text
Config set
  ├─ ThreadPoolExecutor: build candidate functions → lower → compile
  │                       └─ record compilation failures
  └─ Default _SyncExecutor: run benchmark runners one by one
                            └─ collect timings or exceptions
```

Compilation concurrency and device measurement are controlled separately. The five timing iterations come from the runner's default `iterations=5`; they do not mean that five candidates run concurrently. `timeout` controls waiting on futures. It is not a hard deadline that forcibly terminates running tasks. [Tuning executors][src-autotuner], [timing runner][src-benchmark].

#### 7.3 The Top-Level autotune() Function

The top-level entry point accepts three kinds of input:

1. **Callable**: Lowers it with the supplied arguments, then extracts workloads.
2. **Nonempty list/tuple of BoundArguments**: Processes the supplied specifications directly.
3. **Lowered object**: Extracts workloads directly from existing StableHLO.

| Parameter | Default | Meaning |
|---|---|---|
| `ignore_cache` | `False` | By default, tunes only specifications whose cache lookup returned no data |
| `all_implementations` | `False` | Expands the implementations to tune through an internal registry |
| `progress_bar` | `True` | Displays tuning progress |
| `timeout` | `600.0` | Wait parameter passed to the tuning executor |
| `max_workers` | `None` | Specifies or derives the compilation thread count |

The default candidates are the union of the heuristic configuration and the set returned by `_get_autotuning_configs`. Each backend defines its search space. A backend that supplies no additional candidates may offer only one configuration. [Single-Op candidates and tuning][src-op-autotune].

The registry used by `all_implementations` covers only some operator families. It expands what gets measured without changing the public API's backend priority. The top-level function also logs and skips some failed tasks, so check the returned results for completeness. [Top-level tuning][src-autotuning-api].

#### 7.4 AutotuningResult

The result object organizes workloads and their tuning data by device kind and records the Tokamax version. It supports scoped overrides, JSON encoding/decoding, and merging results for the same device. For the same key, configuration data from the right-hand operand takes precedence. [Result object][src-autotuning-result].

**Complete CPU example: extract → tune → save → load → enable the result**

This example uses XLA Normalization's single `NullConfig` candidate to validate the tooling workflow. A performance search over Triton configurations requires a GPU.

```python
import jax
import jax.numpy as jnp
import numpy as np
import tokamax

x = jax.random.normal(jax.random.key(7), (4, 17), dtype=jnp.float32)

def model(a):
    return tokamax.layer_norm(a, scale=None, offset=None, implementation="xla")

lowered = jax.jit(model).lower(x)
bound_args = tokamax.autotuning.get_bound_args(lowered)
if not bound_args:
    raise RuntimeError("No Tokamax operator specifications were extracted")

result = tokamax.autotune(
    lowered, ignore_cache=True, progress_bar=False, max_workers=2
)
if len(result.data) != len(bound_args):
    raise RuntimeError("At least one operator tuning task returned no result")
for bound, data in result.data:
    print(type(bound.op).__name__, data.fastest_config)

with open("normalization-tuning.json", "w", encoding="utf-8") as fp:
    result.dump(fp, prune_errors=True)
with open("normalization-tuning.json", encoding="utf-8") as fp:
    restored = tokamax.AutotuningResult.load(fp)

compiled_model = jax.jit(model)
with restored:
    y = compiled_model(x)
np.testing.assert_allclose(y, model(x), rtol=2e-5, atol=2e-5)
print("extract / tune / save / reuse: PASS")
```

Keep these details in mind:

- `dump/load` work with file objects; `dumps/loads` work with strings.
- Check both the number of returned results and each entry's `fastest_config`, rather than just whether an object was returned.
- The JSON stores tuning results and does not directly modify an already compiled binary. The result scope affects subsequent tracing/compilation.
- To tune training, extract the Ops from the actual gradient function so that backward workloads are covered too.

#### 7.5 HLO Introspection: get_bound_args

This version writes Op information into XLA metadata. Nested calls place the outer payload first:

```python
# _src/ops/op.py
# sym:_tokamax_metadata

@contextlib.contextmanager
def _tokamax_metadata(json_data: str):
  """Sets the Tokamax payload, stacking onto an enclosing one outermost-first."""
  prev = _ACTIVE_TOKAMAX_PAYLOAD.get()
  payload = f"{prev}/tokamax:{json_data}" if prev else f"tokamax:{json_data}"
  reset = _ACTIVE_TOKAMAX_PAYLOAD.set(payload)
  try:
    with xla_metadata.set_xla_metadata(xla_metadata_payload=payload):
      yield
  finally:
    _ACTIVE_TOKAMAX_PAYLOAD.reset(reset)
```

The JSON passed by `Op.__call__` comes from abstracted bound arguments:

```python
# _src/ops/op.py
# sym:Op.__call__.fwd

json_op = self.replace(config=config, vjp=None)
json_ba = BoundArguments(json_op, _abstractify(dict(ba.arguments)))
json_data = str(BOUND_ARGS_ADAPTER.dump_json(json_ba), "utf-8")
with _tokamax_metadata(json_data):
    out, residuals = self._fwd(*args, config=config, **kwargs)
```

The read path is:

```text
Lowered.compiler_ir("stablehlo")
  → traverse the MLIR module
  → read xla_metadata_payload from mhlo.frontend_attributes
  → get_opspecs restores BoundArguments
  → get_bound_args removes selected Config and deduplicates by class name and key
```

The public interface is `tokamax.autotuning.get_bound_args(...)`. When metadata is absent, the older `op_name` path remains as a fallback. Candidate programs and models using the new configurations still need to be compiled separately. [StableHLO extractor][src-hlo], [extraction and deduplication helpers][src-hlo-common].

#### 7.6 arg_specs: Tuning Samples

Normalization samples are created by `_make_argspec`. They include input shapes/dtypes, parameter dtypes, the reduction axis, and normalization switches. One sample from the source is:

```python
# _src/ops/normalization/arg_specs.py
# sym:ARG_SPECS (first-entry excerpt)

_make_argspec(
    name="alphafold_384res_64chan",
    project="alphafold",
    x_shape=(384, 384, 64),
    dtype="bfloat16",
    param_dtype="float32",
)
```

`ArgSpec` provides the specification model. Before using these samples, check that they cover the actual shapes, dtypes, masks, grouping, and forward/backward modes of your workload. [Sample source][src-normalization-specs], [ArgSpec][src-arg-spec].

---

### Chapter 8: Heuristics — Default Configurations

#### 8.1 A Line-by-Line Walkthrough

The complete Normalization heuristic at this revision is:

```python
# _src/ops/normalization/pallas_triton_config.py
# sym:get_heuristics_config

def get_heuristics_config(
    x: jax.Array | jax.ShapeDtypeStruct,
    scale: jax.Array | jax.ShapeDtypeStruct | None,
    offset: jax.Array | jax.ShapeDtypeStruct | None,
    *,
    axis: int,
    block_size_per_warp: int = 1024,
    vmap_axis_sizes: tuple[int, ...],
    **_,
) -> Config:
  """Returns a config based on heuristics."""
  x = jax.ShapeDtypeStruct(canonicalize_shape(x.shape, axis), x.dtype)
  # We get diminishing returns, and worse load-balancing, with `block_m > 32`.
  # `block_m == 1` appears best whenever not reducing in trailing axis.
  block_m = 32 if x.ndim == 2 else 1
  if scale is None and offset is None:
    block_m = 1  # There is no oportunity to re-use data.

  block_size = block_m * pl.next_power_of_2(x.shape[1])
  num_blocks = pl.cdiv(x.shape[0], block_m) * math.prod(vmap_axis_sizes)
  if x.ndim > 2:
    # Read full cache line at a time.
    els_per_cache_line = (
        gpu_utils.CACHE_LINE_SIZE_BYTES // jnp.dtype(x.dtype).itemsize
    )
    block_n = min(els_per_cache_line, pl.next_power_of_2(x.shape[2]))
    block_size *= block_n
    num_blocks *= pl.cdiv(x.shape[2], block_n)
  else:
    block_n = None

  # Pick a block size that fits into registers, and launch enough blocks to fill
  # the device.
  max_block_size = gpu_utils.NUM_REGISTERS_PER_SM // 4
  min_num_blocks = 4 * jax.devices()[0].core_count
  while (block_m > 1) and (
      (block_size > max_block_size) or (num_blocks < min_num_blocks)
  ):
    block_m //= 2
    block_size //= 2
    num_blocks *= 2

  num_warps = min(pl.cdiv(block_size, block_size_per_warp), 4)
  return Config(block_m=block_m, block_n=block_n, num_warps=num_warps)
```

Following the code, it has four steps:

1. **Choose the initial block size**: Start with `block_m=32` for a reduction over the last axis and 1 otherwise. Also start at 1 when neither scale nor offset is present.
2. **Estimate the work**: Calculate elements per block and the number of blocks, including the vmap sizes. For three-dimensional shapes, also account for N and the number of elements per cache line.
3. **Halve the block size**: If blocks are too large or there are too few of them, repeatedly halve `block_m`, balancing reuse against the amount of schedulable work.
4. **Choose the warp count**: Estimate from a default of 1024 elements per warp, capped at 4 warps.

For an input with `M=4096, K=1024, N=1` and a scale parameter, take `core_count` to be 80:

| Step | block_m | Elements per block | Total blocks | Assessment |
|---|---:|---:|---:|---|
| Initial | 32 | 32768 | 128 | Blocks are too large, and the count is below the target of 320 |
| Halve once | 16 | 16384 | 256 | Still too few blocks |
| Halve again | 8 | 8192 | 512 | Both estimated conditions are satisfied |

The result is `num_warps=min(ceil(8192/1024),4)=4`. This is an evaluation of the rules. The register constants and thresholds are not the compiler's actual register allocation or measured device timings. [Heuristic implementation][src-normalization-config].

#### 8.2 The Role of Heuristics

Heuristics serve two purposes:

- On a cache miss with the `heuristics` strategy, they provide the configuration directly.
- During autotuning, they provide one starting candidate in the search set.

These rules aim to provide a reasonable configuration; finding the fastest still requires measuring candidates. Forward and backward passes can share rules while changing parameters. Normalization VJP reuses the forward heuristic but changes `block_size_per_warp` to 2048. [Backward heuristic][src-vjp-config].

TPU Attention provides a default Splash configuration and enumerates candidates. Ragged Dot TPU v2 can delegate tile selection to the lower-level implementation. Each backend defines its own strategy. [TPU Attention][src-attention-tpu], [TPU v2][src-ragged-tpu-v2].

---

### Chapter 9: The default_config Decision Chain

#### 9.1 Configuration Priority and the NullConfig Special Case

`default_config` translates the cache-miss strategy into arguments for `get_config`. The method's core logic is:

```python
# _src/ops/op.py
# sym:BoundArguments.get_config

# TODO: Add logging.
if (config := self.op.config) is not None:
  return config

if (heuristics_config := self.heuristics_config) is _NULL_CONFIG:
  return heuristics_config

if check_autotuning_cache:
  if (data := self.cached_autotuning_data) is not None and data.items():
    return data.fastest_config

if autotune_configs is not None:
  return self.autotune(
      autotune_configs, cache_results=cache_autotuning_results
  ).fastest_config

if allow_heuristics:
  return heuristics_config

raise ValueError(f"No config found for {self}.")
```

**Read selection priority together with evaluation order**:

- An explicit Config returns first.
- Heuristics are evaluated before cache lookup to identify the `_NULL_CONFIG` singleton. An Op with no tunable parameters stops here.
- For a tunable Op, a nonempty cache takes precedence over the heuristic result.
- After a miss, the setting selects search, heuristics, or an error. A failed search does not automatically retry with heuristics here. [Decision implementation][src-op-config].

#### 9.2 Fine-Grained Control with get_config()

```python
# _src/ops/op.py
# sym:BoundArguments.get_config

def get_config(
    self,
    check_autotuning_cache: bool = True,
    autotune_configs: set[C] | type[AUTO] | None = None,
    cache_autotuning_results: bool = True,
    allow_heuristics: bool = True,
) -> C:
  ...
```

| Parameter | What it controls |
|---|---|
| `check_autotuning_cache` | Whether this call queries the cache; explicit Config still takes precedence |
| `autotune_configs=None` | Does not run a search |
| `autotune_configs=AUTO` | Uses the union of the heuristic configuration and default backend candidates |
| `autotune_configs={...}` | Uses the supplied candidate set |
| `cache_autotuning_results` | Whether search results are written to the process cache |
| `allow_heuristics` | Whether the heuristic result may be selected at the end; does not prevent its earlier evaluation |

`AUTO` is a marker type in `ops.op`, with different semantics from `None`.

#### 9.3 Global Configuration

The public option `tokamax.config.autotuning_cache_miss_fallback` accepts:

| Value | Behavior after a cache miss |
|---|---|
| `"heuristics"` (default) | Uses the heuristic configuration |
| `"autotune"` | Measures candidates and uses the fastest successful one |
| `"error"` | Raises an error when no configuration is found |

A complete CPU example:

```python
import jax.numpy as jnp
import tokamax

x = jnp.arange(24, dtype=jnp.float32).reshape(3, 8)
with tokamax.config.autotuning_cache_miss_fallback("error"):
    y = tokamax.layer_norm(x, scale=None, offset=None, implementation="xla")
print(y.shape)
```

This still succeeds because the XLA Op returns through the `NullConfig` fast path first. To check Triton cache coverage, select an actual tunable backend and trigger tracing with representative inputs. [Configuration entry point][src-public-config].

#### 9.4 The Full Sequence

```text
User calls op(x)
  │
  ├─ bind, shape checks, and device checks
  ├─ construct array/batch/VJP wrappers
  └─ execute internal fwd
       ├─ reconstruct and bind arguments, including batch information
       ├─ config = ba.default_config
       │    ├─ Explicit Config?       → use it directly
       │    ├─ Heuristics is Null?     → no tunable parameters; return
       │    ├─ Nonempty cache hit?     → fastest_config
       │    └─ Miss strategy
       │         ├─ autotune           → measure candidates
       │         ├─ heuristics         → use the computed heuristic result
       │         └─ error              → raise an error
       ├─ write abstract arguments and Config into metadata
       ├─ self._fwd(..., config=config)
       └─ return output and any residuals required by the call mode
```

We now have one connected call path showing how a configuration is produced and how the kernel consumes it.

---

## Part 3: Infrastructure

Finally, we look at the Op base class connecting kernels to the tuning system, followed by the tools for automatic differentiation, timing, batching, precision, and serialization.

---

### Chapter 10: The Op Base Class System

#### 10.1 Five Generic Parameters

```python
# _src/ops/op.py
# sym:Op (signature and field excerpt)

@dataclasses.dataclass(frozen=True)
class Op[**P, T, R, C, K: Hashable](abc.ABC):
    config_cls: ClassVar[type[Any]] = NullConfig
    supports_symbolic_shapes: ClassVar[bool] = True
    supports_batched_args_capture: ClassVar[bool] = True
    config: C | None = None
```

| Slot | Name | Meaning | Normalization forward example |
|---|---|---|---|
| 1 | P | Input parameter signature | x, scale, offset, and keyword arguments |
| 2 | T | Output type | `jax.Array` |
| 3 | R | Operator residual type | `(mean, rstddev)` |
| 4 | C | Configuration type | Triton Normalization Config |
| 5 | K | Hashable tuning key type | `immutabledict` |

This uses Python 3.12 type parameter syntax. `Op.replace` creates a new instance and treats `vjp=None` specially. A backend's `__post_init__` may automatically fill in a default VJP, so replace must ensure that the caller's explicit request to remove the VJP takes effect. [Op and replace][src-op-call].

#### 10.2 The Complete `__call__` Lifecycle in Eight Steps

```text
op(*args, **kwargs)
  │
  ├─ 1. Check symbolic shapes
  │     supports_symbolic_shapes / contains_symbolic_shape
  │
  ├─ 2. Bind arguments
  │     self.bind → validate and fill defaults → BoundArguments
  │
  ├─ 3. Check devices
  │     infer_devices / supported_on
  │     bypass_device_check or cross_compile can affect this step
  │
  ├─ 4. Split arguments
  │     flatten → arrays / other / merge
  │
  ├─ 5. Define fwd with batch capture
  │
  ├─ 6. Define internal forward logic
  │     reconstruct arguments → bind again → default_config
  │     → metadata → self._fwd(config=...)
  │
  ├─ 7. Set up VJP conditionally
  │     handwritten VJP, or automatic VJP for batch capture compatibility
  │     no VJP + batch capture disabled allows a direct call
  │
  └─ 8. return f(*arrays)
        invoke the wrapper constructed above
```

Steps 5–7 construct wrappers; the final step actually invokes them. Reading only `_fwd` misses input validation, batch information, configuration selection, and automatic differentiation support. [Call lifecycle][src-op-call].

#### 10.3 BoundArguments

```python
# _src/ops/op.py
# sym:BoundArguments (fields and initialization excerpt)

@dataclasses.dataclass(frozen=True, slots=True)
class BoundArguments[C, K: Hashable]:
    op: Op[..., Any, Any, C, K]
    arguments: Mapping[str, Any]

    def __post_init__(self):
        immutable_args = immutabledict.immutabledict(self.arguments)
        object.__setattr__(self, "arguments", immutable_args)
```

This turns a workload into an object that can be operated on directly:

- `args` / `kwargs`: Recover call arguments.
- `default_config` / `heuristics_config`: Obtain configurations.
- `autotuning_cache_key` / `cached_autotuning_data`: Query the cache.
- `autotuning_configs` / `autotune()`: Generate and measure candidates.
- `benchmark()` / `vjp_arg_spec`: Measure performance and construct backward specifications.

Whether created manually through `bind` or extracted from StableHLO, the resulting objects use these same methods. [BoundArguments][src-op-config], [measurement and tuning entry points][src-op-autotune].

#### 10.4 The Residuals System

```python
# _src/ops/op.py
# sym:Residuals (field excerpt)

@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, slots=True)
class Residuals[T, R]:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    out: T
    residuals: R
    ...
```

The framework's residual wrapper contains the arguments, output, and operator residuals. Custom PyTree flatten/unflatten methods carry them between the forward and backward passes. The operator's own R holds the intermediate information required by the mathematics.

| Operator/path | Residual contents | Notes |
|---|---|---|
| LayerNorm | mean, rstddev | Statistics over the actual reduction axis |
| RMSNorm | None, rstddev | Does not save a mean |
| TPU Attention / Splash | max_logits, logsumexp | Used according to the contract between this backend's forward pass and its matching VJP |
| Ragged Dot | May require the dot output before activation | Depends on activation and residual mode |

Identical tuple structures do not make residuals interchangeable across backends. [Framework residuals][src-op-call], [TPU Attention residuals][src-attention-tpu].

#### 10.5 The VJP Mechanism

When there is no handwritten VJP, wrapping depends on batch capture:

```python
# _src/ops/op.py
# sym:Op.__call__ (VJP branch excerpt)

if self.vjp is None:
    if not self.supports_batched_args_capture:
        return f(*arrays)
    bwd = lambda f_vjp, dout: f_vjp(dout[0] if return_residuals else dout)
else:
    ...

f = jax.custom_vjp(f)
f.defvjp(fwd, bwd, optimize_remat=True)
return f(*arrays)
```

The handwritten backward branch unpacks Residuals, calls `self.vjp`, and arranges gradients to match the input structure. If gradients are returned as a dictionary, omitted relevant array parameters receive zero gradients.

When no handwritten VJP exists but custom batching needs to capture context, the implementation builds a gradient function with `jax.vjp` and connects it through `custom_vjp`. Support for differentiation and combinations of transformations still depends on the kernel implementation. [Complete VJP branches][src-op-call].

#### 10.6 Three Roles of base.py

The Normalization base class provides three things:

1. **XLA backend**: `IMPLEMENTATIONS['xla'] = base.Normalization()`.
2. **Operator base class**: Supplies the parameter contract inherited by backends such as Triton.
3. **Reference implementation**: Supplies a pure JAX numerical reference for new backends.

The base `_fwd` promotes data to at least FP32, performs the reduction and optional scale/offset operations, and casts back to the input dtype. When residuals are requested, it also returns the statistics. [Base implementation][src-normalization-base].

---

### Chapter 11: The Benchmarking System

`_src/benchmarking.py` provides general facilities for function standardization, compilation, and timing. Autotuning builds candidate search on top of them.

#### 11.1 BenchmarkData

```python
# _src/benchmarking.py
# sym:BenchmarkData

@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkData:
  """Time and memory benchmarking data."""

  compile_time_ms: float
  lower_time_ms: float
  evaluation_times_ms: tuple[float, ...]
  metadata: dict[str, Any]
  # TODO: Remove default value once all users have been migrated.
  peak_memory_mb: float | None = None

  @property
  def median_evaluation_time_ms(self) -> float:
    return float(np.median(self.evaluation_times_ms))
```

`compile_benchmark` times `.lower(x)` and `.compile()` separately. The returned runner then records execution time. `median_evaluation_time_ms` is calculated from the execution samples. Peak memory comes from the compiled program's `memory_analysis()` and is an estimate reported by the compiler. [Data model and compilation entry point][src-benchmark].

#### 11.2 standardize_function

The function and its arguments are organized as `(fn, array_args)`. `fn(array_args)` takes only the collection of arrays, while other objects remain in a closure. Abstract inputs can be randomly initialized, and batch specifications can be reconstructed as the corresponding vmap.

| mode | Construction at this revision | Measurement focus |
|---|---|---|
| `forward` | Ordinary forward function | Inference / forward-only execution |
| `forward_res` | `jax.vjp(forward, arrays)[0]` | Enters the AD forward path; does not mean that all residuals are returned |
| `vjp` | Obtains the VJP in advance, then uses the cotangent as input | Backward pass; captured intermediates affect compilation and memory |
| `forward_and_vjp` | Places forward and VJP in one function with an optimization barrier | Combined forward and backward passes |

Different modes can compile into different programs, especially when the backend changes its fusion path based on `return_residuals`. [Function standardization][src-benchmark].

#### 11.3 Four Timing Methods

| Method | Timing path in the source | Scope |
|---|---|---|
| `cupti` | JAX `profiler.Cupti` | GPU device execution, including the profiler's own effects |
| `hermetic_xprof` | JAX profiler → local profile → XProf analysis | GPU/TPU; the TPU default |
| `xprof` | XProf session | Requires the corresponding session components, which may be unavailable in an open-source environment |
| `wallclock` | `perf_counter` + `block_until_ready` | General-purpose; includes host scheduling and waiting |

The default is selected by platform: GPU → CUPTI, TPU → hermetic XProf, and everything else → wallclock. The source does not automatically retry with wallclock after a profiler exception. [Timers and default selection][src-benchmark].

**Complete CPU example**:

```python
import jax
import jax.numpy as jnp
import tokamax

x = jnp.arange(64 * 17, dtype=jnp.float32).reshape(64, 17)

def normalize(a):
    return tokamax.layer_norm(a, scale=None, offset=None, implementation="xla")

fn, args = tokamax.standardize_function(normalize, x, mode="forward")
data = tokamax.benchmark(fn, args, method="wallclock", iterations=3)
assert len(data.evaluation_times_ms) == 3
print("lower_ms:", data.lower_time_ms)
print("compile_ms:", data.compile_time_ms)
print("median_wallclock_ms:", data.median_evaluation_time_ms)
```

This code validates the interface and return values. Local wallclock numbers are not GPU/TPU performance results. For a meaningful comparison, fix the shape, dtype, mode, device, and software versions.

**Reading XProf activity time**:

```python
# _src/benchmarking.py
# sym:XprofProfileSession._compute_timing_summary / total_op_time (excerpt)

self._timing_summary = get_kernel_stats_tool.compute_kernel_stats(
    self._profile,
    output_format="dict",
    include_summary=True,
    trace_matchers=trace_matchers,
)
total_us = self._timing_summary.get("total_device_duration_us", 0.0)
```

This path uses activity-interval statistics so that overlapping events are not counted twice by simply adding their durations. The result differs from end-to-end latency. If event filters are used, specify which events were selected. [XProf integration][src-benchmark].

---

### Chapter 12: Batching / vmap Utilities

#### 12.1 BatchedShapeDtype

```python
# _src/batching.py
# sym:BatchedShapeDtype (field excerpt)

class BatchedShapeDtype(jax.ShapeDtypeStruct):
    __slots__ = ("vmap_axes",)
    vmap_axes: tuple[tuple[int, int] | None, ...]
    ...
```

An ordinary shape describes the local input. `vmap_axes` additionally records the position and size of each mapped axis, while `vmap_shape` reconstructs the shape with batch axes inserted. Heuristics use this information to estimate the work generated by the entire batched call. [Batching types][src-batching].

#### 12.2 capture_batched_args

The key connections in the capture rule are:

```python
# _src/batching.py
# sym:capture_batched_args (wiring excerpt)

new_vmap_axis = None if axis is None else (axis, axis_size)
new_vmap_axes = (new_vmap_axis, *shape.vmap_axes)
new_shape = BatchedShapeDtype(shape.shape, shape.dtype, new_vmap_axes)

fn_vmap = jax.custom_batching.custom_vmap(fn_flat)
fn_vmap.def_vmap(vmap_rule)
```

Captured specifications reach the backend through BoundArguments. This is where the `product(vmap_axis_sizes)` factor multiplying `num_blocks` in the Normalization heuristic comes from. Whether the cache key uses all batch information depends on the backend's key function. [Capture rule][src-batching].

#### 12.3 vmap_maybe_bcast and vmap_split

| Utility | Behavior | Constraint |
|---|---|---|
| `vmap_maybe_bcast` | Squeezes relevant input axes of size 1; when all inputs broadcast, calls the function once and restores the output axis | Adapts to broadcasting relationships |
| `vmap_split` | Splits an axis according to `num_parts`, then applies vmap/broadcast logic | Non-broadcast axes must satisfy divisibility requirements |

The caller chooses how `vmap_split` partitions the axis. It does not automatically find a batch size that fits a memory budget. Actual scheduling and parallelism depend on the program generated by the backend. [vmap utilities][src-batching].

---

### Chapter 13: Precision and Quantization

#### 13.1 The Precision System

```python
# _src/precision.py
# sym:DotAlgorithm / DotAlgorithmPreset / Precision

DotAlgorithm = jax.lax.DotAlgorithm
DotAlgorithmPreset = jax.lax.DotAlgorithmPreset
Precision = jax.lax.Precision
```

`F16_F16_F32`, `BF16_BF16_F32`, and `TF32_TF32_F32` are JAX dot-product algorithm presets. Tokamax normalizes `PrecisionLike`, then uses the input dtypes and platform to map it to a representation supported by the backend. [Precision adaptation][src-precision].

Attention carries separate precision information for the QK and PV dot products. These are independent choices for two operations, not “double precision” in the FP64 sense. [Attention contract][src-attention-base].

#### 13.2 Quantization Support

Qwix's `QArray` carries quantized data, scales, and related information. `AsQArray` preserves the intent to quantize later:

```python
# _src/quantization.py
# sym:AsQArray (fields and conversion excerpt)

@dataclasses.dataclass(frozen=True, slots=True)
class AsQArray:
    value: jax.Array
    qtype: jax.typing.DTypeLike
    _: dataclasses.KW_ONLY
    channelwise_axes: Collection[int] = ()
    tiled_axes: Mapping[int, int | float] | None = None
    calibration_method: str = "absmax"

    def as_qarray(self) -> QArray:
        return qwix.quantize(
            self.value,
            self.qtype,
            channelwise_axes=self.channelwise_axes,
            tiled_axes=self.tiled_axes,
            calibration_method=self.calibration_method,
        )
```

| Conversion | Result |
|---|---|
| `as_array_or_qarray` | Quantizes AsQArray into QArray when needed |
| `as_array` | Further dequantizes to an ordinary array when needed |
| `as_array_or_qarray_without_zero_point` | Dequantizes when the corresponding zero-point path is unsuitable |

This representation allows quantization to be fused into a kernel. Whether intermediate tensors are materialized depends on the backend path. Ragged Dot's quantization bit width, scale tiles, layout, and forward/backward direction all affect implementation selection. [Quantization adaptation][src-quantization], [GPU quantization paths][src-ragged-gpu], [TPU quantization paths][src-ragged-tpu].

---

### Chapter 14: Configuration and Utility Libraries

#### 14.1 The Global Configuration System

`_ConfigOption` connects `absl.flags` with JAX user context:

```python
# _src/config.py
# sym:_ConfigOption (initialization excerpt)

def __post_init__(self):
    if self.config is None:
        object.__setattr__(self, "config", jax.make_user_context(_DEFAULT))
```

Scoped overrides use `self.config(value)`. Reads prefer the current context and use the flag only when there is no override. The first parse uses `known_only=True`, allowing library configuration to coexist with other arguments in the host CLI. [Configuration implementation][src-config].

| Option | Availability |
|---|---|
| `autotuning_cache_miss_fallback` | Publicly exported from `tokamax.config` |
| `cross_compile` | Publicly exported from `tokamax.config` |
| `ignore_autotuning_cache` | Internal option in `_src.config` |
| `disable_multi_core_mode` | Temporary option in `_src.config` used by some GMM/TGMM v2 paths |

[Public configuration exports][src-public-config].

#### 14.2 HLO Utilities

| Entry point/structure | Responsibility |
|---|---|
| `get_kernel_info` | Extracts relevant kernel information from StableHLO |
| `get_opspecs` | Recovers BoundArguments from metadata or the older name-based path |
| `get_bound_args` | Removes the selected configuration and extracts unique workloads |
| `TritonKernelInfo` / `MosaicGpuKernelInfo` / `MosaicTpuKernelInfo` | Describe different kernel paths |
| `dedupe_wrapper_kernels` | Applies specific deduplication to conversions, broadcasts, and other wrappers carrying a payload |
| `DISABLE_JAX_EXPORT_CHECKS` | Allows the listed Triton custom calls to be exported |

The number of introspection records, the number of unique workloads, and the final number of device launches are three different metrics. Exporting StableHLO containing custom kernels does not remove device or ABI constraints. [HLO utilities][src-hlo], [information structures and deduplication][src-hlo-common].

#### 14.3 Shape and Numerical Utilities

| Function/type | Location | Purpose |
|---|---|---|
| `exact_div` | `utils.py` | Checks exact divisibility |
| `split_merge` | `utils.py` | Splits arrays and other objects by a predicate and provides a reconstruction function |
| `canonicalize_shape_3d` | Normalization Config module | Maps input dimensions to M/K/N |
| `pad_to_next_multiple_of` | `shape.py` | Pads to a specified multiple |
| `contains_symbolic_shape` | `shape.py` | Checks for symbolic shapes |
| `random_initialize` | `numerics.py` | Generates data for abstract inputs |
| `DiffSummary` / `array_diff_summary` | `numerics.py` | Summarize numerical differences |

[Shape utilities][src-shape], [general utilities][src-utils], [numerical utilities][src-numerics].

#### 14.4 Device Capability Checks

```python
# _src/gpu_utils.py
# sym:is_sm80 / is_sm90 / is_sm100

def is_sm80(device=None):
    return _cc_between(8.0, 9.0, device)

def is_sm90(device=None):
    return _cc_between(9.0, 10.0, device)

def is_sm100(device=None):
    return _cc_between(10.0, 11.0, device)
```

The common Mosaic GPU check allows experimental SM80 support, but individual operators impose further restrictions. Cross-compilation can affect device checks; it cannot replace execution on the actual device. [GPU capability checks][src-gpu-utils].

This commit also adds device-kind inference. When there is no concrete input device, it tries to read `abstract_device.device_kind` from the active abstract mesh to look up the target device's tuning cache. Whether a particular kernel can compile without the target hardware still needs to be checked individually. [Device inference][src-op-device], [corresponding commit][src-revision].

#### 14.5 JAX Type Serialization

`pydantic.py` provides adapters for:

- **ShapeDtype**: Shapes, dtypes, and related JAX specifications.
- **NumPy dtype encoding/decoding**: Converts dtypes to a serializable representation and back.
- **PowerOfTwo**: Validates powers of two.
- **AnyInstanceOf**: Encodes instances together with their concrete class information.
- **Argument specification models**: Creates validation/serialization models from an Op's function signature.

These adapters connect Config, BoundArguments, JSON caches, and metadata so that decoded data can satisfy the operator contract again. [Serialization adapters][src-pydantic].

#### 14.6 ad.py — VJP Construction Utilities

```python
# _src/ad.py
# sym:get_vjp_taking_residuals

def get_vjp_taking_residuals[T, R](
    fn: Callable[..., tuple[T, R]], *primals
) -> Callable[[R, T], Any] | None:
  ...
```

The function analyzes and rewrites jaxpr, attempting to construct a VJP that explicitly accepts user-provided residuals. It can succeed only if those residuals suffice to compute the gradient. Returning `None` means that a VJP could not be constructed from that residual set. [AD utilities][src-ad].

#### 14.7 Other Small Utilities

- `jaxtyping.py`: Type-checking adapters and related switches.
- `mosaic_gpu.py`: Helpers for GPU layouts, quantization encoding, synchronization, and resource estimation.
- `mosaic_tpu.py`: TPU quantization BlockSpecs, generation-specific parameters, and buffered pipelines.
- `version.py`: Version constants and the Git build identifier written during release; the latter may be empty in a source checkout.

[GPU utilities][src-mosaic-gpu], [TPU utilities][src-mosaic-tpu], [version definition][src-version].

---

### Chapter 15: Key Design Patterns

#### 15.1 Strategy Pattern and Fallback Chain

The API registers implementations and tries them in order. Each concrete Op then chooses its own Config. Backend selection and configuration selection are separate layers and should be examined separately when analyzing performance. [API dispatch][src-normalization-api], [configuration decisions][src-op-config].

#### 15.2 Op as a Hashable Configuration Object

A frozen dataclass makes Op an immutable configuration object, and `replace` produces a new instance. Configurations, argument specifications, and caches are organized around this object. [Op base class][src-op-call].

#### 15.3 Offline Autotuning Through HLO Introspection

Abstract arguments are written into metadata, then recovered from StableHLO as independent workloads. Tuning tools can reuse call specifications from the model without manually reconstructing every layer. Compiling candidates and recompiling the model with new configurations are still required. [Introspection entry point][src-hlo].

#### 15.4 Offline Results and Scoped Overrides

Bundled JSON, the process cache, and AutotuningResult overlays serve different purposes: preset data, reuse within a process, and temporary overrides. Explicitly saved results can be reused across sessions. Record the source SHA, device, and software environment alongside them. [Cache][src-cache], [result scopes][src-autotuning-result].

#### 15.5 Combining custom_vjp with custom_vmap

When custom batching needs to capture context, Op uses the corresponding VJP wrapper to handle the composition. With no VJP and batch capture disabled, a direct-call branch is available. The extent of support in a particular backend still needs testing. [Automatic differentiation branches][src-op-call].

#### 15.6 Managing Context and Shared State

Global configuration uses JAX user context, overlays and nested metadata use ContextVar, and Pallas load/store patches use a lock. They address JIT cache distinctions, scope restoration, and modifications to shared Python attributes, respectively. [Configuration][src-config], [Op context][src-op-call], [Pallas wrapper][src-block].

#### 15.7 Optional Imports and Backend Fallback

Modules register available backends through optional imports, then use device and input checks to decide the call path. Being importable does not mean that a backend supports the current device/shape. The XLA reference implementation provides an important portable path. [Backend registration][src-normalization-api].

#### 15.8 Reference Implementation = Base Class

The pure JAX base class serves three roles: parameter contract, reference computation, and XLA backend. A new kernel can start by reusing its contract and numerical reference, then add tiling, quantization, VJP, and performance strategies incrementally. [Normalization base class][src-normalization-base].

---

## Appendix

### A. Main File Map

The table below lists the main files discussed in the article. Tests and experimental files can be explored further in their corresponding source directories.

<table>
<tr><th>File</th><th>Purpose</th></tr>
<tr><td colspan="2"><b>Package top level — public entry points</b></td></tr>
<tr><td><code>__init__.py</code></td><td>Computation APIs, Op, autotune, and benchmark exports</td></tr>
<tr><td><code>autotuning.py</code></td><td>get_bound_args and JSON specification utilities</td></tr>
<tr><td><code>benchmarking.py</code></td><td>Timing helper entry points</td></tr>
<tr><td><code>config.py</code></td><td>Two public configuration options</td></tr>
<tr><td colspan="2"><b>_src/ops/ — operator implementations</b></td></tr>
<tr><td><code>op.py</code></td><td>Op, BoundArguments, configuration, metadata, VJP</td></tr>
<tr><td><code>normalization/api.py</code></td><td>Public LayerNorm / RMSNorm dispatch</td></tr>
<tr><td><code>normalization/base.py</code></td><td>Input contract and reference implementation</td></tr>
<tr><td><code>normalization/pallas_triton.py</code></td><td>Forward wrapper and kernel</td></tr>
<tr><td><code>normalization/pallas_triton_config.py</code></td><td>Config, key, heuristics</td></tr>
<tr><td><code>normalization/pallas_triton_vjp.py</code></td><td>Triton backward pass</td></tr>
<tr><td><code>normalization/pallas_triton_vjp_config.py</code></td><td>Backward configuration and keys</td></tr>
<tr><td><code>normalization/arg_specs.py</code></td><td>Typical workloads</td></tr>
<tr><td><code>attention/api.py</code></td><td>Attention dispatch and public shape contract</td></tr>
<tr><td><code>attention/base.py</code></td><td>Reference implementation, Mask, PagingInfo, and other contracts</td></tr>
<tr><td><code>attention/pallas_triton.py</code></td><td>Triton Config and kernel</td></tr>
<tr><td><code>attention/pallas_mosaic_gpu.py</code></td><td>Mosaic GPU architecture dispatch</td></tr>
<tr><td><code>attention/pallas_mosaic_gpu_kernel_sm90.py</code></td><td>SM90 Attention</td></tr>
<tr><td><code>attention/pallas_mosaic_gpu_kernel_sm100.py</code></td><td>SM100 Attention</td></tr>
<tr><td><code>attention/pallas_mosaic_tpu.py</code></td><td>TPU Config and Splash forward adapter</td></tr>
<tr><td><code>attention/xla_chunked.py</code></td><td>Chunked XLA path</td></tr>
<tr><td><code>attention/jax_nn.py</code></td><td>JAX / cuDNN Attention adapter</td></tr>
<tr><td><code>ragged_dot/api.py</code></td><td>Ragged Dot / General entry points</td></tr>
<tr><td><code>ragged_dot/pallas_mosaic_gpu.py</code></td><td>GPU grouped matrix multiplication and quantization dispatch</td></tr>
<tr><td><code>ragged_dot/pallas_mosaic_tpu.py</code></td><td>TPU GMM/TGMM adapter</td></tr>
<tr><td><code>ragged_dot/pallas_mosaic_tpu_v2.py</code></td><td>TPU v2 Op and connection to lower-level tiling</td></tr>
<tr><td><code>gated_linear_unit/api.py</code></td><td>Public GLU entry point</td></tr>
<tr><td><code>gated_linear_unit/pallas_mosaic_gpu.py</code></td><td>GLU architecture and residual-path dispatch</td></tr>
<tr><td><code>linear_softmax_cross_entropy_loss/api.py</code></td><td>XLA, chunked XLA, and TPU entry points for Linear CE</td></tr>
<tr><td><code>triangle_multiplication/api.py</code></td><td>XLA path for Triangle Multiplication</td></tr>
<tr><td><code>ragged_gather/api.py</code>, <code>ragged_scatter/api.py</code>, <code>ragged_gather_reduce/api.py</code></td><td>Internal ragged data-movement entry points</td></tr>
<tr><td><code>experimental/kda/api.py</code>, <code>experimental/mla/api.py</code>, <code>experimental/tpu/topk/api.py</code></td><td>Experimental computation entry points</td></tr>
<tr><td colspan="2"><b>_src/autotuning/ — autotuning</b></td></tr>
<tr><td><code>api.py</code></td><td>autotune, AutotuningResult, implementation expansion</td></tr>
<tr><td><code>autotuner.py</code></td><td>Compilation executors, measurement, AutotuningData</td></tr>
<tr><td><code>cache.py</code></td><td>Lazy JSON loading and key reconstruction</td></tr>
<tr><td><code>arg_spec.py</code></td><td>Sample specification models</td></tr>
<tr><td colspan="2"><b>_src/pallas/ — Pallas extensions</b></td></tr>
<tr><td><code>block.py</code></td><td>BlockRef, masks, pallas_call wrapper</td></tr>
<tr><td><code>grid.py</code></td><td>Grouping cost and program ID reordering</td></tr>
<tr><td colspan="2"><b>_src/ — infrastructure</b></td></tr>
<tr><td><code>benchmarking.py</code></td><td>Function standardization, compilation, timing, and profiling</td></tr>
<tr><td><code>hlo_utils.py</code> / <code>hlo_utils_common.py</code></td><td>StableHLO introspection, kernel information, and deduplication</td></tr>
<tr><td><code>batching.py</code></td><td>Batch specifications and custom_vmap capture</td></tr>
<tr><td><code>precision.py</code></td><td>Precision representation normalization</td></tr>
<tr><td><code>quantization.py</code></td><td>QArray / AsQArray conversions</td></tr>
<tr><td><code>config.py</code></td><td>absl.flags and JAX context</td></tr>
<tr><td><code>shape.py</code> / <code>utils.py</code></td><td>Shape handling, exact division, splitting, and reconstruction</td></tr>
<tr><td><code>ad.py</code></td><td>VJP construction from specified residuals</td></tr>
<tr><td><code>gpu_utils.py</code></td><td>GPU capability checks</td></tr>
<tr><td><code>pydantic.py</code></td><td>JAX type and Op encoding/decoding</td></tr>
<tr><td><code>numerics.py</code></td><td>Sample initialization and numerical comparison</td></tr>
<tr><td><code>jaxtyping.py</code></td><td>Type-checking adapters</td></tr>
<tr><td><code>mosaic_gpu.py</code> / <code>mosaic_tpu.py</code></td><td>Backend utilities</td></tr>
<tr><td><code>version.py</code></td><td>Version and build identifier</td></tr>
</table>

### B. Backend Support Matrix

This table shows the implementation paths at the pinned revision. Each backend still has its own restrictions on dtype, shape, quantization, masks, residuals, and differentiation. XLA computation may remain available when there is no dedicated kernel.

| Backend | Attention | RaggedDot | GLU | LayerNorm | TriangleMul | LinearCE |
|---|---|---|---|---|---|---|
| XLA | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| cuDNN | ✅ | — | — | — | — | — |
| Triton | ✅ | ✅ | ✅ | ✅ | — | — |
| Mosaic GPU | ✅ | ✅ | ✅ | — | — | — |
| Mosaic TPU | ✅ | ✅ | — | — | — | ✅ |
| Mosaic TPU v2 | — | ✅ | — | — | — | — |

Implementation names vary by API. Chunked XLA is called `xla_chunked` for Attention and `chunked_xla` for Linear CE. Ragged Dot v2 is `mosaic_tpu_v2`. Internal and experimental operators are not counted as top-level APIs in this table. [Attention][src-attention-api], [Ragged Dot][src-ragged-api], [GLU][src-glu-api], [Normalization][src-normalization-api], [Linear CE][src-linear-api], [Triangle][src-triangle-api].

### C. End-to-End Data Flow

```text
                  User code: layer_norm(x, scale, offset)
                                      │
                                      ▼
                       api.py: select a backend in priority order
                                      │
                                      ▼
                       Op.__call__: bind, check, build wrappers
                                      │
                                      ▼
                    Internal fwd: select Config for this workload
                         ├─ explicit configuration
                         ├─ NullConfig fast return
                         ├─ cache hit → fastest_config
                         └─ miss → autotune / heuristics / error
                                      │
                         ┌────────────┴─────────────┐
                         ▼                          ▼
                  Record metadata             Backend _fwd
                         │                    shape → Fusion
                         │                    → BlockSpec / grid
                         │                    → pallas_call
                         │                          │
                         │                          ▼
                         │              load → block computation → store
                         │
                         ▼
                 StableHLO after lowering
                         │
                         ▼
                 get_bound_args extracts specifications
                         │
                         ▼
                 Compile candidates → benchmark each candidate
                         │
                         ▼
                   AutotuningResult
                         │
                         └─ with result → reuse selected Config in later compilation
```

To follow the source's evolution, compare [0.0.12 → this article's main revision][compare-old] and [the 0.0.13 release snapshot → this article's main revision][compare-release]. Major changes include metadata and context mechanisms, experimental SM80 paths, TPU v2, new operator directories, and target-device cache lookup in the pinned main revision. The fixed SHA makes these changes easier to revisit.

Source excerpts in this article come from the official public repository under Apache-2.0. Complete examples illustrate invocation and validation workflows. [Upstream license][src-license].


[src-revision]: https://github.com/openxla/tokamax/commit/f21f67cc921f6ad56785a084ba30dc3d85278e9a
[pypi-release]: https://pypi.org/project/tokamax/0.0.13/
[compare-old]: https://github.com/openxla/tokamax/compare/964354016004720905931f1249706a1772706752...f21f67cc921f6ad56785a084ba30dc3d85278e9a
[compare-release]: https://github.com/openxla/tokamax/compare/96e18848f78db56a19a1422ffb0e5ee3cc30d64f...f21f67cc921f6ad56785a084ba30dc3d85278e9a
[src-version]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/version.py
[src-project]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/pyproject.toml
[src-exports]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/__init__.py
[src-license]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/LICENSE
[src-ops-tree]: https://github.com/openxla/tokamax/tree/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops
[src-normalization-tree]: https://github.com/openxla/tokamax/tree/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/normalization
[src-normalization-api]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/normalization/api.py#L46-L128
[src-normalization-base]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/normalization/base.py#L31-L134
[src-normalization-config]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/normalization/pallas_triton_config.py#L30-L125
[src-normalization-kernel]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/normalization/pallas_triton.py#L40-L79
[src-normalization-triton]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/normalization/pallas_triton.py#L82-L227
[src-normalization-specs]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/normalization/arg_specs.py
[src-op-call]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/op.py#L49-L410
[src-op-config]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/op.py#L417-L532
[src-op-cache-lookup]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/op.py#L534-L562
[src-op-autotune]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/op.py#L534-L623
[src-op-device]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/op.py#L679-L711
[src-block]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/pallas/block.py
[src-grid]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/pallas/grid.py
[src-cache]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/autotuning/cache.py#L52-L111
[src-autotuner]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/autotuning/autotuner.py
[src-autotuning-result]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/autotuning/api.py#L87-L197
[src-autotuning-api]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/autotuning/api.py#L204-L396
[src-arg-spec]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/autotuning/arg_spec.py
[src-benchmark]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/benchmarking.py
[src-hlo]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/hlo_utils.py
[src-hlo-common]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/hlo_utils_common.py
[src-config]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/config.py
[src-public-config]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/config.py
[src-batching]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/batching.py
[src-precision]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/precision.py
[src-quantization]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/quantization.py
[src-pydantic]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/pydantic.py
[src-numerics]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/numerics.py
[src-ad]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ad.py
[src-gpu-utils]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/gpu_utils.py
[src-mosaic-tpu]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/mosaic_tpu.py
[src-attention-api]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/attention/api.py
[src-attention-base]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/attention/base.py
[src-attention-triton]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/attention/pallas_triton.py
[src-attention-gpu]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/attention/pallas_mosaic_gpu.py
[src-attention-sm100]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py#L493-L573
[src-attention-tpu]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/attention/pallas_mosaic_tpu.py
[src-ragged-api]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/ragged_dot/api.py
[src-ragged-gpu]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py
[src-ragged-tpu]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py
[src-ragged-tpu-v2]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2.py
[src-glu-api]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/gated_linear_unit/api.py
[src-glu-gpu]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/gated_linear_unit/pallas_mosaic_gpu.py
[src-linear-api]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/linear_softmax_cross_entropy_loss/api.py
[src-triangle-api]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/triangle_multiplication/api.py
[jax-blockspec]: https://docs.jax.dev/en/latest/pallas/grid_blockspec.html
[jax-tpu]: https://docs.jax.dev/en/latest/pallas/tpu/details.html
[src-vjp-config]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/ops/normalization/pallas_triton_vjp_config.py
[src-shape]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/shape.py
[src-utils]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/utils.py
[src-mosaic-gpu]: https://github.com/openxla/tokamax/blob/f21f67cc921f6ad56785a084ba30dc3d85278e9a/tokamax/_src/mosaic_gpu.py

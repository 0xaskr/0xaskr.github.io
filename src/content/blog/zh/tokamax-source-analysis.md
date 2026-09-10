---
title: 'Tokamax 源码解析：从 Pallas 内核到自动调优'
description: '以 LayerNorm 为主线，沿固定源码提交解析 Tokamax 的 GPU/TPU 内核、Config 决策、调优缓存、Benchmarking 与 JAX 变换，附完整调用示例和源码链接。'
pubDate: '2026/9/10'
tags: ["Tokamax", "JAX", "Pallas", "GPU", "TPU", "Autotuning"]
---

## 阅读约定与版本基线

- **示例算子**：Part 1-2 以 Normalization 算子为主线，再拿 Attention、Ragged Dot 和 GPU/TPU 后端做对照。
- **代码路径**：所有路径相对于包目录 `tokamax/`。
- **`# sym:X`**：指对应源码里的符号。
- **前置知识**：默认用过 JAX 的数组、`jit`、`grad` 和 `vmap`。
- **源码版本**：全文固定main commit id [`f21f67cc921f6ad56785a084ba30dc3d85278e9a`][src-revision]。

---

## 源码目录总览

```text
tokamax/                                  # 包根目录
├── __init__.py                           # 计算 API、Op、调优和计时入口
├── autotuning.py                         # get_bound_args / JSON 规格工具
├── benchmarking.py                       # benchmark 辅助工具导出
├── config.py                             # 两个公开配置选项
├── benchmarks/                           # 性能测试入口
├── data/autotuning/                      # 72 个预置调优 JSON
├── experimental/utils/tuning/tpu/        # TPU 调优工具
└── _src/
    ├── ops/
    │   ├── op.py                         # Op / BoundArguments / 配置决策
    │   ├── normalization/
    │   │   ├── api.py                    # layer_norm 调度
    │   │   ├── base.py                   # Normalization / VJP 参考实现
    │   │   ├── pallas_triton.py          # 前向 Op、Fusion 适配与 kernel
    │   │   ├── pallas_triton_vjp.py      # 反向 Op 与 kernel
    │   │   ├── pallas_triton_config.py   # Config / key / heuristics
    │   │   ├── pallas_triton_vjp_config.py
    │   │   └── arg_specs.py              # 调优与测试样本
    │   ├── attention/                    # Triton、Mosaic GPU/TPU、XLA、cuDNN
    │   ├── ragged_dot/                   # 分组矩阵乘与量化变体
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
    │       ├── fused_moe_rs/              # fused MoE 实验路径
    │       └── tpu/                      # Splash / Ring Attention、TopK
    ├── autotuning/
    │   ├── api.py                        # autotune / AutotuningResult
    │   ├── autotuner.py                  # 候选编译、测量与结果
    │   ├── cache.py                      # JSON 懒加载
    │   └── arg_spec.py                   # ArgSpec 数据模型
    ├── pallas/
    │   ├── block.py                      # BlockRef / pallas_call 包装
    │   └── grid.py                       # program ID 分组与重排
    ├── benchmarking.py                  # standardize / compile / timer
    ├── hlo_utils.py                     # StableHLO 算子信息提取
    ├── hlo_utils_common.py              # kernel 信息结构与去重
    ├── batching.py                      # vmap 信息捕获
    ├── precision.py                     # 精度规范化
    ├── quantization.py                  # Qwix QArray / AsQArray 适配
    ├── config.py                        # absl.flags + JAX user context
    ├── shape.py                         # padding、广播、符号形状
    ├── ad.py                            # 残差驱动的 VJP 构建
    ├── gpu_utils.py                     # GPU 设备能力检查
    ├── pydantic.py                      # JAX 类型与 Op 序列化
    ├── numerics.py                      # 初始化与数值比较
    ├── utils.py                         # 整除、拆分与重组
    ├── mosaic_gpu.py                    # GPU 布局、量化与同步工具
    ├── mosaic_tpu.py                    # TPU 量化分块与 pipeline 工具
    └── version.py                       # 版本与构建标识
```

---

<details>
<summary>展开全文目录：三部分、十五章</summary>

- **Part 1: Kernel Layer — GPU / TPU 内核层**
  - [第一章：一个算子目录的结构](#第一章一个算子目录的结构)
  - [第二章：BlockSpec、Grid 与 Shape 规范化](#第二章blockspecgrid-与-shape-规范化)
  - [第三章：block.pallas_call — Pallas 扩展](#第三章blockpallas_call--pallas-扩展)
  - [第四章：其他算子 Kernel 总览](#第四章其他算子-kernel-总览)
- **Part 2: Autotuning System — 自动调优系统**
  - [第五章：Config — 可调参数定义](#第五章config--可调参数定义)
  - [第六章：缓存键与 Autotuning Cache](#第六章缓存键与-autotuning-cache)
  - [第七章：Benchmark 驱动的 Autotuning](#第七章benchmark-驱动的-autotuning)
  - [第八章：Heuristics — 启发式默认配置](#第八章heuristics--启发式默认配置)
  - [第九章：default_config 决策链](#第九章default_config-决策链)
- **Part 3: Infrastructure — 基础设施**
  - [第十章：Op 基类系统](#第十章op-基类系统)
  - [第十一章：Benchmarking 系统](#第十一章benchmarking-系统)
  - [第十二章：Batching / vmap 工具](#第十二章batching--vmap-工具)
  - [第十三章：精度与量化](#第十三章精度与量化)
  - [第十四章：配置系统与工具库](#第十四章配置系统与工具库)
  - [第十五章：关键设计模式总结](#第十五章关键设计模式总结)
- [附录](#附录)

</details>

---

## Part 1: Kernel Layer — GPU / TPU 内核层

先沿 `layer_norm` 的调用路径，从common API 走到kernel内部。Normalization 的归约、分块和边界处理都足够集中，正好用来观察一次 JAX 调用是怎么变成 Pallas kernel 的；之后再拿 Attention、Ragged Dot 和 GPU/TPU 后端做对照。

---

### 第一章：一个算子目录的结构

以 `_src/ops/normalization/` 为例：

| 文件 | 角色 | 说明 |
|---|---|---|
| `api.py` | 调度层 | 对外提供 `layer_norm`，按候选顺序选择实现 |
| `base.py` | 参考实现 + 算子基类 | 参数校验、纯 JAX 前向和 VJP 契约 |
| `pallas_triton.py` | Triton 前向后端 | `PallasTritonNormalization` 与 `_normalization_kernel` |
| `pallas_triton_vjp.py` | Triton 反向后端 | 独立的 VJP Op 与 kernel |
| `pallas_triton_config.py` | 前向配置 | Config、缓存键与启发式 |
| `pallas_triton_vjp_config.py` | 反向配置 | 反向自己的 Config 与键 |
| `arg_specs.py` | 调优样本 | 典型输入的 shape、dtype 和开关 |
| `*_test.py` / `test_base.py` | 测试 | 数值、变换与后端行为检查 |

**规律**：这个目录把 kernel、算子契约和调优策略拆开。Normalization 前向和反向各有自己的配置文件；别的算子有时把 Config 塞在后端文件里，有时抽到 `*_common.py`。[Normalization 目录][src-normalization-tree]

#### 四层调用链

从用户调用到设备计算，沿四层代码向下看：

```text
tokamax.layer_norm(x, scale, offset)
  │
  ▼
① api.py
  IMPLEMENTATIONS → 按顺序尝试 PallasTritonNormalization / Normalization
  │
  ▼
② op.py
  bind → 设备/形状检查 → 构造 batch/VJP 包装 → 内部 fwd 选择 config
  │
  ▼
③ pallas_triton.py::_fwd
  canonicalize_shape_3d → Fusion → BlockSpec → grid → block.pallas_call
  │
  ▼
④ _normalization_kernel
  BlockRef.load → mean / var / rstddev → scale / offset → BlockRef.store
```

在 `jax.jit` 下，选后端、选 Config 这些 Python 决策主要发生在 tracing/lowering 阶段。编译好的程序每次执行，用的都是此前定下的 kernel 配置。[Op 调用实现][src-op-call]

---

**① 调度层 `api.py` — 注册后端 + 按优先级回退**

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

注册完之后，公共入口把 `implementation` 规范成候选序列。下面保留关键分支，省略参数文档和具体调用参数：

```python
# _src/ops/normalization/api.py
# sym:layer_norm（结构节选）

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

几个值得注意的地方：

- Triton 能导入时，默认候选是 `("triton", "xla")`；手动指定 `"triton"` 就只试这一项。
- 谁先成功就用谁，没有"先把所有后端测一遍再挑最快的"这一步。
- 只有 `NotImplementedError` 会被接住继续回退；未知实现名、输入校验错误都走各自的异常路径。
- 公共签名的完整参数是 `axis`、`epsilon`、`scale_offset`、`subtract_mean`，没有 `config=`。[完整 API][src-normalization-api]

---

**② Op 基类 `op.py` — 参数绑定、选择配置、记录 metadata、接 VJP**

下面是结构节选；被省略的数组拆分、batch 捕获和 VJP 分支在第十章展开。

```python
# _src/ops/op.py
# sym:Op.__call__（结构节选）

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

`BoundArguments` 把算子和这次调用的参数捆在一起，`default_config` 负责定执行配置。写入 metadata 是为了模型 lower 之后还能找回这个 workload。另外有个短路分支：没有自定义 VJP 且关了 batch capture 时直接调用，不走最后的 `custom_vjp`。[完整调用分支][src-op-call]

---

**③ 后端实现层 `pallas_triton.py` — 形状规范化、切块、启动 kernel**

下面节选 `_fwd` 的分块和启动主体：

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

这里发生的事：

- 输入先被整理成 `(M, K, N)`，K 是归约轴。
- `fuser.Fusion` 包住局部的输入计算；`pull_block_spec` 把输出块的需求反向传到真正的输入上。
- `scale`、`offset` 的块索引不随 program ID 变，每个 program 都用同一份归约轴参数。
- `return_residuals` 决定要不要输出均值和倒标准差；默认的 alias 还受 callable 输入限制。
- `input_output_aliases` 只是调用内部的缓冲区关系，Python 层的数组语义和复制仍归 JAX 管。[前向包装与 alias 条件][src-normalization-triton]

---

**④ Kernel 层 — 对当前数据块完成归一化**

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

`axis_len` 取的是原始完整形状。归约轴被补齐之后，均值和方差依然除以真实的 K，不是补齐后的长度。

LayerNorm 还多一步 padding 修正：无效元素 load 进来是 0，减去均值后就变成 `-mean`，平方求和前必须再清零一次。`subtract_mean=False` 时整个分支走 RMSNorm。[Kernel 计算][src-normalization-kernel]

---

### 第二章：BlockSpec、Grid 与 Shape 规范化

#### 2.1 canonicalize_shape_3d：统一形状约定

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

| 符号 | 含义 | `(2, 3, 17), axis=-1` |
|---|---|---|
| M | 归约轴之前各维度之积 | 6 |
| K | 归约轴长度 | 17 |
| N | 归约轴之后各维度之积 | 1 |

`_fwd` 用的是三维形状 `(6, 17, 1)`；Config 的启发式和缓存键可以用去掉尾部 1 的 `(6, 17)`。两种视角分别给不同层用。[形状规范化][src-normalization-config]

#### 2.2 BlockSpec：声明分块方式

```python
# _src/ops/normalization/pallas_triton.py
# sym:PallasTritonNormalization._fwd

x_spec = pl.BlockSpec(
    (block_m, block_a, block_n),
    lambda i, j: (i, 0, j),
)
```

| 要素 | 含义 | 此处取值 |
|---|---|---|
| `block_shape` | 当前块的形状 | `(block_m, block_a, block_n)` |
| `index_map` | program 坐标 → 块索引 | `lambda i, j: (i, 0, j)` |

普通 blocked indexing 会把块索引乘上对应的块尺寸，所以元素起点是 `(i * block_m, 0, j * block_n)`，归约轴不再切成多个 program。[BlockSpec 约定][jax-blockspec]

#### 2.3 Grid：决定并行度

```python
# _src/ops/normalization/pallas_triton.py
# sym:PallasTritonNormalization._fwd

grid = (
    pl.cdiv(x.shape[0], block_m),
    pl.cdiv(x.shape[2], block_n),
)
```

对 `(M, K, N)=(6,17,1)`、`block_m=4`：

```text
block_shape = (4, 32, 1)
grid        = (2, 1)

program (0, 0)：M 位置 0..3，K 位置 0..31
program (1, 0)：M 位置 4..7，K 位置 0..31
               M 的 6..7、K 的 17..31 由 mask 处理
```

grid 的乘积只是逻辑 invocation 总数。GPU 上同时驻留多少个还受 SM、寄存器和共享内存限制；TPU 得结合顺序 grid、pipeline 和 `dimension_semantics` 才能看懂哪些轴真能并行。[TPU 执行语义][jax-tpu]

#### 2.4 配置如何控制分块

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

| 参数 | 来源 | 作用 |
|---|---|---|
| `block_m` | Config | M 方向块大小 |
| `block_n` | Config；`None` 在执行层变为 1 | N 方向块大小 |
| `block_a` | `next_power_of_2(K)` | 覆盖归约轴 |
| `num_warps` | Config → CompilerParams | Triton 的 warp 参数 |

对齐限制因后端而异：Triton 看操作形状，Mosaic GPU 看访存对齐，TPU block 有自己的规则，常见约束盯着末两维的 8/128 对齐和整维覆盖等特例。真要写 kernel 还得逐个翻具体 Config 和 kernel 的限制。[后端分块规则][jax-blockspec]

> 到这里可以记住一件事：后端把 Config 解释成分块和编译参数，kernel 按这个执行形态计算。Config 从哪来，是 Part 2 的内容。

---

### 第三章：block.pallas_call — Pallas 扩展

Normalization 的 Triton 路径用 `_src/pallas/block.py` 包住原生 Pallas 调用，给 Ref 补上完整形状和边界处理信息。

#### 3.1 BlockRef 概念

```python
# _src/pallas/block.py
# sym:BlockRef（字段节选）

@dataclasses.dataclass(frozen=True, slots=True)
class BlockRef:
    ref: Any
    full_shape: tuple[int, ...]
    spec: pl.BlockSpec
```

kernel 里的 `load()`、`store()` 靠它算出当前块的有效区域，`full_shape` 让归约能用真实长度。`at[...]` 走 `BlockRefIndexer`，把这些信息一路带着。[BlockRef][src-block]

#### 3.2 自动越界检查

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

普通 blocked indexing 看维度是否整除；用了 `pl.Element` 就走对应的检查路径。`inbounds_masks` 再结合块起点和块内索引生成 mask。

有 mask 且调用者没给 `other` 时，`BlockRef.load` 补 0；store 会挡住越界写入。这是 Tokamax 包装器的行为，原生 Pallas 的 padding 值没有这个保证。[边界实现][src-block]、[Pallas padding][jax-blockspec]

#### 3.3 缓存优化

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

这个提示只在 `pl.Element` 索引且调用者没给 eviction policy 时才尝试生成。做法是检查 `index_map` 是不是直接返回了所有 program ID 对象，当数据复用与否的粗略启发式用；实际驻留效果还是得看具体程序和设备。[load 缓存提示][src-block]

#### 3.4 mock.patch 实现

```python
# _src/pallas/block.py
# sym:pallas_call.wrapped_kernel（节选）

with (
    _PL_LOAD_STORE_PATCH_LOCK,
    mock.patch.object(plgpu, "load", wrapped_load),
    mock.patch.object(plgpu, "store", wrapped_store),
):
    return kernel(*in_refs, *out_refs)
```

包装器在 tracing kernel 期间把 Triton 的 load/store 换成接受 BlockRef 的版本，再解包到底层 Ref。锁保护的是共享 Python 属性被修改这件事，并不会把设备上的 program 变成串行。其他后端可以直接用原生 `pl.pallas_call`，或者别的包装器。[调用包装][src-block]

#### 3.5 Grid 调度优化

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

`get_cheapest_grid_pids` 在 M、N 两个方向上试不同的 group size，用估算的活跃块数和数据成本挑一种 program ID 重排。它只改访问顺序，逻辑计算范围不变；这个成本模型选出来的结果，还得拿到真实 workload 上验。[Grid 工具][src-grid]

---

### 第四章：其他算子 Kernel 总览

#### 4.1 Flash Attention (Triton)

Attention 在 Q 块内遍历需要的 KV 区间，维护在线 softmax 统计量和输出累积值：

```text
Q block → KV block → QK logits → 更新 softmax 统计 → PV 累积
                         ↑                         │
                         └──── 遍历剩余 KV ─────────┘
```

当前 Triton Config 为：

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

`block_q/block_k`、head 维切分、`split_k` 和 mask 处理各自对应不同的执行路径，`use_base2` 是 Op 上的字段。Q、K、V 的块映射和前反向统计量都比 Normalization 复杂一档。[Triton Attention][src-attention-triton]

公共输入形状是 `Q: (*B,T,N,H)`、`K: (*B,S,K,H)`、`V: (*B,S,K,h)`。Q/KV 头数和输出 head 维度可以不一样，具体后端还会再检查自己支持哪些组合。[Attention API][src-attention-api]

#### 4.2 Mosaic GPU (SM80/SM90/SM100)

| 算子 | 当前架构分派 | 主要文件 |
|---|---|---|
| Attention | SM90、SM100 | `pallas_mosaic_gpu.py` + 两个架构 kernel |
| GLU | SM80、SM90、SM100 | `pallas_mosaic_gpu.py` + 三个架构 kernel |
| Ragged Dot | 架构、量化与布局共同决定 | `pallas_mosaic_gpu.py` + 多个 kernel 变体 |

SM80 目前还是实验路径，别拿公共 GPU 能力检查的结果去推断每个算子的支持范围。[GPU 工具][src-gpu-utils]、[Attention 分派][src-attention-gpu]、[GLU 分派][src-glu-gpu]

以 SM100 Attention 为例，源码显式安排：

```text
copy_gmem_to_smem → K/V 多阶段 buffer → produced barrier
                                            │
                          QK/PV: tcgen05_mma
                                            │
                          consumed barrier → buffer 复用
```

这条路径上可调的还包括 pipeline 槽位、共享内存用量，以及计算阶段之间怎么分工。[SM100 kernel][src-attention-sm100]

#### 4.3 Mosaic TPU — TPU 后端

TPU 共享 Op 与配置接口，但 kernel 的布局、存储空间和流水线安排由 TPU 后端实现。Attention 前向会调整 Q/K/V 布局并调用 Splash；其 Config 包含：

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

`_get_autotuning_configs` 会枚举 tile、layout 和 scheduler 再过滤不合适的组合，所以 TPU 这边也不是只有手工查表一条路。[TPU Attention][src-attention-tpu]

`custom_buffered_pallas_call` 负责把 scalar-prefetch grid 接到 `pltpu.emit_pipeline`：索引和动态 grid 信息走 SMEM 传递，搬运靠 buffer 配置安排。工具里的代际参数管布局和分块的选择。[TPU 工具][src-mosaic-tpu]

#### 4.4 Ragged Dot 的 Kernel 变体群

| 文件/路径 | 用途 |
|---|---|
| `pallas_triton.py` | Triton 分组矩阵乘 |
| `pallas_mosaic_gpu_kernel_sm80.py` | SM80 路径 |
| `pallas_mosaic_gpu_kernel_sm90.py` / `*_sm90_quant.py` | SM90 普通与量化路径 |
| `pallas_mosaic_gpu_kernel_sm100.py` / `*_sm100_quant.py` | SM100 普通与量化路径 |
| `*_sm100_fp8_quant.py` / `*_sm100_i8_quant.py` | FP8 / INT8 变体 |
| `*_sm100_quant_post_scale.py` | 后缩放量化变体 |
| `pallas_mosaic_tpu.py` / `pallas_mosaic_tpu_kernel.py` | TPU 原有 GMM/TGMM 路径 |
| `pallas_mosaic_tpu_v2.py` + `experimental/gmm_v2/` | TPU v2 路径 |

GPU 靠实际的分派代码选变体。TPU v2 的 `tile_m/tile_k/tile_n` 可以留空，让底层按 VMEM 等约束自己算分块；`implementation="mosaic"` 还是映射到 `mosaic_gpu` 或 `mosaic_tpu`，v2 要走自己的入口。[GPU 分派][src-ragged-gpu]、[TPU v2][src-ragged-tpu-v2]、[公共分派][src-ragged-api]

#### 4.5 GLU 的融合 Kernel

GLU 的数学结构：

```text
gate = x @ W_gate
up   = x @ W_up
out  = activation(gate) * up
```

Mosaic GPU 把矩阵乘之后的激活和乘法接进 kernel 内部，省掉中间张量的物化。不过一旦请求 `return_residuals=True`，`_fwd` 就转去基类路径，训练和纯前向走的代码可能不一样。

一个容易想岔的地方：HBM 流量要看输入、权重、输出、残差和编译结果，不能拿 Python 里写了几个表达式去推显存往返次数。[GLU 实现][src-glu-gpu]

#### 4.6 统一的 Kernel 模式

1. **Op 契约**：校验输入，并连接配置、VJP 与上下文。
2. **Config**：描述此实现的可调执行形态。
3. **分块与布局**：把逻辑输入映射到每次 kernel invocation。
4. **kernel**：计算块内结果，或参与更长的流水线/通信过程。
5. **调优与计时**：测量候选，把结果复用于后续编译。

| 方面 | GPU (Triton) | GPU (Mosaic) | TPU (Mosaic) |
|---|---|---|---|
| 分块约束 | 操作形状与具体 kernel 限制 | 访存对齐、布局与架构限制 | TPU block 规则与算子约束 |
| 常见配置 | block、warp、stage | tile、stage、共享内存与计算分工 | tile、layout、buffer、scheduler |
| 调优来源 | 候选测量 + heuristics | 候选测量 + heuristics | 候选测量、heuristics、部分 LUT 或底层分块决策 |
| grid 含义 | 逻辑 program 空间 | 与具体执行模型配合 | 与顺序、pipeline 和多核语义配合 |

当前源码还扩展了 Ragged Gather/Scatter/Reduce、KDA、MLA、TPU TopK、fused MoE 与 fused causal-conv/Gated Delta Rule 等目录；内部/实验性状态和顶层导出得分开查。[扩展算子目录][src-ops-tree]

---

## Part 2: Autotuning System — 自动调优系统

Part 1 留了个问题：kernel 的 Config 是谁选的？入口是 `BoundArguments.default_config`。先看选择关系，再按 Config、Cache、Benchmark、Heuristics、最终决策链展开。

### Autotuning 优先级总览

```text
op.config 显式指定？
  ├─ 是 → 直接采用
  └─ 否 → 求值 heuristics
            ├─ NullConfig → 无可调参数，直接返回
            └─ 可调 Op → 查询 autotuning cache
                          ├─ 命中非空数据 → fastest_config
                          └─ 未命中 → 按 fallback 选择一个分支
                                       ├─ "autotune"   → 测量候选
                                       ├─ "heuristics" → 采用启发式配置
                                       └─ "error"      → 报错
```

缓存 miss 之后走哪个分支，由 `autotuning_cache_miss_fallback` 决定；调优失败不会在这里自动转去 heuristics。`get_config` 之所以在查缓存前先算 heuristics，是为了认出无可调参数的 `NullConfig` 特例。对可调 Op，只要缓存里有数据，就优先于启发式结果。完整顺序见第九章。[决策源码][src-op-config]

---

### 第五章：Config — 可调参数定义

#### 5.1 pydantic dataclass 模式

Normalization 的配置定义：

```python
# _src/ops/normalization/pallas_triton_config.py
# sym:Config

@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
  block_m: pydantic_lib.PowerOfTwo
  block_n: pydantic_lib.PowerOfTwo | None
  num_warps: pydantic_lib.PowerOfTwo
```

| 修饰符/类型 | 作用 | 在调优中的意义 |
|---|---|---|
| `frozen=True` | 实例不可变 | 可作为候选集合和结果映射的键 |
| `kw_only=True` | 要求关键字构造 | 明确指定哪个字段 |
| `slots=True` | 固定字段存储 | 避免每个实例的动态属性字典 |
| `PowerOfTwo` | 校验二的幂 | 对应本实现的参数约束 |

`block_m` 控制 M 方向，`block_n` 控制 N 方向；末轴归约时 `block_n=None`，执行层将其转为 1。K 方向由 `block_a=next_power_of_2(K)` 覆盖。[Config 定义][src-normalization-config]

#### 5.2 Config 怎么使用

**① 公共 API：由框架选择配置**

普通调用不传 Config。CPU 示例先固定本文源码和验证用的 JAX 版本：

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install 'jax[cpu]==0.11.1' \
  'git+https://github.com/openxla/tokamax.git@f21f67cc921f6ad56785a084ba30dc3d85278e9a'
```

下面是完整 CPU 示例，使用非二的幂的归约长度 17，检查前向、梯度和 vmap：

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

这里选 `xla` 是为了跑参考路径。手上有受支持的 NVIDIA GPU 的话，可以换 `implementation="triton"`，在 GPU 环境里重新做数值对照。[公共调用][src-normalization-api]

**② 源码调试：手动构造后端 Op**

公共 `layer_norm` 没有 `config=` 参数，要手动配就得直接构造后端 Op。下面的完整示例在 CPU 上就能检查配置构造和参数绑定，不会真的执行 GPU kernel：

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

| 字段 | 对执行形态的影响 |
|---|---|
| `block_m=4` | 一个 program 覆盖 4 个 M 位置 |
| `block_n=None` | 此例为末轴归约，执行层的 N 块大小为 1 |
| `num_warps=4` | 传入 Triton compiler 参数 |

显式配置优先于缓存。注意字段合法、设备上可编译、性能合理是三件不同的事。[配置选择][src-op-config]

**③ 性能实验：autotune 后固化 Config**

下面是完整 GPU 示例，需要受支持的 NVIDIA GPU 和匹配的 JAX GPU 后端。它只测两个给定候选，再检查所选配置的输出；本文没在 GPU 上跑过这个示例。

```python
import jax
import jax.numpy as jnp
import numpy as np
import tokamax
from tokamax._src.ops.normalization import pallas_triton
from tokamax._src.ops.normalization import pallas_triton_config

if jax.default_backend() != "gpu":
    raise RuntimeError("此示例需要受支持的 NVIDIA GPU 和 JAX GPU 后端")

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

`fast_op` 固定的只是这次 workload 的前向配置；VJP Op 有自己独立的配置。想扩大搜索范围，就用后端默认的候选集合。另外提醒一句：调优结果可不可靠，还得靠数值检查和稳定的测量环境来兜底。[单 Op 调优][src-op-autotune]

#### 5.3 不同算子的 Config 复杂度对比

| 算子/实现 | 配置内容 | 特点 |
|---|---|---|
| Normalization Triton | `block_m, block_n, num_warps` | 归约轴整体处理 |
| Attention Triton | `block_q, block_k, num_stages, num_warps, block_d, split_k` 等 | 增加循环、head 维分块与 KV 切分 |
| Attention Mosaic GPU | SM90 / SM100 各自的 Config | 具体架构决定候选类型 |
| Attention Mosaic TPU | Q/KV tile、QKV layout、scheduler | 适配 Splash 路径 |
| Ragged Dot TPU v2 | `tile_m, tile_k, tile_n, bucket_base` | 默认可把分块交给底层 |
| 无可调参数的 XLA Op | `NullConfig` | 直接使用参考计算 |

字段同名也不代表实现之间能互换。Config 的语义是由消费它的 `_fwd` 和 kernel 定义的。[Attention Config][src-attention-triton]、[GPU Config 分派][src-attention-gpu]、[TPU Config][src-attention-tpu]、[Ragged Dot v2][src-ragged-tpu-v2]

#### 5.4 Config 的序列化

pydantic 负责字段校验与 JSON 编码。针对第五章的 Config，可以通过 TypeAdapter 序列化，下面是延续前文 `config` 对象的用法片段：

```python
import pydantic

adapter = pydantic.TypeAdapter(type(config))
encoded = adapter.dump_json(config)
decoded = adapter.validate_json(encoded)
assert decoded == config
```

注意配置对象、包内预置缓存和 `AutotuningResult` 文件是三样东西。想把整次调优的设备、workload 和计时结果一起存下来，用第七章的结果序列化入口。[类型适配层][src-pydantic]

---

### 第六章：缓存键与 Autotuning Cache

#### 6.1 缓存键设计

Normalization 把抽象输入规格压成可哈希映射：

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

几个设计点：

1. **只认 shape/dtype**：不拿具体数组数值做键。
2. **开关按类分组**：epsilon 和 scale offset 只区分"是不是零"，这正是本实现眼里的调优等价类。
3. **前向模式算进键**：减不减均值、返不返回 residuals，都会改变工作负载。
4. **漏参数会炸**：末尾的 `assert not kwargs` 就是为了抓没进键的新参数。

缓存还按 Op 和设备种类分开。batch 信息怎么进键，各后端自己定；Normalization 的重写方法里还留着用 batched args 的 TODO。[键函数][src-normalization-config]、[后端重写][src-normalization-triton]

#### 6.2 AutotuningCache：JSON 懒加载

```python
# _src/autotuning/cache.py
# sym:AutotuningCache.__missing__

def __missing__(self, device_kind: DeviceKind) -> DeviceAutotuningCache:
  self[device_kind] = (cache := self._load_cache(device_kind))
  return cache
```

首次访问设备种类时，读取相应 JSON。设备字符串转成小写并用下划线替换空格；文件名取**具体 Op 类名的 snake_case**：

```text
data/autotuning/{device_kind}/{op_class_name}.json

data/autotuning/nvidia_h100_80gb_hbm3/pallas_triton_normalization.json
data/autotuning/nvidia_b200/pallas_mosaic_gpu_ragged_dot.json
```

`PallasTritonNormalization` 不会因为属于 normalization 家族就去读 `normalization.json`。加载之后还会 `op.bind(**k)` 一次，用当前实现重新生成缓存键。多个路径里同键的数据，后加载的覆盖先加载的。[懒加载实现][src-cache]

#### 6.3 全局缓存与覆盖层

当前覆盖层使用 ContextVar，并同时维护 JAX user context：

```python
# _src/ops/op.py
# sym:AUTOTUNING_CACHE_OVERLAY_STACK

_AUTOTUNING_CACHE: dict[Op, AutotuningCache] = {}

AUTOTUNING_CACHE_OVERLAY_STACK = contextvars.ContextVar(
    "AUTOTUNING_CACHE_OVERLAY_STACK", default=()
)
AUTOTUNING_CACHE_OVERLAY_JAX_CONFIG = jax.make_user_context(())
```

查找先从 overlay 栈最内层开始，再访问进程缓存及其懒加载的预置 JSON。进入 `AutotuningResult` 时，JAX context 会带上该结果对象的标识，让作用域变化参与 JIT 缓存区分。[覆盖查询][src-op-cache-lookup]、[作用域管理][src-autotuning-result]

| 操作 | 实际效果 |
|---|---|
| `bound.autotune(cache_results=True)` | 写入进程缓存 |
| `tokamax.autotune(...)` | 内部使用 `cache_results=False`，返回结果对象 |
| `with result:` | 临时启用结果作为 overlay |
| `result.dump(fp)` / `result.dumps()` | 显式持久化/序列化 |

内部的 `_src.config.ignore_autotuning_cache` 会连 overlay 一起跳过缓存查询。这个开关没在公开的 `tokamax.config` 里重新导出。[公开配置入口][src-public-config]

---

### 第七章：Benchmark 驱动的 Autotuning

缓存 miss 且选了 `autotune` 策略时，系统会测量后端给出的候选配置；也可以像第五章那样自己指定候选，再发起实验。

#### 7.1 AutotuningData

`AutotuningData` 保存 `Config → BenchmarkData | Exception`。最优配置由成功结果的中位执行时间决定：

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

| 方法 | 作用 |
|---|---|
| `fastest_config` | 成功候选中，中位时间最小的一项 |
| `prune()` | 只保留最快候选 |
| `prune_errors()` | 删除异常，保留所有成功候选 |

数据为空和候选全失败走不同的异常。另外 `dumps(prune_errors=True)` 只是清掉错误项，不等于只留最快配置。[结果模型][src-autotuner]

#### 7.2 Autotuner：并行编译与同步测量

默认执行器配置：

```python
# _src/autotuning/autotuner.py
# sym:Autotuner（字段节选）

@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Autotuner:
    compile_executor_fn: Callable[[], futures.Executor] | None = (
        futures.ThreadPoolExecutor
    )
    executor_fn: Callable[[], futures.Executor] = _SyncExecutor
```

`_SyncExecutor` 在提交时直接执行函数：

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

两阶段的数据流：

```text
Config 集合
  ├─ ThreadPoolExecutor：构造候选函数 → lower → compile
  │                       └─ 记录编译失败
  └─ 默认 _SyncExecutor：逐个运行 benchmark runner
                          └─ 收集计时或异常
```

编译的并发和设备测量是分开控制的。默认五次计时来自 runner 的 `iterations=5`，别误读成五个候选并发跑。`timeout` 只是 futures 的等待参数，不是能强杀运行中任务的硬期限。[调优执行器][src-autotuner]、[计时 runner][src-benchmark]

#### 7.3 autotune() 顶层函数

顶层入口吃三种输入：

1. **可调用对象**：根据实参 lower 后提取 workload。
2. **非空 BoundArguments 列表/元组**：直接处理提供的规格。
3. **Lowered 对象**：从已有的 StableHLO 里直接提取。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `ignore_cache` | `False` | 默认仅继续调优缓存查询没有返回数据的规格 |
| `all_implementations` | `False` | 通过内部注册表扩展可调实现 |
| `progress_bar` | `True` | 展示调优进度 |
| `timeout` | `600.0` | 传递给调优执行器的等待参数 |
| `max_workers` | `None` | 指定或推导编译线程数 |

默认候选是启发式配置与 `_get_autotuning_configs` 返回集合的并集，搜索范围由后端定义；不扩展候选的后端可能就一个配置可选。[单 Op 候选与调优][src-op-autotune]

`all_implementations` 用的注册表只覆盖一部分算子家族。它只是扩大测量对象，不会改写公共 API 的后端优先级。另外顶层会记录并跳过某些失败任务，拿到结果后最好检查一下完整性。[顶层调优][src-autotuning-api]

#### 7.4 AutotuningResult

结果对象按设备种类组织 workload 和对应的调优数据，并记录 Tokamax 版本。支持上下文覆盖、JSON 编解码和同设备结果合并；同键合并时右边的配置数据优先。[结果对象][src-autotuning-result]

**完整 CPU 示例：提取 → 调优 → 保存 → 加载 → 启用结果**

这里用 XLA Normalization 的单个 `NullConfig` 候选验证工具流程；真正的 Triton 配置性能搜索得上 GPU。

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
    raise RuntimeError("未提取到 Tokamax 算子规格")

result = tokamax.autotune(
    lowered, ignore_cache=True, progress_bar=False, max_workers=2
)
if len(result.data) != len(bound_args):
    raise RuntimeError("至少一个算子的调优任务没有返回结果")
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

注意：

- `dump/load` 吃文件对象，`dumps/loads` 吃字符串。
- 别只看返回了对象，结果数量和每项的 `fastest_config` 都要查。
- 存下来的是调优结果，已编译的二进制不会被这个 JSON 直接改写；结果作用域影响的是后续的 tracing/编译。
- 要调训练，就去提取实际梯度函数里的 Op，把反向 workload 也覆盖进去。

#### 7.5 HLO 内省：get_bound_args

当前版本把 Op 信息写进 XLA metadata，嵌套调用时 payload 外层在前：

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

`Op.__call__` 传入的 JSON 来自抽象化的绑定参数：

```python
# _src/ops/op.py
# sym:Op.__call__.fwd

json_op = self.replace(config=config, vjp=None)
json_ba = BoundArguments(json_op, _abstractify(dict(ba.arguments)))
json_data = str(BOUND_ARGS_ADAPTER.dump_json(json_ba), "utf-8")
with _tokamax_metadata(json_data):
    out, residuals = self._fwd(*args, config=config, **kwargs)
```

读取路径：

```text
Lowered.compiler_ir("stablehlo")
  → 遍历 MLIR module
  → 读取 mhlo.frontend_attributes 中的 xla_metadata_payload
  → get_opspecs 恢复 BoundArguments
  → get_bound_args 移除已选 config，按实现类名与键去重
```

公开用法是 `tokamax.autotuning.get_bound_args(...)`。没有 metadata 时还有旧的 `op_name` 路径兜底。注意候选模型和使用新配置的模型都要各自编译。[StableHLO 提取器][src-hlo]、[辅助提取与去重][src-hlo-common]

#### 7.6 arg_specs：调优样本

Normalization 的样本由 `_make_argspec` 创建，包含输入 shape/dtype、参数 dtype、归约轴和归一化开关。源码里的一个样本：

```python
# _src/ops/normalization/arg_specs.py
# sym:ARG_SPECS（首项节选）

_make_argspec(
    name="alphafold_384res_64chan",
    project="alphafold",
    x_shape=(384, 384, 64),
    dtype="bfloat16",
    param_dtype="float32",
)
```

`ArgSpec` 提供规格模型。用之前最好核对一下样本有没有覆盖真实的 shape、dtype、mask、分组方式和前反向模式。[样本来源][src-normalization-specs]、[ArgSpec][src-arg-spec]

---

### 第八章：Heuristics — 启发式默认配置

#### 8.1 逐行解析

Normalization 现在的完整启发式函数：

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

按代码顺序拆成四步：

1. **初始块大小**：末轴归约从 `block_m=32` 开始，其他情况从 1 开始；没有 scale/offset 时也从 1 开始。
2. **估计工作规模**：算块元素数和块数，乘进 vmap 大小；三维形状还要考虑 N 方向和 cache-line 元素数。
3. **折半块大小**：块太大或总块数不足时反复折半 `block_m`，在复用和可调度工作量之间折中。
4. **选 warp 数**：按每 warp 默认 1024 个元素估计，最多 4 个 warp。

拿 `M=4096, K=1024, N=1`、带 scale 的输入算一遍，`core_count` 取 80：

| 步骤 | block_m | 块元素数 | 总块数 | 判断 |
|---|---:|---:|---:|---|
| 初始 | 32 | 32768 | 128 | 块太大，且少于目标 320 |
| 折半一次 | 16 | 16384 | 256 | 块数仍不足 |
| 再折半 | 8 | 8192 | 512 | 满足两个估算条件 |

最终 `num_warps=min(ceil(8192/1024),4)=4`。这整段只是规则演算，寄存器常量和阈值不能等同于编译器的实际寄存器分配或设备上的真实计时。[启发式实现][src-normalization-config]

#### 8.2 Heuristics 的定位

Heuristics 有两种作用：

- cache miss 且选择 `heuristics` 时，直接给出配置。
- 执行 autotune 时，作为候选集合的一项起点。

这些规则保证配置"合理"，最快配置还是得靠候选测量。前反向会复用一部分规则再改参数：Normalization VJP 复用前向函数，只是把 `block_size_per_warp` 改成 2048。[反向启发式][src-vjp-config]

TPU Attention 提供默认 Splash 配置并展开候选；Ragged Dot TPU v2 可以把 tile 决策交给底层。策略由每个后端自己定。[TPU Attention][src-attention-tpu]、[TPU v2][src-ragged-tpu-v2]

---

### 第九章：default_config 决策链

#### 9.1 配置优先级与 NullConfig 特例

`default_config` 把 miss 策略翻译成 `get_config` 的参数。方法核心代码：

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

**优先级和计算顺序要放在一起看**：

- 显式 Config 最先返回。
- heuristics 在查缓存前先算，是为了认出 `_NULL_CONFIG` 单例；没有可调参数的 Op 到这就结束了。
- 可调 Op 只要缓存非空，就用缓存，不用启发式结果。
- miss 之后按设置走搜索、启发式或报错；搜索失败不会在这里自动回头试 heuristics。[决策实现][src-op-config]

#### 9.2 get_config() 的精细化控制

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

| 参数 | 控制内容 |
|---|---|
| `check_autotuning_cache` | 本次是否查询缓存；显式 config 仍优先 |
| `autotune_configs=None` | 不运行搜索 |
| `autotune_configs=AUTO` | 使用启发式与后端默认候选的并集 |
| `autotune_configs={...}` | 使用给定候选集合 |
| `cache_autotuning_results` | 是否将搜索结果写入进程缓存 |
| `allow_heuristics` | 是否允许最终采用启发式结果；不禁止前面的求值 |

`AUTO` 是 `ops.op` 里的标记类型，和 `None` 不是一回事。

#### 9.3 全局配置

公开选项 `tokamax.config.autotuning_cache_miss_fallback`：

| 值 | cache miss 后的行为 |
|---|---|
| `"heuristics"`（默认） | 采用启发式配置 |
| `"autotune"` | 测量候选并采用最快成功项 |
| `"error"` | 没有配置时抛错 |

完整 CPU 示例：

```python
import jax.numpy as jnp
import tokamax

x = jnp.arange(24, dtype=jnp.float32).reshape(3, 8)
with tokamax.config.autotuning_cache_miss_fallback("error"):
    y = tokamax.layer_norm(x, scale=None, offset=None, implementation="xla")
print(y.shape)
```

它照样能跑，因为 XLA Op 先走了 `NullConfig` 快速返回。想验证 Triton 的缓存覆盖，得选实际可调的后端，拿有代表性的输入触发 tracing。[配置入口][src-public-config]

#### 9.4 全流程序列

```text
用户调用 op(x)
  │
  ├─ bind、形状和设备检查
  ├─ 构造数组/batch/VJP 包装
  └─ 执行内部 fwd
       ├─ 重组并绑定参数，纳入 batch 信息
       ├─ config = ba.default_config
       │    ├─ 显式 config？          → 直接采用
       │    ├─ 求 heuristics，Null？   → 无可调参数，直接返回
       │    ├─ 非空 cache 命中？       → fastest_config
       │    └─ miss 策略
       │         ├─ autotune          → 测候选
       │         ├─ heuristics        → 采用已有启发式结果
       │         └─ error             → 报错
       ├─ 写入抽象参数和配置的 metadata
       ├─ self._fwd(..., config=config)
       └─ 返回输出，以及调用模式要求的残差
```

到这里，kernel 怎么消费配置、配置怎么产生，这条调用路径算是串起来了。

---

## Part 3: Infrastructure — 基础设施

最后看把 kernel 和调优系统串起来的 Op 基类，以及自动微分、计时、批处理、精度和序列化这些工具。

---

### 第十章：Op 基类系统

#### 10.1 五个泛型参数

```python
# _src/ops/op.py
# sym:Op（签名与字段节选）

@dataclasses.dataclass(frozen=True)
class Op[**P, T, R, C, K: Hashable](abc.ABC):
    config_cls: ClassVar[type[Any]] = NullConfig
    supports_symbolic_shapes: ClassVar[bool] = True
    supports_batched_args_capture: ClassVar[bool] = True
    config: C | None = None
```

| 槽位 | 名称 | 含义 | Normalization 前向示例 |
|---|---|---|---|
| 1 | P | 输入参数签名 | x、scale、offset 与关键字参数 |
| 2 | T | 输出类型 | `jax.Array` |
| 3 | R | 算子残差类型 | `(mean, rstddev)` |
| 4 | C | 配置类型 | Triton Normalization Config |
| 5 | K | 可哈希的调优键类型 | `immutabledict` |

这是 Python 3.12 的类型参数语法。`Op.replace` 创建新实例，对 `vjp=None` 有特殊处理：后端的 `__post_init__` 可能自动补默认 VJP，replace 得保证调用者"明确不要 VJP"这件事最终生效。[Op 与 replace][src-op-call]

#### 10.2 `__call__` 全生命周期（8 步）

```text
op(*args, **kwargs)
  │
  ├─ 1. 符号形状检查
  │     supports_symbolic_shapes / contains_symbolic_shape
  │
  ├─ 2. 参数绑定
  │     self.bind → 校验、补默认值 → BoundArguments
  │
  ├─ 3. 设备检查
  │     infer_devices / supported_on
  │     bypass_device_check 或 cross_compile 可影响此步骤
  │
  ├─ 4. 参数拆分
  │     flatten → arrays / other / merge
  │
  ├─ 5. 定义带 batch 捕获的 fwd
  │
  ├─ 6. 定义内部前向逻辑
  │     重组参数 → 重新 bind → default_config
  │     → metadata → self._fwd(config=...)
  │
  ├─ 7. 按条件设置 VJP
  │     手写 VJP，或用于 batch capture 兼容的自动 VJP
  │     无 VJP 且关闭 batch capture 时可直接调用
  │
  └─ 8. return f(*arrays)
        真正进入前面构造的包装函数
```

第 5–7 步在搭包装，最后一步才真正调用。如果只读 `_fwd`，会漏掉输入校验、batch 信息、配置选择和自动微分适配这些环节。[调用生命周期][src-op-call]

#### 10.3 BoundArguments

```python
# _src/ops/op.py
# sym:BoundArguments（字段与初始化节选）

@dataclasses.dataclass(frozen=True, slots=True)
class BoundArguments[C, K: Hashable]:
    op: Op[..., Any, Any, C, K]
    arguments: Mapping[str, Any]

    def __post_init__(self):
        immutable_args = immutabledict.immutabledict(self.arguments)
        object.__setattr__(self, "arguments", immutable_args)
```

它把一个 workload 变成可以直接操作的对象：

- `args` / `kwargs`：恢复调用参数。
- `default_config` / `heuristics_config`：取得配置。
- `autotuning_cache_key` / `cached_autotuning_data`：查询缓存。
- `autotuning_configs` / `autotune()`：生成并测量候选。
- `benchmark()` / `vjp_arg_spec`：计时和构造反向规格。

手动 `bind` 出来的和从 StableHLO 提取出来的，最后都汇合到这些方法上。[BoundArguments][src-op-config]、[测量与调优入口][src-op-autotune]

#### 10.4 Residuals 系统

```python
# _src/ops/op.py
# sym:Residuals（字段节选）

@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, slots=True)
class Residuals[T, R]:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    out: T
    residuals: R
    ...
```

框架残差封装的是参数、输出和算子残差，靠自定义 PyTree flatten/unflatten 在前反向之间传递。算子自己的 R 才是数学上真正需要的中间信息。

| 算子/路径 | 残差内容 | 注意点 |
|---|---|---|
| LayerNorm | mean、rstddev | 真实归约统计 |
| RMSNorm | None、rstddev | 不保存均值 |
| TPU Attention / Splash | max_logits、logsumexp | 按该后端前向与匹配 VJP 的约定使用 |
| Ragged Dot | 可能需要激活前的 dot 输出 | 随激活和残差模式变化 |

tuple 结构一样也不代表不同后端的残差能互换。[框架残差][src-op-call]、[TPU Attention 残差][src-attention-tpu]

#### 10.5 VJP 机制

没有手写 VJP 时，是否包装取决于 batch capture：

```python
# _src/ops/op.py
# sym:Op.__call__（VJP 分支节选）

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

手写反向分支解包 Residuals，调 `self.vjp`，再按输入结构整理梯度。梯度以字典返回时，没指定的相关数组参数会补零。

没有手写 VJP、又需要 custom batching 捕获上下文时，内部用 `jax.vjp` 构造梯度函数，再通过 `custom_vjp` 接上。具体 kernel 对微分和各种变换组合支持到什么程度，由实现自己说了算。[完整 VJP 分支][src-op-call]

#### 10.6 base.py 的三重身份

Normalization 基类同时提供：

1. **XLA 后端**：`IMPLEMENTATIONS['xla'] = base.Normalization()`。
2. **算子基类**：供 Triton 等后端继承参数契约。
3. **参考实现**：用纯 JAX 计算为新后端提供数值对照。

基类 `_fwd` 先把数据升到至少 FP32，做归约和可选的 scale/offset，最后转回输入 dtype；请求 residuals 时再额外返回统计量。[基类实现][src-normalization-base]

---

### 第十一章：Benchmarking 系统

`_src/benchmarking.py` 提供函数标准化、编译和计时这些通用能力，autotuning 在它上面组织候选搜索。

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

`compile_benchmark` 给 `.lower(x)` 和 `.compile()` 分别计时，返回的 runner 再记录执行时间。`median_evaluation_time_ms` 由执行序列算出来；峰值内存来自编译程序的 `memory_analysis()`，是编译器报告的估计值。[数据与编译入口][src-benchmark]

#### 11.2 standardize_function

函数和参数被整理成 `(fn, array_args)`，`fn(array_args)` 只收数组集合，其他对象留在闭包里。抽象输入可以随机初始化，batch 规格能恢复成对应的 vmap。

| mode | 当前构造方式 | 测量重点 |
|---|---|---|
| `forward` | 普通前向函数 | 推理/纯前向 |
| `forward_res` | `jax.vjp(forward, arrays)[0]` | 进入 AD 前向路径，不等于返回所有残差 |
| `vjp` | 预先取得 VJP，再把 cotangent 作为输入 | 反向；闭包中间量影响编译与内存 |
| `forward_and_vjp` | 前向与 VJP 放在一个函数中，并加 optimization barrier | 联合前反向 |

不同 mode 可能编译出不同的程序，尤其后端会根据 `return_residuals` 改融合路径的时候。[函数标准化][src-benchmark]

#### 11.3 四种计时方法

| 方法 | 源码中的计时路径 | 适用范围 |
|---|---|---|
| `cupti` | JAX `profiler.Cupti` | GPU 设备执行，包含 profiler 自身影响 |
| `hermetic_xprof` | JAX profiler → 本地 profile → XProf 解析 | GPU/TPU，TPU 默认 |
| `xprof` | XProf session | 依赖相应 session 组件，开源环境未必具备 |
| `wallclock` | `perf_counter` + `block_until_ready` | 通用，包含主机调度与等待 |

默认按平台选：GPU → CUPTI，TPU → hermetic XProf，其他 → wallclock。profiler 抛异常之后，源码不会自动退回 wallclock 重试。[计时器与默认选择][src-benchmark]

**完整 CPU 示例**：

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

这段代码只验证接口和返回值，别把本机 wallclock 的数字当成 GPU/TPU 的性能结论。真要横向比较，shape、dtype、mode、设备和软件版本都得固定住。

**XProf 活动时间的读取**：

```python
# _src/benchmarking.py
# sym:XprofProfileSession._compute_timing_summary / total_op_time（节选）

self._timing_summary = get_kernel_stats_tool.compute_kernel_stats(
    self._profile,
    output_format="dict",
    include_summary=True,
    trace_matchers=trace_matchers,
)
total_us = self._timing_summary.get("total_device_duration_us", 0.0)
```

当前路径按活动区间统计，重叠事件不会被简单累加成重复计时。它和端到端延迟是两回事；用了事件过滤的话，还要说明白到底选中了哪些事件。[XProf 集成][src-benchmark]

---

### 第十二章：Batching / vmap 工具

#### 12.1 BatchedShapeDtype

```python
# _src/batching.py
# sym:BatchedShapeDtype（字段节选）

class BatchedShapeDtype(jax.ShapeDtypeStruct):
    __slots__ = ("vmap_axes",)
    vmap_axes: tuple[tuple[int, int] | None, ...]
    ...
```

普通 shape 描述局部输入，`vmap_axes` 额外记下每层映射轴的位置和大小，`vmap_shape` 能恢复插入 batch 轴之后的形状。启发式就是靠它估计整个批量调用的工作量。[Batching 类型][src-batching]

#### 12.2 capture_batched_args

捕获规则里的关键连接：

```python
# _src/batching.py
# sym:capture_batched_args（连接节选）

new_vmap_axis = None if axis is None else (axis, axis_size)
new_vmap_axes = (new_vmap_axis, *shape.vmap_axes)
new_shape = BatchedShapeDtype(shape.shape, shape.dtype, new_vmap_axes)

fn_vmap = jax.custom_batching.custom_vmap(fn_flat)
fn_vmap.def_vmap(vmap_rule)
```

捕获的规格经 BoundArguments 传给后端；Normalization 启发式里 `num_blocks` 乘的那个 `product(vmap_axis_sizes)` 就是从这条路径来的。缓存键用不用全部 batch 信息，得看后端的键函数。[捕获规则][src-batching]

#### 12.3 vmap_maybe_bcast 与 vmap_split

| 工具 | 行为 | 约束 |
|---|---|---|
| `vmap_maybe_bcast` | 大小为 1 的相关输入轴先 squeeze；全部广播时调用一次后恢复输出轴 | 按广播关系适配 |
| `vmap_split` | 按 `num_parts` 拆分轴，再使用 vmap/broadcast 逻辑 | 非广播轴需要满足整除条件 |

`vmap_split` 的拆分方式由调用者定，不会自动找一个满足内存限制的 batch size。实际调度和并行效果还是得看后端生成出来的程序。[vmap 工具][src-batching]

---

### 第十三章：精度与量化

#### 13.1 精度系统

```python
# _src/precision.py
# sym:DotAlgorithm / DotAlgorithmPreset / Precision

DotAlgorithm = jax.lax.DotAlgorithm
DotAlgorithmPreset = jax.lax.DotAlgorithmPreset
Precision = jax.lax.Precision
```

`F16_F16_F32`、`BF16_BF16_F32`、`TF32_TF32_F32` 这些是 JAX 的点积算法预设。Tokamax 把 `PrecisionLike` 规范化，再结合输入 dtype 和平台映射到后端支持的表达。[精度适配][src-precision]

Attention 给 QK 和 PV 两次点积分别带精度信息，指的是两次运算各自的精度选择，不是 FP64 意义上的"双精度"。[Attention 契约][src-attention-base]

#### 13.2 量化支持

Qwix 的 `QArray` 携带量化数据和 scale 等信息，`AsQArray` 保留延迟量化的意图：

```python
# _src/quantization.py
# sym:AsQArray（字段与转换节选）

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

| 转换 | 结果 |
|---|---|
| `as_array_or_qarray` | 必要时把 AsQArray 量化成 QArray |
| `as_array` | 必要时进一步反量化成普通数组 |
| `as_array_or_qarray_without_zero_point` | 不适合相应 zero-point 路径时反量化 |

这种表示给 kernel 内融合量化留了机会，中间张量物不物化由后端路径决定。Ragged Dot 的量化位宽、scale tile、布局、前反向方向，都会影响最终选哪个实现。[量化适配][src-quantization]、[GPU 量化路径][src-ragged-gpu]、[TPU 量化路径][src-ragged-tpu]

---

### 第十四章：配置系统与工具库

#### 14.1 全局配置系统

`_ConfigOption` 把 `absl.flags` 和 JAX user context 绑在一起：

```python
# _src/config.py
# sym:_ConfigOption（初始化节选）

def __post_init__(self):
    if self.config is None:
        object.__setattr__(self, "config", jax.make_user_context(_DEFAULT))
```

作用域覆盖走 `self.config(value)`，读取时优先拿当前 context，没有覆盖才取 flag。首次解析用 `known_only=True`，这样库自己的配置能和宿主 CLI 的其他参数共存。[配置实现][src-config]

| 选项 | 入口范围 |
|---|---|
| `autotuning_cache_miss_fallback` | 从 `tokamax.config` 公开导出 |
| `cross_compile` | 从 `tokamax.config` 公开导出 |
| `ignore_autotuning_cache` | `_src.config` 内部选项 |
| `disable_multi_core_mode` | `_src.config` 中用于部分 GMM/TGMM v2 路径的临时选项 |

[公开配置导出][src-public-config]

#### 14.2 HLO 工具

| 入口/结构 | 职责 |
|---|---|
| `get_kernel_info` | 从 StableHLO 提取相关 kernel 信息 |
| `get_opspecs` | 从 metadata 或旧名称路径恢复 BoundArguments |
| `get_bound_args` | 去掉已选配置，提取唯一 workload |
| `TritonKernelInfo` / `MosaicGpuKernelInfo` / `MosaicTpuKernelInfo` | 描述不同 kernel 路径 |
| `dedupe_wrapper_kernels` | 对带 payload 的转换、广播等 wrapper 做特定去重 |
| `DISABLE_JAX_EXPORT_CHECKS` | 允许列出的特定 Triton custom calls 进入导出 |

内省拿到的记录数、唯一 workload 数、最终设备 launch 数是三个不同的指标。带 custom kernel 的 StableHLO 导出之后，设备和 ABI 的约束也不会消失。[HLO 工具][src-hlo]、[信息结构与去重][src-hlo-common]

#### 14.3 形状与数值工具

| 函数/类型 | 位置 | 作用 |
|---|---|---|
| `exact_div` | `utils.py` | 整除检查 |
| `split_merge` | `utils.py` | 按谓词拆分数组/其他对象，并提供重组函数 |
| `canonicalize_shape_3d` | Normalization Config 模块 | 输入维度映射到 M/K/N |
| `pad_to_next_multiple_of` | `shape.py` | 按给定倍数 padding |
| `contains_symbolic_shape` | `shape.py` | 检查符号形状 |
| `random_initialize` | `numerics.py` | 为抽象输入生成数据 |
| `DiffSummary` / `array_diff_summary` | `numerics.py` | 数值差异摘要 |

[形状工具][src-shape]、[通用工具][src-utils]、[数值工具][src-numerics]

#### 14.4 设备能力检测

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

通用 Mosaic GPU 检查已经放行实验性 SM80，但具体算子还会加自己的限制。Cross-compilation 能影响设备检查，代替不了真实设备上的执行。[GPU 能力检查][src-gpu-utils]

这个提交还补了设备种类推断：没有具体输入设备时，试着读活动 abstract mesh 里的 `abstract_device.device_kind`，用于目标设备的缓存查询。至于具体 kernel 能不能脱离目标硬件编译，还得逐个核对。[设备推断][src-op-device]、[对应提交][src-revision]

#### 14.5 JAX 类型序列化

`pydantic.py` 为这些对象提供适配：

- **ShapeDtype**：shape、dtype 与相关 JAX 规格。
- **NumPy dtype 编解码**：把 dtype 转成可序列化表示并恢复。
- **PowerOfTwo**：二的幂校验。
- **AnyInstanceOf**：携带具体类信息的实例编码。
- **参数规格模型**：根据 Op 的函数签名创建验证/序列化模型。

它把 Config、BoundArguments、JSON 缓存和 metadata 串起来，保证读回来的数据能重新进算子契约。[序列化适配][src-pydantic]

#### 14.6 ad.py — VJP 构建工具

```python
# _src/ad.py
# sym:get_vjp_taking_residuals

def get_vjp_taking_residuals[T, R](
    fn: Callable[..., tuple[T, R]], *primals
) -> Callable[[R, T], Any] | None:
  ...
```

函数会分析并重写 jaxpr，尝试构造显式接收用户残差的 VJP。只有选出来的残差够算梯度时才会成功，返回 `None` 表示按这组残差构造不出来。[AD 工具][src-ad]

#### 14.7 其他小工具

- `jaxtyping.py`：类型检查适配与相应开关。
- `mosaic_gpu.py`：GPU 布局、量化编码、同步和资源估算辅助。
- `mosaic_tpu.py`：TPU 量化 BlockSpec、代际参数与 buffered pipeline。
- `version.py`：版本常量和发布时写入的 Git 构建标识；源码 checkout 里后者可能是空的。

[GPU 工具][src-mosaic-gpu]、[TPU 工具][src-mosaic-tpu]、[版本定义][src-version]

---

### 第十五章：关键设计模式总结

#### 15.1 策略模式 + 回退链

API 注册实现并挨个尝试，具体 Op 再选自己的 Config。后端选择和配置选择是两层，分析性能的时候要分开看。[API 分派][src-normalization-api]、[配置决策][src-op-config]

#### 15.2 Op 作为可哈希配置载体

Frozen dataclass 把 Op 变成不可变配置对象，`replace` 产出新实例。配置、参数规格和缓存都围着这个对象转。[Op 基类][src-op-call]

#### 15.3 HLO 内省式离线 autotuning

抽象参数写进 metadata，再从 StableHLO 提取独立 workload。调优工具因此能直接复用模型里的调用规格，不用手工重写每一层；候选编译和换配置后的模型编译还是省不掉。[内省入口][src-hlo]

#### 15.4 离线结果与作用域覆盖

预置 JSON、进程缓存、AutotuningResult overlay 各管一摊：预置数据、进程内复用、临时覆盖。显式保存的结果能跨会话复用；存的时候最好连源码 SHA、设备和软件环境一起记下来。[缓存][src-cache]、[结果作用域][src-autotuning-result]

#### 15.5 custom_vjp 兼容 custom_vmap

需要 custom batching 捕获上下文时，Op 用对应的 VJP 包装处理组合关系。没有 VJP 且关了 batch capture 时有直接调用分支；具体后端的支持程度还是要靠测试说话。[自动微分分支][src-op-call]

#### 15.6 上下文与共享状态管理

全局配置用 JAX user context，overlay 和嵌套 metadata 用 ContextVar，Pallas load/store patch 用锁。三者各管一件事：编译缓存区分、作用域恢复、共享 Python 属性修改。[配置][src-config]、[Op 上下文][src-op-call]、[Pallas 包装][src-block]

#### 15.7 可选导入与后端回退

模块通过可选导入注册可用的后端，再结合设备和输入检查决定调用路径。能导入不代表当前设备/shape 一定支持；XLA 参考实现提供了重要的可移植路径。[后端注册][src-normalization-api]

#### 15.8 参考实现 = 基类

纯 JAX 基类一肩三职：参数契约、参考计算、XLA 后端。写新 kernel 时可以先用它的契约和数值对照，再逐步补分块、量化、VJP 和性能策略。[Normalization 基类][src-normalization-base]

---

## 附录

### A. 主要文件映射

下表汇总正文涉及的主要文件；源码树里的测试和实验文件可以从对应目录继续展开。

<table>
<tr><th>文件</th><th>用途</th></tr>
<tr><td colspan="2"><b>包顶层 — 公共入口</b></td></tr>
<tr><td><code>__init__.py</code></td><td>计算 API、Op、autotune、benchmark 导出</td></tr>
<tr><td><code>autotuning.py</code></td><td>get_bound_args 与 JSON 规格工具</td></tr>
<tr><td><code>benchmarking.py</code></td><td>计时辅助入口</td></tr>
<tr><td><code>config.py</code></td><td>两个公开配置选项</td></tr>
<tr><td colspan="2"><b>_src/ops/ — 算子实现</b></td></tr>
<tr><td><code>op.py</code></td><td>Op、BoundArguments、配置、metadata、VJP</td></tr>
<tr><td><code>normalization/api.py</code></td><td>LayerNorm / RMSNorm 公共调度</td></tr>
<tr><td><code>normalization/base.py</code></td><td>输入契约与参考实现</td></tr>
<tr><td><code>normalization/pallas_triton.py</code></td><td>前向包装与 kernel</td></tr>
<tr><td><code>normalization/pallas_triton_config.py</code></td><td>Config、key、heuristics</td></tr>
<tr><td><code>normalization/pallas_triton_vjp.py</code></td><td>Triton 反向</td></tr>
<tr><td><code>normalization/pallas_triton_vjp_config.py</code></td><td>反向配置与键</td></tr>
<tr><td><code>normalization/arg_specs.py</code></td><td>典型 workload</td></tr>
<tr><td><code>attention/api.py</code></td><td>Attention 调度与公共形状契约</td></tr>
<tr><td><code>attention/base.py</code></td><td>参考实现、Mask、PagingInfo 等契约</td></tr>
<tr><td><code>attention/pallas_triton.py</code></td><td>Triton Config 与 kernel</td></tr>
<tr><td><code>attention/pallas_mosaic_gpu.py</code></td><td>Mosaic GPU 架构分派</td></tr>
<tr><td><code>attention/pallas_mosaic_gpu_kernel_sm90.py</code></td><td>SM90 Attention</td></tr>
<tr><td><code>attention/pallas_mosaic_gpu_kernel_sm100.py</code></td><td>SM100 Attention</td></tr>
<tr><td><code>attention/pallas_mosaic_tpu.py</code></td><td>TPU Config、Splash 前向适配</td></tr>
<tr><td><code>attention/xla_chunked.py</code></td><td>分块 XLA 路径</td></tr>
<tr><td><code>attention/jax_nn.py</code></td><td>JAX / cuDNN Attention 适配</td></tr>
<tr><td><code>ragged_dot/api.py</code></td><td>Ragged Dot / General 入口</td></tr>
<tr><td><code>ragged_dot/pallas_mosaic_gpu.py</code></td><td>GPU 分组矩阵乘与量化分派</td></tr>
<tr><td><code>ragged_dot/pallas_mosaic_tpu.py</code></td><td>TPU GMM/TGMM 适配</td></tr>
<tr><td><code>ragged_dot/pallas_mosaic_tpu_v2.py</code></td><td>TPU v2 Op 与底层分块连接</td></tr>
<tr><td><code>gated_linear_unit/api.py</code></td><td>GLU 公共入口</td></tr>
<tr><td><code>gated_linear_unit/pallas_mosaic_gpu.py</code></td><td>GLU 架构与残差路径分派</td></tr>
<tr><td><code>linear_softmax_cross_entropy_loss/api.py</code></td><td>Linear CE 的 XLA、chunked XLA 与 TPU 入口</td></tr>
<tr><td><code>triangle_multiplication/api.py</code></td><td>Triangle Multiplication 的 XLA 路径</td></tr>
<tr><td><code>ragged_gather/api.py</code>、<code>ragged_scatter/api.py</code>、<code>ragged_gather_reduce/api.py</code></td><td>内部 ragged 数据移动入口</td></tr>
<tr><td><code>experimental/kda/api.py</code>、<code>experimental/mla/api.py</code>、<code>experimental/tpu/topk/api.py</code></td><td>实验性计算入口</td></tr>
<tr><td colspan="2"><b>_src/autotuning/ — 自动调优</b></td></tr>
<tr><td><code>api.py</code></td><td>autotune、AutotuningResult、实现扩展</td></tr>
<tr><td><code>autotuner.py</code></td><td>编译执行器、测量、AutotuningData</td></tr>
<tr><td><code>cache.py</code></td><td>JSON 懒加载与 key 重建</td></tr>
<tr><td><code>arg_spec.py</code></td><td>样本规格模型</td></tr>
<tr><td colspan="2"><b>_src/pallas/ — Pallas 扩展</b></td></tr>
<tr><td><code>block.py</code></td><td>BlockRef、mask、pallas_call 包装</td></tr>
<tr><td><code>grid.py</code></td><td>分组成本与 program ID 重排</td></tr>
<tr><td colspan="2"><b>_src/ — 基础设施</b></td></tr>
<tr><td><code>benchmarking.py</code></td><td>函数标准化、编译、计时与 profile</td></tr>
<tr><td><code>hlo_utils.py</code> / <code>hlo_utils_common.py</code></td><td>StableHLO 内省、kernel 信息和去重</td></tr>
<tr><td><code>batching.py</code></td><td>batch 规格和 custom_vmap 捕获</td></tr>
<tr><td><code>precision.py</code></td><td>精度表达规范化</td></tr>
<tr><td><code>quantization.py</code></td><td>QArray / AsQArray 转换</td></tr>
<tr><td><code>config.py</code></td><td>absl.flags 与 JAX context</td></tr>
<tr><td><code>shape.py</code> / <code>utils.py</code></td><td>形状处理、整除、拆分和重组</td></tr>
<tr><td><code>ad.py</code></td><td>从指定残差构造 VJP</td></tr>
<tr><td><code>gpu_utils.py</code></td><td>GPU 能力检查</td></tr>
<tr><td><code>pydantic.py</code></td><td>JAX 类型与 Op 编解码</td></tr>
<tr><td><code>numerics.py</code></td><td>样本初始化与数值比较</td></tr>
<tr><td><code>jaxtyping.py</code></td><td>类型检查适配</td></tr>
<tr><td><code>mosaic_gpu.py</code> / <code>mosaic_tpu.py</code></td><td>后端工具</td></tr>
<tr><td><code>version.py</code></td><td>版本与构建标识</td></tr>
</table>

### B. 后端支持矩阵

表示当前实现路径；各后端在 dtype、shape、量化、mask、残差和微分上仍有自己的限制。没有专用 kernel 时，仍可能走 XLA 计算。

| 后端 | Attention | RaggedDot | GLU | LayerNorm | TriangleMul | LinearCE |
|---|---|---|---|---|---|---|
| XLA | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| cuDNN | ✅ | — | — | — | — | — |
| Triton | ✅ | ✅ | ✅ | ✅ | — | — |
| Mosaic GPU | ✅ | ✅ | ✅ | — | — | — |
| Mosaic TPU | ✅ | ✅ | — | — | — | ✅ |
| Mosaic TPU v2 | — | ✅ | — | — | — | — |

实现名称按 API 区分：Attention 的分块 XLA 叫 `xla_chunked`，Linear CE 叫 `chunked_xla`，Ragged Dot 的 v2 叫 `mosaic_tpu_v2`。表里没把内部或实验性算子算作顶层 API。[Attention][src-attention-api]、[Ragged Dot][src-ragged-api]、[GLU][src-glu-api]、[Normalization][src-normalization-api]、[Linear CE][src-linear-api]、[Triangle][src-triangle-api]

### C. 数据流全景图

```text
                  用户代码：layer_norm(x, scale, offset)
                                  │
                                  ▼
                        api.py：按优先级选后端
                                  │
                                  ▼
                    Op.__call__：bind、检查、构造包装
                                  │
                                  ▼
                    内部 fwd：确定这次 workload 的 Config
                         ├─ 显式配置
                         ├─ NullConfig 快速返回
                         ├─ cache 命中 → fastest_config
                         └─ miss → autotune / heuristics / error
                                  │
                         ┌────────┴─────────┐
                         ▼                  ▼
                  记录 metadata       后端 _fwd
                         │            shape → Fusion
                         │            → BlockSpec / grid
                         │            → pallas_call
                         │                  │
                         │                  ▼
                         │        load → 块内计算 → store
                         │
                         ▼
                 lower 后的 StableHLO
                         │
                         ▼
                  get_bound_args 提取规格
                         │
                         ▼
                 编译候选 → 逐候选 benchmark
                         │
                         ▼
                   AutotuningResult
                         │
                         └─ with result → 后续编译复用所选配置
```

想追源码演进的话，可以对照 [0.0.12 → 本文 main][compare-old] 和 [0.0.13 发布快照 → 本文 main][compare-release]。主要变化包括 metadata 与上下文机制、实验性 SM80 路径、TPU v2 与新增算子目录，以及当前 main 的目标设备缓存查询；SHA 固定在这里，方便以后复核。

文中源码节选来自 Apache-2.0 许可的官方公开仓库，完整示例用于说明调用和验证方式。[上游许可证][src-license]

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

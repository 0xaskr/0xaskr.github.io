---
title: 'MLIR StableHLO 与 Mosaic IR 如何连接模型和芯片'
description: '从一行矩阵计算出发，理解中间表示、MLIR 方言、StableHLO 的可移植性，以及 Mosaic 如何描述面向 TPU 和 GPU 的内核实现。附编译路径图解和可在 CPU 上运行的 JAX 示例。'
pubDate: '2026/10/10'
tags: ["MLIR", "StableHLO", "Mosaic", "JAX", "Compiler"]
draft: true
---

写机器学习代码时，我们经常只需要表达一件事：把输入乘上权重，加上偏置，再经过激活函数。

```python
y = jnp.maximum(x @ w + b, 0.0)
```

但芯片还需要知道更多细节：矩阵要切成多大的块，数据放在哪里，什么时候搬运，由哪些计算单元处理。编译器的工作，就是把前面的计算描述，逐步变成后面的执行安排。

MLIR、StableHLO 和 Mosaic 经常出现在这段过程里。理解它们，可以先抓住各自解决的问题：

| 名称 | 它是什么 | 主要解决的问题 |
| --- | --- | --- |
| MLIR | 支持多层次中间表示的编译器基础设施 | 怎样表达、分析和转换不同抽象层次的程序 |
| StableHLO | 基于 MLIR 的高层张量运算集及其规范 | 框架和编译器之间，怎样准确、稳定地交接计算 |
| Mosaic | 基于 MLIR 的内核编译技术，包含 TPU 和 GPU 路径 | 怎样把内核里的计算与数据搬运映射到目标硬件 |

这三个名字处在不同的分类维度。**MLIR 提供表达和转换程序的基础，StableHLO 侧重计算语义的交接，Mosaic 侧重内核实现。** 下文中的 Mosaic 指 JAX/Pallas 生态里的 Mosaic；“Mosaic IR”指这套编译流程中使用的中间表示。[MLIR 概述](https://mlir.llvm.org/docs/Rationale/Rationale/)、[StableHLO 概述](https://openxla.org/stablehlo)、[Pallas 编译设计](https://docs.jax.dev/en/latest/pallas/design/design.html#lowering-pallas)。

## 为什么编译器需要中间表示

IR 是 Intermediate Representation 的缩写，中文叫**中间表示**。它把程序写成便于编译器分析和改写的结构，处在源代码与最终执行代码之间。

同一个矩阵乘法，可以有不同层次的表达：

- 在数学层面，它是两个矩阵相乘，输出的形状由输入决定。
- 在分块层面，它是许多小矩阵乘法，以及沿着归约维度累加部分结果的循环。
- 在硬件层面，它涉及具体的存储位置、指令和执行顺序。

每个层次保留的信息不同。看到“矩阵乘法”，编译器容易考虑调用矩阵库、合并相邻运算；看到内存读写和指令，编译器才能处理地址计算与指令调度。过早把矩阵乘法拆成大量标量操作，会让它更难重新识别原来的整体结构。

因此，现代编译器常常分阶段补上执行细节。这个过程叫 **lowering**，可以理解为“把抽象操作逐步落实成更具体的实现”。MLIR 的多层次设计正是为了支持这样的渐进转换。[MLIR 设计动机](https://mlir.llvm.org/docs/Rationale/Rationale/#introduction-and-motivation)。

还有一个常见词是 **pass**：编译器中的一道处理步骤。例如，某个 pass 删除无用计算，另一个合并重复表达式，还有一些 pass 负责转换表示。Lowering 可以由多个 pass 完成；也有很多 pass 只在当前层次做优化。

## MLIR 让不同层次的语言共用一套基础设施

MLIR 的全称是 **Multi-Level Intermediate Representation**。它既有共同的 IR 结构，也提供构建编译器所需的工具。

它的关键概念是 **dialect，方言**。每种方言定义一组操作、类型或属性，用来描述某一类问题。例如：

| 方言 | 典型操作 | 表达的内容 |
| --- | --- | --- |
| `func` | `func.func`、`func.return` | 函数及其返回值 |
| `arith` | `arith.addf`、`arith.addi` | 浮点和整数算术 |
| `scf` | `scf.for`、`scf.if` | 循环和条件分支 |
| `memref` | `memref.load`、`memref.store` | 对内存缓冲区的访问 |
| `stablehlo` | `stablehlo.dot_general` | 高层张量计算 |

操作名称里的前缀，就是所属方言的命名空间。各方言的定义可以在 [MLIR 方言文档](https://mlir.llvm.org/docs/Dialects/)和 [StableHLO 规范](https://openxla.org/stablehlo/spec)中查到。

这些方言可以出现在同一个程序里。一段 StableHLO 程序可以用 `func.func` 表达函数，用 `stablehlo.add` 表达张量加法。不同方言共用操作、值、类型、代码区域等基础结构，因此能够接入共同的解析、打印和变换工具。[MLIR 语言参考](https://mlir.llvm.org/docs/LangRef/#dialects)。

这有点像一套公共文档格式：不同专业可以定义自己的术语和内容结构，同时共用编辑器和检查工具。不过，方言还定义了操作的精确语义，编译器必须理解这些规则才能正确改写程序。

### 共用基础设施之后仍然需要转换规则

采用 MLIR，并不会自动获得“任意方言互相转换”的能力。开发者仍然需要提供规则，说明一个操作如何用另一组操作实现，并检查转换后的类型和语义是否合法。[MLIR 方言转换框架](https://mlir.llvm.org/docs/DialectConversion/)。

例如，张量层面描述的是一个值，而内存层面还要确定它由哪个缓冲区承载、能否复用已有空间、什么时候必须复制。在 MLIR 中，把张量语义转换为内存缓冲区语义的过程叫 **bufferization**。这包含实质性的分析与决策。[MLIR Bufferization 文档](https://mlir.llvm.org/docs/Bufferization/)。

MLIR 的价值在于让这些工作能够复用基础设施、分层组合。一个编译器最终支持哪些程序、生成多好的代码，还取决于它实现了哪些方言、转换和优化。

## StableHLO 把计算的含义交接清楚

假设一个框架已经知道，某段程序需要矩阵乘法、广播和逐元素加法。它希望把计算交给不同的编译器，而不必为每个后端重新解释一次这些操作的含义。

StableHLO 提供的就是这样的交接层：它定义高层机器学习运算的输入、输出、类型约束和语义，供产生它的框架与消费它的编译器共同遵守。这里的 HLO 指 High-Level Operations，高层操作。[StableHLO 概述](https://openxla.org/stablehlo)。

以 `x @ w` 为例，StableHLO 可以用 `stablehlo.dot_general` 表达。这个操作既能覆盖普通矩阵乘法，也能表达带批次维度的点积；属性负责说明哪些维度对应批次、哪些维度参与求和。对形状为 `[2, 3]` 和 `[3, 4]` 的两个矩阵，参与求和的是左边的第 1 维与右边的第 0 维，输出形状是 `[2, 4]`。[dot_general 规范](https://openxla.org/stablehlo/spec#dot_general)。

此时，“计算什么”已经比较明确，但该选择多大的计算块、怎样安排片上存储，仍可以留给后续编译阶段。保留这些选择空间，才能让不同设备采用适合自己的实现。

### Stable 指有边界的兼容性承诺

StableHLO 对通过指定兼容性 API 生成的 **portable artifact，可移植制品**，提供有条件的版本兼容保证：新版库可以读取五年窗口内的旧制品；旧版库在两年窗口内可以读取新版生成的制品，但程序不能依赖旧版尚未支持的新特性。

这些保证有明确范围：随手打印出来的 MLIR 文本不自动享有相同承诺，跨设备、跨版本的浮点结果也不保证逐位相同。实际部署还需要接收端支持所用操作及扩展。[StableHLO 兼容性说明](https://openxla.org/stablehlo/compatibility)。

因此，把一段计算保存为规范的 StableHLO 制品，有助于长期交接；而将调试输出复制到文件，主要解决的是“方便人阅读”这个问题。

### StableHLO 和 XLA HLO 各有用途

阅读 XLA 资料时，还会遇到 HLO。可以把 StableHLO 看成框架与编译器之间的一种公共交接表示，把 XLA 内部的 HLO 看成 XLA 优化程序时使用的表示。

在 XLA 的 GPU 编译流程中，HLO 会经历布局分配、融合、库调用选择等处理，逐渐包含更多实现决策。因此，前端导出的 StableHLO 与优化后的 HLO，反映的是不同阶段的问题。[XLA 从 HLO 到可执行程序的流程](https://openxla.org/xla/hlo_to_thunks)。

## Mosaic 让内核的执行安排进入表示

再看开头的矩阵计算。如果自动编译得到的性能还不够好，开发者可能想控制更多细节：一次处理哪一块数据，把临时结果保留在哪里，能否让数据搬运与计算重叠。

这时就进入了 **kernel，内核**的层面：开发者为一段计算安排更具体的执行方式。JAX 的 Pallas 是编写 GPU、TPU 自定义内核的语言，提供分块、内存访问和流水线相关能力；Mosaic 是它背后的编译路径之一。[Pallas 概述](https://docs.jax.dev/en/latest/pallas/index.html)。

### Mosaic TPU 使用通用方言和 TPU 专用操作

在 TPU 路径中，Pallas 会把内核里的 JAX 操作转换成 MLIR，使用 `vector`、`arith` 等方言。Mosaic 再把这些表示继续转换为 TPU 后续编译所需的低层表示。JAX 的设计文档把后者称为 LLO。[Pallas 到 Mosaic TPU 的转换](https://docs.jax.dev/en/latest/pallas/design/design.html#lowering-pallas-to-mosaic-for-tpu)。

这套表示也包含硬件相关操作。例如，公开的 TPU 方言定义了 `tpu.matmul` 和 `tpu.enqueue_dma`，分别涉及矩阵乘法与异步数据搬运。由此可见，“Mosaic IR”通常要结合所在编译阶段来理解，它可能是多种 MLIR 方言共同构成的程序。[TPU 方言操作定义](https://github.com/openxla/xla/blob/3cfc839fda885f8e470c0fc7f8d8cf95d5588919/xla/mosaic/dialect/tpu/tpu_ops.td)。

对一个较大的矩阵乘法，可以把其中一种实现安排想象成下面的过程。**这是执行思路的伪代码，不是 Mosaic 的实际语法。**

```text
选定输出矩阵的一块 Y_tile
为这块输出准备累加器
沿矩阵乘法的求和维度，逐块处理：
    把需要的 X_tile 和 W_tile 搬到片上存储
    等待当前块的数据就绪
    执行这一块的矩阵乘法并累加
    在缓冲区和依赖允许时，提前搬运下一块
加上偏置并执行激活函数
把 Y_tile 写回输出
```

这里的 **tile** 就是“大矩阵切出来的小块”。**Layout，布局**决定数据如何分布；**pipeline，流水线**则安排不同阶段在时间上的重叠。

以 TPU TensorCore 内核为例，HBM 是容量较大的设备主存，VMEM 是更靠近计算单元的片上向量存储。数据从 HBM 搬到 VMEM 后，可以被内核更快地访问。Pallas 的 `BlockSpec` 用来描述每次内核调用访问哪一块数组，编译器可以据此安排搬运；开发者也能进一步控制流水线。[Pallas TPU 内存与分块说明](https://docs.jax.dev/en/latest/pallas/tpu/details.html#blockspecs-and-grid-iteration)。

同一个数学公式，可能有多种这样的安排。块太小，可能增加搬运和调度开销；块太大，又可能超过可用存储或增加寄存器压力。具体哪种更快，需要结合硬件与实际测量判断。

### Mosaic 也有 GPU 路径

Mosaic GPU 同样使用 MLIR 基础设施，但它面对的是 GPU 的执行和存储模型，需要处理线程协作、共享内存、布局与异步操作等问题。JAX 的 Mosaic GPU lowering 实现也会组合使用多种 MLIR 方言。[Mosaic GPU 文档](https://docs.jax.dev/en/latest/pallas/gpu/reference.html)、[Mosaic GPU lowering 源码](https://github.com/jax-ml/jax/blob/6aaf234b1ee98800fcb80b9014a052a66dadbd8a/jax/experimental/mosaic/gpu/dialect_lowering.py)。

所以，读到“Mosaic IR”时，需要同时确认两个信息：**目标是 TPU 还是 GPU，以及当前看到的是哪个编译阶段。** 两条路径共享部分思路，但具体的操作、布局和硬件约束有所不同。

## 把普通计算和自定义内核放到同一张地图上

对普通的 JAX 数组计算，可以用下面的简图理解主要过程。Jaxpr 是 JAX 在跟踪 Python 计算后得到的内部表示。

```text
JAX 数组程序
    ↓ 跟踪计算
Jaxpr
    ↓ lowering
StableHLO
    ↓ 交给 XLA
HLO 优化与目标设备代码生成
    ↓
CPU / GPU / TPU 可执行程序
```

这张图省略了大量优化步骤，但对应了 JAX 文档所描述的跟踪、降级、编译和执行阶段。[JAX 编译阶段说明](https://docs.jax.dev/en/latest/aot.html)。

当程序中调用 Pallas 自定义内核时，还会有内核自身的编译路径。外层负责将它接入整体程序，内层负责实现这段计算：

```text
外层 JAX 程序
    ↓
StableHLO 中的普通运算 + 自定义调用
                           │
                           └── 对接 Pallas 内核
                                   ↓
                              内核的 Jaxpr
                                   ↓
                         按所选后端进入编译流程
                           ↙                 ↘
                  Mosaic TPU             Mosaic GPU
                           ↘                 ↙
                           对应设备上的内核执行
```

这是一张职责关系图，箭头不表示编译器内部严格的执行时序。JAX 的 TPU 接入代码会将 Mosaic 模块降级为自定义调用；GPU 也有对应的内核接入实现。[TPU 接入源码](https://github.com/jax-ml/jax/blob/6aaf234b1ee98800fcb80b9014a052a66dadbd8a/jax/_src/pallas/mosaic/pallas_call_registration.py)、[GPU 接入源码](https://github.com/jax-ml/jax/blob/6aaf234b1ee98800fcb80b9014a052a66dadbd8a/jax/_src/pallas/mosaic_gpu/pallas_call_registration.py)。

因此，在外层 StableHLO 中看见 `custom_call`，只能说明那里接入了一个由实现定义的操作。它的具体行为由调用目标及相关配置决定；接收端也必须认识这个目标，才能执行它。包装在 StableHLO 里，并不会自动消除内核对特定后端的依赖。[StableHLO custom_call 规范](https://openxla.org/stablehlo/spec#custom_call)。

**MLIR 是这些表示共用的基础设施，不需要作为图里独立的一站。普通 JAX 程序也不需要为了生成机器代码，先变成开发者编写的 Pallas/Mosaic 内核。**

## 在 CPU 上亲眼看看 StableHLO

已经安装 JAX 的环境里，可以运行下面这段代码。它只需要 CPU，就能观察开头那行矩阵计算的中间表示。

```python
import jax
import jax.numpy as jnp


def dense(x, w, b):
    return jnp.maximum(x @ w + b, 0.0)


x = jax.ShapeDtypeStruct((2, 3), jnp.float32)
w = jax.ShapeDtypeStruct((3, 4), jnp.float32)
b = jax.ShapeDtypeStruct((4,), jnp.float32)

print(jax.make_jaxpr(dense)(x, w, b))

lowered = jax.jit(dense).lower(x, w, b)
print(lowered.compiler_ir(dialect="stablehlo"))
```

`ShapeDtypeStruct` 只提供形状和数据类型，不需要分配真实输入数组。这里让 JAX 跟踪并降低计算，还没有请求最终可执行程序的编译或运行。[JAX AOT 文档](https://docs.jax.dev/en/latest/aot.html)。

下面摘取输出中的矩阵乘法和加法，并把矩阵乘法这一行折行显示。示例输出来自 JAX/jaxlib 0.11.1；不同版本的临时值名称和辅助操作可能变化。

```text
%0 = stablehlo.dot_general %arg0, %arg1,
  contracting_dims = [1] x [0],
  precision = [DEFAULT, DEFAULT]
  : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>

%3 = stablehlo.add %0, %2 : tensor<2x4xf32>
```

读这几行时，可以按下面的顺序拆开：

1. `%arg0` 和 `%arg1` 是输入值，`%0` 是矩阵乘法的结果。`%` 后的名字用来连接数据依赖，不是硬件寄存器编号。
2. `tensor<2x3xf32>` 表示形状为 `2 × 3`、元素类型为 32 位浮点数的张量。
3. `contracting_dims = [1] x [0]` 表示沿左输入第 1 维、右输入第 0 维求和，这里维度从 0 开始编号。
4. `%2` 是前面广播偏置得到的值，形状已经扩展为 `[2, 4]`，所以可以和 `%0` 逐元素相加。

这些值采用 SSA，也就是静态单赋值的形式：每个 SSA 值由一个定义产生，后面的操作引用它。这样编译器能沿着定义和使用关系分析程序。[MLIR IR 结构](https://mlir.llvm.org/docs/LangRef/#high-level-structure)。

完整输出里还能看到 `stablehlo.broadcast_in_dim`、零常量和 `stablehlo.maximum`。Python 里的隐式广播，在这里变成了显式操作；`maximum` 则把负数截为零。看懂它们，就已经能把这段 IR 对回最初的公式。

这里也能直接观察到 MLIR 的“多方言共存”：函数外壳是 `func.func`，函数内部的张量运算属于 `stablehlo`。

### 从 IR 推断性能时要看清阶段

这份 StableHLO 说明了有哪些计算和数据依赖，但后续编译器仍可能融合操作、改变布局或选择不同实现。看到三个运算，不能据此认定最终会启动三个设备内核；看到一个广播，也不能据此认定会在设备内存里复制出完整数组。[XLA 优化流程](https://openxla.org/xla/hlo_to_thunks)。

同样，打印一个包含 Pallas 调用的外层程序时，可能主要看到自定义调用边界。想理解内核内部的搬运和布局，就要继续查看相应后端的 IR 或生成代码。

## 遇到问题时应该看哪一层

不必从第一天起就读懂全部编译器。可以先用手头的问题选择观察位置：

| 你想弄清的问题 | 适合先看的内容 |
| --- | --- |
| Python 代码被跟踪成了哪些计算 | Jaxpr |
| 矩阵维度、广播、数据类型是否符合预期 | StableHLO |
| 融合、布局或库调用发生了什么变化 | 优化后的 HLO 与编译器报告 |
| Pallas 内核如何分块、搬运、同步 | Pallas 源码及对应 Mosaic 编译阶段的 IR |
| 时间实际花在计算、搬运还是等待上 | 设备 profiler 与测量结果 |
| 准备为编译器增加操作或转换 | MLIR 方言、验证规则和 pass |

这些层次能够互相解释：高层表示让我们看清计算意图，低层表示帮助我们追踪执行安排，而实际测量检验安排是否有效。读 IR 时，先确认它保留了什么信息、还有什么决策留给后续阶段，往往比记住更多操作名更有用。

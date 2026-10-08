---
title: "JAX 软件栈：从 JAX 到 TPU、CPU 和 GPU"
description: "从 Jaxpr、MLIR Module、IFRT Array、PJRT Buffer 到 HLO、LLVM IR 和设备运行时，结合分层 SVG 追踪 JAX 的编译、执行、结果返回与 Pallas 内核接入。"
pubDate: "2026-10-09"
tags: ["JAX", "XLA", "Pallas", "Compiler", "TPU", "CPU", "GPU"]
draft: false
---

> 本文由文章作者提出大纲，AI完成，并且经由作者进行润色，去AI味和审阅，作者对文章内容负责。

本文是[分层架构 SVG](/blog/jax-software-stack/overview-software-stack-components-layered.svg) 的配套说明, 请配合图片进行阅读。

[![JAX 软件栈分层架构：编译、执行与完整返回路径](/blog/jax-software-stack/overview-software-stack-components-layered.svg)](/blog/jax-software-stack/overview-software-stack-components-layered.svg)

图中文字较密，建议[打开完整 SVG](/blog/jax-software-stack/overview-software-stack-components-layered.svg)后放大查看；节点、接口索引和源码链接均可点击。

## 1. 不同组件的核心抽象概念

每个组件选取一个核心抽象概念，并围绕此进行展开，从该概念抽象了哪些东西以及具体的数据结构如何表示？数据结构如何产生？会经过哪些变换？会被哪些组件消费？ 这四个角度来进行探索。

| 组件 | 核心抽象 | 含义 | 生命周期 |
|---|---|---|---|
| JAX | [Jaxpr](/blog/jax-software-stack/overview-software-stack-components-layered.svg#outer) | 用于表达jax/pallas 代码的IR, 描述输入如何通过原语及其组合产生输出 | 由 Python 函数和抽象输入经[trace][src-trace]产生，经过[自动微分（如 JVP）][src-jvp-jaxpr]、[批处理（vmap）][src-batch-jaxpr]、[部分求值][src-partial-eval-jaxpr]和[死代码消除][src-dce]等变换；随后由 JAX 的 [lowering][src-lower] 按原语规则生成外层 MLIR Module，交给原生编译路径。 |
| jaxlib | [MLIR Module（StableHLO/Mosaic）](/blog/jax-software-stack/overview-software-stack-components-layered.svg#binding_module) | 以 [MLIR Module][src-module-op] 为容器，使用 [StableHLO][src-stablehlo-add]、[Mosaic 等专用 IR][src-mosaic-module] 表示计算图的中间表示。 | 由 [JAX][src-lower]/[Pallas lowering][src-pallas-lowering] 从 Jaxpr 产生; jaxlib 把MLIR module 封装成ifrt:HloProgram, ifrt通过调用pjrt交给具体的设备后端的编译接口。|
| IFRT | [Array](/blog/jax-software-stack/overview-software-stack-components-layered.svg#ifrt_array) | 跨设备的逻辑数组。[`ifrt::Array`][src-ifrt-array] 通过 `ArraySpec` 描述元素类型、形状、分片和布局；| 将输入数据对应的设备 Buffer 与类型、形状、分片信息[组合为逻辑数组][src-array-create]，供 [LoadedExecutable 执行][src-ifrt-execute]；执行返回时，再将输出 Buffer [组织为新的逻辑数组][src-ifrt-outputs]。 |
| PJRT | [PjRtBuffer](/blog/jax-software-stack/overview-software-stack-components-layered.svg#pjrt_buffer) | 对[单个设备上的数据存储的统一抽象][src-pjrt-buffer]，描述所属设备、内存空间、数据形状与布局，并提供所有权和[就绪状态][src-ready]的管理接口。 | 在接收[输入数据][src-buffer-from-host]或准备输出存储时创建，可按需要[复制][src-buffer-copy]或复用；作为 [PjRtLoadedExecutable][src-pjrt-execute] 的执行输入和输出，返回的输出 Buffer 由 IFRT [组织为逻辑数组][src-ifrt-outputs]，不再使用时[释放存储引用][src-buffer-delete]。 |
| XLA | [HLO（HloModule）](/blog/jax-software-stack/overview-software-stack-components-layered.svg#hlo) | 用于优化、规划和代码生成的计算图。[`HloModule`][src-hlo] 包含入口及其他 `HloComputation`，由 `HloInstruction` 表达计算和依赖，并携带形状、布局及分片等约束。 | 由 [StableHlo 导入][src-import]产生，经 [HLO pass pipeline][src-hlo-passes] 规范化并进行目标相关优化，再交给 [CPU][src-cpu-backend]／[GPU 后端][src-gpu-backend]组织存储规划、目标代码与执行计划，形成相应 `Executable`。 |
| XLA CPU 后端 | [LLVM IR](/blog/jax-software-stack/overview-software-stack-components-layered.svg#cpu_ir) | 面向 CPU 代码生成的低层中间表示。描述具体的数据运算，内存访问和控制流，并携带目标平台与数据布局信息。 | 由 CPU后端根据优化后的HLO、存储规划和目标信息生成，经 [LLVM 优化][src-cpu-ir-passes]后交给[目标代码生成][src-cpu-machine-code]得到对象文件；链接后的函数库与 thunk 执行计划共同组成 `CpuExecutable`。 |
| XLA GPU 后端 | [LLVM IR](/blog/jax-software-stack/overview-software-stack-components-layered.svg#gpu_ir) | 面向 GPU kernel 代码生成的[低层中间表示][src-llvm-module]，描述 kernel 内的数据运算、内存访问和控制流，并通过目标相关的指令与约定表达线程协作、地址空间和 kernel 入口。 | 由 [GPU 后端][src-gpu-emit]根据优化后的 HLO、存储规划和目标设备信息生成，按需链接设备 bitcode，经 [LLVM 优化][src-gpu-ir-passes]和目标代码生成得到设备代码；其中 [NVIDIA 路径][src-nvptx-binary]生成 PTX 并编译为 cubin。设备代码与存储规划、thunk 执行计划共同组成 [`GpuExecutable`][src-gpu-backend]，交回 PJRT。 |
| TPU provider / libtpu | [HLO/LLO](/blog/jax-software-stack/overview-software-stack-components-layered.svg#tpu_boundary) | 本概览保留的 TPU 私有程序表示研究项。当前固定公开源码尚不能确认 LLO 的结构、语义和约束；Mosaic TPU MLIR 是另一种已知表示。 | 公开可确认的[编译边界][src-pjrt-program]是：程序与选项进入 TPU provider，返回已加载可执行对象供 PJRT 执行。LLO 在内部从何产生、经过哪些变换、由谁消费，仍待固定版本证据。 |
| 目标汇编表示 | [ASM](/blog/jax-software-stack/overview-software-stack-components-layered.svg#asm) | 以目标指令及其操作数描述计算、数据访问和控制转移的可读程序表示，可面向硬件指令集或 PTX 等虚拟指令集。 | 由编译器的目标代码生成产生，经汇编或[进一步目标编译][src-nvptx-binary]转换为机器代码，供后续链接和加载；也可由已有机器代码反汇编得到，供开发者检查代码生成结果和分析性能。 |
| 设备运行时与驱动 | [执行提交](/blog/jax-software-stack/overview-software-stack-components-layered.svg#submit) | 一次程序运行的请求与[执行上下文][src-cpu-submit]，关联已加载程序、输入输出存储、执行参数和依赖关系，描述执行什么、使用哪些数据以及何时可以执行。 | 由 [PJRT 执行接口][src-pjrt-execute]进入具体后端，结合可执行对象和输入 Buffer [准备所需存储][src-cpu-execute]，按执行计划和依赖关系[调度计算、通信与数据搬运][src-thunk-execute]；执行结果写入输出 Buffer，并向上层传播[完成或错误状态][src-ready]。 |
| 指令集接口与硬件 | [ISA](/blog/jax-software-stack/overview-software-stack-components-layered.svg#isa) | 软件与硬件之间的指令级语义约定，规定指令及其编码、寄存器等可见状态，以及执行指令时的计算、访存和控制转移行为。 | 由处理器体系结构规范定义，并随[架构版本与扩展演进][doc-isa-evolution]；编译器依据目标支持的[指令与特性][src-cpu-target]生成[机器代码][src-cpu-machine-code]，硬件按相应指令语义执行程序，更新寄存器、内存和控制状态。 |

图中 JAX 和 jaxlib 居中；IFRT/PJRT、XLA/编译后端、ASM/设备运行时左右并列；硬件位于底部。这个分布同时表达组件职责和调用关系，纵向位置不代表所有请求都必须依次经过每个框。

填色和粗边框用于核心节点。蓝线表示编译与程序传递，绿线表示执行与输入数据，紫线表示程序变换，橙色虚线表示编译结果返回，粉色实线表示输出对象返回，粉色虚线表示完成状态。灰色虚线说明结构、持有或观察关系；棕色虚线说明 TPU 内部证据边界。线上的拱桥表示交叉处不连接。

## 2. 从 JAX 代码到可执行程序的编译流程

一次需要编译的 JAX 调用，会先把jax代码lowering到各种IR，再生成目标设备可以执行的代码。

下面讨论 JAX 通过 jaxlib、IFRT 和 PJRT 调用设备后端的路径。文中的设备后端实现，也就是源码和图中常见的 device backend，负责实现具体设备的编译、加载和执行接口。

### 2.1 追踪和函数变换：构造 Jaxpr

JAX 追踪 Python 函数时，根据输入的shape和dtype等抽象信息，创建Tracer， 作为占位符代表动态输入参与函数调用中，并将过程中发生的原语运算记录为jaxpr，追踪得到的 Jaxpr 则记录输入、常量、原语方程、输出和 effects。每个变量的 `aval` 描述其抽象类型，方程说明这些变量如何参与计算。相关定义在 [`Jaxpr`][src-jaxpr]、[`JaxprEqn`][src-eqn] 和 [`Var`][src-var] 中，追踪入口之一是 [`trace_to_jaxpr_dynamic`][src-trace]。

Jaxpr 承接 JAX 的函数变换。不同变换处理的问题不同：

| 变换 | 对程序做什么 | 源码入口 |
|---|---|---|
| 自动微分 | 例如 JVP 在原函数的计算之外，构造切向量的传播与计算。 | [`jvp_jaxpr`][src-jvp-jaxpr] |
| 批处理 | 根据输入的批次维度改写原语调用，得到可以处理一批输入的程序。 | [`batch_jaxpr`][src-batch-jaxpr] |
| 部分求值 | 按输入是否已知拆分计算，并记录两部分之间需要传递的中间结果。 | [`partial_eval_jaxpr_nounits`][src-partial-eval-jaxpr] |
| 死代码消除 | 根据输出使用情况和 effects，移除不再需要的计算。 | [`dce_jaxpr`][src-dce] |

这些变换可以参与追踪，也可以消费已有的 Jaxpr。具体组合取决于函数使用了哪些 JAX 变换。自动微分和批处理会改变程序计算的内容，死代码消除则主要减少无用工作。

### 2.2 lowering：从 Jaxpr 构造 MLIR Module

[`lower_jaxpr_to_module`][src-lower] 建立模块、函数入口和输入输出约束；[`jaxpr_subcomp`][src-subcomp] 遍历 Jaxpr 方程，按原语和目标平台选择 lowering 规则。规则接收已有的 MLIR 输入值，生成操作，并把结果值交给后续方程。

这里的 MLIR Module 是程序容器。常规张量计算主要用 StableHLO 表达，分片等信息由配套方言和属性记录。jaxlib 提供构造这些对象的绑定。Pallas 内核还会生成自己的专用表示，具体接入方式见第四章。参见 [Module 定义][src-module-op]和 [StableHLO 操作定义][src-stablehlo-add]。

模块构造完成后，JAX 调用 MLIR verifier 检查结构、类型和操作约束；启用 Shardy 时，还会运行 `sdy-lift-inlined-meshes` 整理 mesh 表示。这些处理的对象仍然是 MLIR Module。校验入口和模块 pass 的调用都在 [`lower_jaxpr_to_module`][src-lower] 中。

### 2.3 jaxlib、IFRT 和 PJRT 如何传递编译请求

jaxlib 的 [`PyClient::CompileAndLoad`][src-py-compile] 接收 Module 和编译选项，克隆 Module，再把它封装为 `ifrt::HloProgram`。这个名称容易与 `HloModule` 混淆：[`HloProgram`][src-hlo-program] 在此处持有的仍然是 MLIR Module。

随后，jaxlib 调用 IFRT 的默认编译器。当前讨论的 IFRT 实现通过 PJRT 工作：[`PjRtCompiler::CompileAndLoad`][src-ifrt-compiler] 取出 Module 和编译选项，交给 IFRT 的可执行对象构造流程，最终调用设备实现的 [`PjRtClient::CompileAndLoad`][src-pjrt-compile]。

| 调用方 → 接收方 | 传递的内容 | 接收方接着做什么 |
|---|---|---|
| JAX → jaxlib | MLIR Module、设备信息和编译选项。 | 克隆 Module，构造 `HloProgram` 和 IFRT 编译选项。 |
| jaxlib → IFRT | `HloProgram`、IFRT 编译选项。 | 通过默认 Compiler 进入相应的编译实现。 |
| IFRT → PJRT | 从 Program 中取出的 Module、XLA 编译选项。 | 调用具体设备的编译与加载入口。 |

这一段主要处理接口适配、对象所有权和编译配置。CPU/GPU 路径在后续导入过程中才构造 XLA 的 `HloModule`。

### 2.4 XLA：导入 HLO，优化计算并安排存储

`HloModule` 表示一个完整的 XLA 程序，包含入口 `HloComputation` 和其他被调用的计算。每个计算由 `HloInstruction` 及其依赖关系组成，形状、布局、分片和编译配置进一步约束程序。[`HloModule` 的定义][src-hlo]说明了模块与入口计算的关系。

CPU 编译入口先通过 MLIR 编译准备流程导出 HLO，再用 `HloModule::CreateFromProto` 构造模块。之后，`JitCompile` 调用 `CpuCompiler::RunHloPasses` 和 `RunBackend`，分别完成 HLO 变换和后端编译。参见 [CPU 的 Module 编译入口][src-cpu-module]、[HloModule 构造][src-cpu-hlo-create]和 [`JitCompile`][src-cpu-jit]。

HLO passes 会做计算简化、死代码消除、算子融合、分片和布局分配等工作。后端根据目标设备及编译配置选择 pass 和执行顺序。经过这些变换，程序逐步确定哪些计算合并执行、数据如何分布、张量在内存中如何排列。pass 的组织方式见 [`HloPassPipeline`][src-hlo-passes]，具体流水线见 [CPU 编译器][src-cpu-hlo-passes]和 [GPU 编译器][src-gpu-backend]。

调度确定计算的执行顺序，存储规划据此安排输入、输出和临时值使用哪些空间。[`BufferAssignment`][src-buffer-assignment] 记录编译期的分配及切片关系，代码生成和运行时都要使用这些信息；实际分配和持有设备存储的对象则在运行时创建。

### 2.5 CPU 和 GPU 如何生成代码

CPU 后端根据优化后的 HLO、存储规划和目标信息生成 LLVM IR。`llvm::Module` 中的函数、基本块和指令描述具体运算、内存访问与控制流。CPU 的 emitter 生成计算函数，`ThunkEmitter` 同时组织这些函数和库调用，形成运行时要执行的工作序列。相关构造集中在 [`CompileCpuExecutable`][src-cpu-compile-body]。

随后，[`IrCompiler::RunIrPasses`][src-cpu-ir-passes] 运行 LLVM 优化，[`EmitMachineCode`][src-cpu-machine-code] 生成对象代码。当前 JIT 路径将代码链接为可调用的函数库，再与 `BufferAssignment`、thunk 序列和常量等共同构成 `CpuExecutable`。thunk 表示一次具体工作，例如调用计算函数、调用库、复制数据或处理控制流。对象关系可结合 [CPU LLVM IR 详细图](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/research/software-stack/xla/cpu-llvm-ir-centered-hub.svg)阅读。

GPU 后端根据 HLO 操作选择 kernel 生成路径或设备库实现。需要生成 kernel 的部分经过相应 lowering，形成面向 GPU 的 LLVM IR，其中用目标相关约定表达地址空间、kernel 入口和线程操作；设备库调用则进入 thunk 执行计划。存储规划、代码生成上下文和 thunk 构造的入口是 [`CompileModuleToLlvmIr`][src-gpu-emit]。

GPU LLVM IR 按需链接设备 bitcode，并经过 [LLVM 优化][src-gpu-ir-passes]。NVIDIA 路径由 [`NVPTXCompiler::CompileTargetBinary`][src-nvptx-binary] 生成 PTX，再调用 compilation provider 编译为 cubin。设备代码、常量、存储分配信息和 thunk 执行计划一起构成 [`GpuExecutable`][src-gpu-executable-create]。更多分支见 [GPU IR 详细图](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/research/software-stack/xla/gpu-ir-centered-hub.svg)。

### 2.6 汇编、机器代码与 ISA

汇编用目标指令和操作数表示具体程序。它可以是编译工具之间传递的程序，也可以是从已有机器代码反汇编得到的观察结果。在上述 NVIDIA 路径中，PTX 会实际交给后续编译器；CPU 的 `EmitMachineCode` 可以直接输出对象代码，查看汇编时再使用相应的反汇编工具。参见 [PTX 编译实现][src-nvptx-binary]和 [LLVM 反汇编工具][doc-llvm-objdump]。

ISA 规定这些指令的编码、操作语义，以及寄存器和内存等软件可见状态的变化。它由处理器体系结构定义，并随架构版本和扩展演进。编译器根据目标支持的指令与特性生成机器代码，硬件实现这些指令语义。CPU 后端通过 [`TargetMachine`][src-cpu-target] 选择目标能力；架构演进可参考 [Arm 的 ISA 更新说明][doc-isa-evolution]。PTX 定义的是虚拟指令集，还需要转换为目标 GPU 的机器代码。[PTX 规范][doc-ptx]给出了这一层的定义。

### 2.7 TPU：插件、编译输入与后端程序

JAX 的 [`make_tpu_client`][src-tpu-client] 按需加载并初始化 `libtpu.so` 的 PJRT 插件，取得 Client。上层继续使用 IFRT/PJRT 的接口；具体的 TPU 编译、加载与执行由这个 provider 实现。

程序到达 [`PjRtCApiClient::CompileAndLoad`][src-capi-compile] 后，`InitializeArgsAndCompile` 序列化编译选项，并调用 `SerializeProgram` 处理程序。MLIR 输入按插件报告的 StableHLO、Shardy 版本序列化；`XlaComputation` 输入则序列化为 HLO proto。程序字节与格式组成 `PJRT_Program`，经插件的 `PJRT_Client_Compile` 入口提交。成功返回的 `PJRT_LoadedExecutable` 被包装为 `PjRtCApiLoadedExecutable`，继续交回 IFRT 和 jaxlib。这里传递的是编译输入与执行能力，尚未传入本次运行的数组数据。参见 [程序序列化][src-capi-program]和 [编译参数构造与调用][src-capi-compile-call]。

TPU 路径需要分清外层程序、内层 kernel 与后端低层程序。外层 MLIR 中的常规张量操作交给 XLA 编译；使用 Pallas 时，`tpu_custom_call` 的配置还携带独立的 Mosaic TPU MLIR bytecode。固定版本的 [Pallas 设计文档][src-pallas-mosaic-design]明确说明 Mosaic 消费 MLIR 并生成 LLO。因此，Mosaic TPU MLIR 与 LLO 是编译过程中不同阶段的表示；这个设计说明本身尚未给出 LLO 的数据结构、变换函数和最终编码接口。

这些对象的定义、两条程序来源及后续消费关系在 [TPU 后端说明](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/research/software-stack/libtpu/llo-boundary-centered-hub.md)中展开。[libtpu 询证 issue #76][issue-tpu-overview]集中追问内部 HLO、LLO、机器代码和运行时对象之间的具体调用关系；内部答复与固定公开源码分别注明来源。

### 2.8 编译结果如何回到 JAX

编译返回时，各层为后续执行保留自己需要的对象和元信息。CPU 是一个可以完整追踪的例子：`CpuExecutable` 被包装为 `PjRtCpuExecutable`，再由 `LoadInternal` 构造已加载的 `PjRtCpuLoadedExecutable`。参见 [CPU 编译产物包装][src-cpu-wrap]和 [加载实现][src-cpu-load]。

接下来，IFRT 包装设备实现返回的 `PjRtLoadedExecutable`，并保存输出类型、形状、分片和布局等元信息。jaxlib 等待 IFRT 的编译结果，构造 `PyLoadedExecutable`。JAX 的 `ExecuteReplicated` 持有这个对象，后续通过它发起执行。对应入口是 [IFRT 包装][src-ifrt-wrap]、[Python 包装][src-py-wrap]和 [`ExecuteReplicated`][src-dispatch]。

```text
设备后端生成并加载程序
  → PJRT 的已加载可执行对象
  → IFRT LoadedExecutable
  → jaxlib PyLoadedExecutable
  → JAX 执行入口持有并复用
```

这些包装保存的是执行能力及其元信息。数组结果要等实际调用程序时产生。

## 3. 从输入数组到输出数组的运行流程

执行时需要同时提供两类东西：编译好的程序，以及本次调用的数据。程序决定计算和存储安排，输入数组提供实际值；运行时把二者组合起来，处理依赖并提交工作。

### 3.1 IFRT Array 如何组织设备上的数据

在这条路径中，Python 可见的设备数组通过 `PyArray` 持有 IFRT Array。IFRT Array 表示一个完整的逻辑数组，`ArraySpec` 描述元素类型、全局形状、分片和布局。一个逻辑数组可以放在单个设备上，也可以分片或复制到多个设备上。“逻辑”强调从完整数组的角度看数据，并不要求一定跨设备或跨主机。参见 [`PyArray`][src-py-array]和 [`ifrt::Array`][src-ifrt-array]。

使用 PJRT 的 IFRT 实现把当前进程可访问的设备 Buffer 与数组元信息组织在一起。`PjRtArray::Create` 接收这些信息并构造数组；`DisassembleIntoSingleDeviceArrays` 可以再取得按设备组织的数组。这里的组合和拆分主要处理数组视图及存储引用，是否复制数据取决于具体操作和复制语义。参见 [数组构造][src-array-create]和 [按设备拆分][src-array-disassemble]。

### 3.2 PJRT Buffer 管理哪些内容

`PjRtBuffer` 表示与某个设备及内存空间关联的数据存储，提供形状、布局、就绪状态和所有权接口。主机数据可以经 `BufferFromHostBuffer` 进入这套存储管理，已有数据可以通过复制接口转到其他内存空间，执行也会返回新的输出 Buffer。主机数据的接收方式由 `HostBufferSemantics` 等参数决定。参见 [`PjRtBuffer`][src-pjrt-buffer]、[接收主机数据][src-buffer-from-host]和 [存储复制][src-buffer-copy]。

编译期的输入输出别名关系和执行时的 donation 约定会影响存储复用。允许复用时，后端可以用输入存储承载输出；调用方需要遵守输入所有权被消费后的使用限制。释放 Buffer 时，`Delete` 先放弃它对设备存储的引用，实际内存还要等相关异步操作和外部引用结束后才能回收。参见 [执行约定][src-pjrt-execute]和 [`Delete`][src-buffer-delete]。

### 3.3 执行请求如何到达设备实现

JAX 的 `ExecuteReplicated.__call__` 先通过输入处理器取得数组，并按需要补充用于 effects 排序的 token，然后调用 `execute_sharded`。jaxlib 的 `PyLoadedExecutable::ExecuteSharded` 取出每个 `PyArray` 持有的 IFRT Array，调用 IFRT 的执行接口。参见 [JAX 执行入口][src-dispatch]和 [jaxlib 执行入口][src-py-execute]。

IFRT 接收到的是按数组组织的输入，而 PJRT 执行接口需要按设备上的计算组织 Buffer。比如两个输入数组都分布在两个设备上，IFRT 需要把“每个数组有哪些分片”整理为“每个设备本次计算需要哪些输入分片”。[`PjRtLoadedExecutable::Execute` 的 IFRT 实现][src-ifrt-execute]完成这个适配，并调用底层 PJRT 可执行对象。

| 层次 | 接收的执行输入 | 交给下一层的内容 |
|---|---|---|
| JAX / jaxlib | Python 数组、执行选项，以及需要的 effects 信息。 | IFRT Array 列表和执行选项。 |
| IFRT 的 PJRT 适配实现 | 逻辑数组及其设备分片。 | 按设备组织的 `PjRtBuffer` 参数列表。 |
| PJRT 的设备实现 | 已加载程序、输入 Buffer、执行选项和数据依赖。 | 具体的计算、通信和数据搬运工作。 |

这几步中的持有关系和参数重排可以复用现有存储。需要迁移数据时，才由对应实现安排复制。

### 3.4 运行时如何调度一次执行

设备实现结合可执行对象的存储规划和输入 Buffer，准备输出及临时空间，再按输入就绪情况和执行依赖组织工作。这里的“执行提交”包含程序、存储、参数和依赖，是一次运行的具体上下文。

CPU 路径中，`CpuPjRtRawLoadedExecutable::Execute` 构造 Buffer 表，再用 `Thunk::ExecuteParams` 传入函数库、实际存储、线程池和通信上下文等信息。`ThunkExecutor::Execute` 根据执行计划选择顺序执行或按依赖调度，调用生成的计算函数和库。参见 [CPU 执行入口][src-cpu-execute]、[执行上下文构造][src-cpu-submit]和 [`ThunkExecutor`][src-thunk-execute]。

GPU 路径由运行时和驱动组织 kernel 启动、设备库调用、通信与数据搬运。以 CUDA 为例，stream 组织工作顺序，event 可以表达不同 stream 之间的依赖；这些机制决定工作何时能够执行。参见 [CUDA 异步执行说明][doc-cuda-async]和 [运行提交详细图](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/research/software-stack/stream-executor/submission-centered-hub.svg)。

TPU 的公开调用入口可继续追到 [`PjRtCApiLoadedExecutable::Execute`][src-capi-execute]：它将按设备组织的输入转换为 `PJRT_Buffer` 参数列表，连同执行选项交给插件的 `PJRT_LoadedExecutable_Execute`。插件返回输出 Buffer；调用方请求完成状态时，还返回逐设备完成事件，C++ 包装层通过 `ConvertCEventToCppFuture` 将其转换为 Future。包装层准备的输出列表是句柄容器，设备存储的分配和实际提交仍由 provider 负责。

### 3.5 输出返回和数据就绪

PJRT 执行返回各设备的输出 Buffer。IFRT 按输出数组重新收集分片，用 `PjRtArray::Create` 加上类型、全局形状、分片和布局信息，放入 `ExecuteResult.outputs`。jaxlib 再通过 `PyExecuteResults` 和输出处理器构造 Python 可见的数组。参见 [IFRT 输出数组构造][src-ifrt-outputs]和 [`ConsumeWithHandlers`][src-py-outputs]。

```text
各设备的输出 PjRtBuffer
  → IFRT 按输出组织逻辑数组
  → jaxlib 输出处理器构造 PyArray
  → Python 调用取得 jax.Array
```

数组对象可以在设备完成计算前返回。它已经具有形状、类型和存储关联，因此后续计算可以继续把它作为输入，由运行时维护依赖。真正读取数据或等待结果时，才需要确认相关工作已经完成。

Buffer 的就绪情况通过 `GetReadyFuture()` 查询；PJRT 执行接口可以返回完成 future；IFRT 的 `ExecuteResult.status` 则只在 `fill_status=true` 时填充。计算失败也通过相应状态传播。源码入口分别是 [`GetReadyFuture`][src-ready]、[PJRT 执行接口][src-pjrt-execute]和 [`ExecuteResult`][src-ifrt-result]。

## 4. Pallas 内核如何接入外层程序

Pallas 允许在 JAX 函数中编写专用 kernel。外层程序组织数组之间的计算，kernel 描述某个计算块如何读写数据。它们都可以用 Jaxpr 表达，但承担的任务和 lowering 方式不同。

### 4.1 kernel Jaxpr 与外层 pallas_call

[`_trace_kernel_to_jaxpr`][src-kernel-trace] 根据 kernel 函数和抽象参数追踪内层 kernel Jaxpr。kernel 使用 Ref 表达数据读写，结果通常通过写入输出 Ref 产生；当前入口要求 kernel 函数返回 `None`。

[`get_grid_mapping`][src-grid-build] 构造的 `GridMapping` 记录执行网格、数据块映射和调用所需的信息。外层的 [`pallas_call_p.bind`][src-pallas-bind] 把 kernel Jaxpr、GridMapping 等作为方程参数，并用数组输入输出把这次调用接入外层数据流。内层程序和映射共同决定 kernel 如何作用于外层数组。

### 4.2 Mosaic TPU lowering 与 custom call

外层 Jaxpr lowering 遇到 `pallas_call` 时，由 [`_pallas_call_lowering`][src-pallas-lowering] 选择平台实现。在原生 Mosaic TPU 路径中，`pallas_call_tpu_lowering_rule` 根据 kernel Jaxpr 和 GridMapping 调用 `lower_jaxpr_to_pipelined_module`，生成独立的 Mosaic TPU MLIR Module。参见 [TPU lowering 规则][src-tpu-lowering]和 [Mosaic 模块构造][src-mosaic-module]。

随后，[`lower_module_to_custom_call`][src-custom-call] 组织模块处理和调用配置。序列化函数会克隆 Mosaic Module，运行 `mosaic-serde` pass，并输出 MLIR bytecode。这份内核程序连同相关配置进入外层 custom call 的 `backend_config`。[`_lower_mosaic_module_to_asm`][src-mosaic-serialize] 虽然名称中带有 `asm`，这里返回的是 MLIR bytecode。

外层构造的操作以 `tpu_custom_call` 为目标，接收外层数组对应的 IR 值，其结果继续供后面的外层操作使用。完整 Module 最后进入前述编译接口。参见 [custom call 操作构造][src-custom-emit]。

```text
kernel 函数 → kernel Jaxpr + GridMapping → Mosaic TPU Module
                                             ↓ 序列化内核程序与配置
外层 Jaxpr 中的 pallas_call → 外层 Module 中的 custom call
                                             ↓
                                  随外层程序提交后端编译
```

`backend_config` 中携带的是编译期的内核表示。运行时输入和输出仍由 Array、Buffer 以及执行接口传递。外层 StableHLO 计算与内层 Mosaic 程序在 custom call 处衔接；Mosaic TPU MLIR 也有自己的定义，不能称为 LLO。

### 4.3 平台选择与解释执行

Pallas 的 GPU 路径按配置选择 Mosaic GPU、Triton 或已注册的平台规则。`interpret` 模式则把 kernel 交给解释实现，再通过普通 JAX lowering 生成程序。当前 `pallas_call` 的 CPU 分支支持解释模式。这些分支由 [`_pallas_call_lowering`][src-pallas-lowering] 明确区分，阅读一次具体编译时要先确认使用了哪条路径。

更细的对象关系见 [Pallas 内外层程序说明](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/research/software-stack/jax/jaxpr-centered-hub.md#pallas)和 [Jaxpr 详细图](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/research/software-stack/jax/jaxpr-centered-hub.svg#inner_jaxpr)。

## 5. 源码版本和图的维护

### 5.1 本文使用的源码

JAX 固定在 `361c43e072cce92b7d3e9bdaf4dd16db26c49043`，XLA 固定在 `dcf304bc5dca1932b99f740b911dbd73631a1a69`。文中的源码链接指向这些固定提交，其他依赖以 [`upstream-sources.lock`](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/upstream-sources.lock)、[`source-archives.lock`](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/source-archives.lock) 和[环境锁](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/env/environment.lock.json)为准。

在线文档用于补充概念和使用方式，具体类、函数与调用顺序以工作区固定源码为准。对照实际运行结果时，应同时核对运行时二进制版本；与所引用源码不一致的结果标记为 `VERSION-SKEW`。

### 5.2 重建配套图

[分层图](/blog/jax-software-stack/overview-software-stack-components-layered.svg)由 [`render_overview_software_stack_flows.py`](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/tools/render_overview_software_stack_flows.py) 生成，节点和核心抽象的配置在 [`software_stack_overview_flow_data.py`](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/tools/software_stack_overview_flow_data.py) 中。[扩展接口图](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/research/software-stack/overview/overview-software-stack-components-extended.svg)由 [`render_overview_software_stack_extensions.py`](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/tools/render_overview_software_stack_extensions.py) 生成。

在仓库根目录运行：

```bash
python3 -B tools/diagram_environment.py run tools/render_overview_software_stack_flows.py
python3 -B tools/diagram_environment.py run tools/render_overview_software_stack_extensions.py
```

两个命令分别更新对应的 SVG。需要先预览时，可以在命令末尾加 `--output-dir /tmp/jax-stack-preview`。旧的 [`render_overview_software_stack_components.py`](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/tools/render_overview_software_stack_components.py) 命令入口仍转交分层图生成器。

生成器检查固定源码锚点和绘图结构。所需源码不在本地检出中时，使用 [`overview_flow_sources.json`](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/tools/overview_flow_sources.json) 记录的提交和 SHA-256 校验缓存；缺少缓存时，可用 `--fetch-sources` 获取相应文件。绘图依赖由 [`env/diagrams.lock.json`](https://github.com/0xaskr/jax-source-analysis/blob/113cb22d90a64f09fddf2e8ee77259efa4d18ad0/env/diagrams.lock.json) 锁定。

修改组件职责或调用关系时，需要同时检查正文、节点配置和图中的连线说明。正文使用组件名、对象名和函数名描述流程，可以单独阅读；图用于查看这些关系在整体架构中的位置。

[src-buffer-assignment]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/buffer_assignment.h#L476
[src-cpu-backend]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L2139
[src-cpu-execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L1602
[src-cpu-jit]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L764
[src-cpu-load]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L709
[src-cpu-module]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L820
[src-cpu-wrap]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L1037
[src-custom-call]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L839
[src-custom-emit]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L461
[src-dce]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L1204
[src-dispatch]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/pxla.py#L338
[src-eqn]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L460
[src-executable]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/executable.h#L263
[src-gpu-backend]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L2943
[src-grid]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/core.py#L946
[src-grid-build]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/core.py#L1268
[src-hlo]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_module.h#L95
[src-hlo-passes]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/pass/hlo_pass_pipeline.cc#L298
[src-ifrt-array]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/ifrt/array.h#L66
[src-ifrt-compile]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_client.cc#L412
[src-ifrt-execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_executable.cc#L835
[src-ifrt-outputs]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_executable.cc#L1106
[src-ifrt-result]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/ifrt/executable.h#L272
[src-ifrt-wrap]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_executable.cc#L798
[src-import]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc#L6232
[src-jaxpr]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L105
[src-kernel-trace]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L788
[src-lower]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L1327
[src-mosaic-module]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/mosaic/lowering.py#L1008
[src-pallas-bind]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L1358
[src-pallas-lowering]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/pallas_call.py#L845
[src-pjrt-buffer]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1108
[src-pjrt-compile]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_executable.cc#L763
[src-pjrt-execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1457
[src-pjrt-loaded]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1390
[src-py-array]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_array.h#L141
[src-py-compile]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_client.cc#L475
[src-py-execute]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_executable.cc#L484
[src-py-outputs]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_executable.cc#L260
[src-py-wrap]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jaxlib/py_client.cc#L430
[src-ready]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1380
[src-stablehlo]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/lib/mlir/dialects/__init__.py#L62
[src-subcomp]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/mlir.py#L2129
[src-tpu-lowering]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/pallas/mosaic/pallas_call_registration.py#L393
[src-trace]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L2133
[src-var]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/core.py#L539

[src-array-create]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_array.cc#L158
[src-array-disassemble]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_array.cc#L353
[src-buffer-copy]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1306
[src-buffer-from-host]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L966
[src-cpu-emit]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1882
[src-cpu-ir-passes]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/cpu/codegen/ir_compiler.cc#L354
[src-cpu-machine-code]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/cpu/codegen/ir_compiler.cc#L477
[src-cpu-submit]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L1809
[src-cpu-target]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/cpu/codegen/ir_compiler.cc#L255
[src-gpu-emit]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/compile_module_to_llvm_ir.cc#L212
[src-gpu-ir-passes]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc#L252
[src-llvm-module]: https://github.com/llvm/llvm-project/blob/75a45c373407c13a44c7abb28a78d891a97fe665/llvm/include/llvm/IR/Module.h#L67
[src-module-op]: https://github.com/llvm/llvm-project/blob/75a45c373407c13a44c7abb28a78d891a97fe665/mlir/include/mlir/IR/BuiltinOps.td#L33
[src-nvptx-binary]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/nvptx_compiler.cc#L583
[src-pjrt-program]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c/pjrt_c_api.h#L739
[src-stablehlo-add]: https://github.com/openxla/stablehlo/blob/7b1b15781ccbd770f50c7eef4b0c3e03834649fd/stablehlo/dialect/StablehloOps.td#L831
[src-thunk-execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/backends/cpu/runtime/thunk_executor.cc#L248
[doc-isa-evolution]: https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-a-profile-architecture-developments-2025

[src-jvp-jaxpr]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/ad.py#L1044
[src-batch-jaxpr]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/batching.py#L416
[src-partial-eval-jaxpr]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/interpreters/partial_eval.py#L655

[src-buffer-delete]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/pjrt_client.h#L1273

[src-hlo-program]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/ifrt/hlo/hlo_program.h#L39
[src-ifrt-compiler]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/python/pjrt_ifrt/pjrt_compiler.cc#L91
[src-cpu-hlo-create]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L989
[src-cpu-hlo-passes]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L596
[src-cpu-compile-body]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/cpu/cpu_compiler.cc#L1722
[src-gpu-executable-create]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L3038
[src-mosaic-serialize]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/tpu_custom_call.py#L491
[doc-llvm-objdump]: https://www.llvm.org/docs/CommandGuide/llvm-objdump.html
[doc-ptx]: https://docs.nvidia.com/cuda/parallel-thread-execution/
[doc-cuda-async]: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
[src-tpu-client]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/jax/_src/xla_bridge.py#L205
[src-capi-program]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L608
[src-capi-compile-call]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L644
[src-capi-compile]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L763
[src-capi-execute]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/c_api_client/pjrt_c_api_client.cc#L3421
[src-pallas-mosaic-design]: https://github.com/0xaskr/jax/blob/361c43e072cce92b7d3e9bdaf4dd16db26c49043/docs/pallas/design/design.md#L515
[issue-tpu-overview]: https://github.com/elbertwang/libtpu-agent/issues/76

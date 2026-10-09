---
title: "The JAX Software Stack: From JAX to TPU, CPU, and GPU"
description: "Follow JAX compilation, execution, and return paths through Jaxpr, MLIR, IFRT, PJRT, HLO, LLVM IR, and device runtimes, with a layered SVG and Pallas kernel integration."
pubDate: "2026-10-09"
tags: ["JAX", "XLA", "Pallas", "Compiler", "TPU", "CPU", "GPU"]
draft: false
---

> The author provided the outline, AI drafted the text, and the author edited and reviewed it. The author takes responsibility for the article's contents.
> This article is intended for readers with some experience developing with the JAX software stack. Beginners can use AI assistance to understand the concepts.

This article accompanies the [layered architecture diagram](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg). Read it alongside the diagram.

[![Layered JAX architecture: compilation, execution, and return paths](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg)](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg)

For readable labels, [open the full SVG](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg) and zoom in. Nodes, interface indexes, and source references are clickable. Linked detail diagrams in the source repository are in Chinese.

## 1. The core abstraction of each component

For each component, we choose one core abstraction and examine it from four perspectives: what it represents and how its data structures express it; how those structures are produced; what transformations they undergo; and which components consume them.

| Component | Core abstraction | Meaning | Lifecycle |
|---|---|---|---|
| JAX | [Jaxpr](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg#outer) | An IR for JAX/Pallas code, describing how primitives and their composition turn inputs into outputs. | Produced by [tracing][src-trace] a Python function with abstract inputs, then processed by transformations such as [automatic differentiation (e.g. JVP)][src-jvp-jaxpr], [batching (vmap)][src-batch-jaxpr], [partial evaluation][src-partial-eval-jaxpr], and [dead code elimination][src-dce]. JAX [lowering][src-lower] then applies primitive-specific rules to build the outer MLIR Module and submit it to the native compilation path. |
| jaxlib | [MLIR Module (StableHLO/Mosaic)](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg#binding_module) | An intermediate representation of the computation, using an [MLIR Module][src-module-op] as its container and [StableHLO][src-stablehlo-add], [Mosaic, and other specialized IRs][src-mosaic-module] to express its operations. | Produced from Jaxpr by [JAX][src-lower]/[Pallas lowering][src-pallas-lowering]. jaxlib wraps the MLIR Module in `ifrt::HloProgram`; IFRT calls PJRT to reach the compilation interface of the device backend. |
| IFRT | [Array](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg#ifrt_array) | A logical array that can span devices. [`ifrt::Array`][src-ifrt-array] uses `ArraySpec` to describe its element type, shape, sharding, and layout. | Device buffers holding the input data are [combined with type, shape, and sharding information into a logical array][src-array-create] for [LoadedExecutable execution][src-ifrt-execute]. On return, output buffers are [organized into new logical arrays][src-ifrt-outputs]. |
| PJRT | [PjRtBuffer](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg#pjrt_buffer) | A [uniform abstraction of data storage on a single device][src-pjrt-buffer], describing its device, memory space, shape, and layout, with interfaces for ownership and [readiness][src-ready]. | Created when receiving [input data][src-buffer-from-host] or preparing output storage; it can be [copied][src-buffer-copy] or reused as needed. Buffers are the execution inputs and outputs of [PjRtLoadedExecutable][src-pjrt-execute]. IFRT [organizes returned output buffers into logical arrays][src-ifrt-outputs]; when a buffer is no longer needed, its [storage reference is released][src-buffer-delete]. |
| XLA | [HLO (HloModule)](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg#hlo) | A computation graph for optimization, planning, and code generation. An [`HloModule`][src-hlo] contains an entry `HloComputation` and other computations; `HloInstruction` objects express computations and dependencies, with constraints such as shape, layout, and sharding. | Produced by [importing StableHLO][src-import], normalized and optimized for the target through an [HLO pass pipeline][src-hlo-passes], then passed to the [CPU][src-cpu-backend] or [GPU backend][src-gpu-backend] for storage planning, target code generation, and execution planning, producing the corresponding `Executable`. |
| XLA CPU backend | [LLVM IR](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg#cpu_ir) | A low-level intermediate representation for CPU code generation. It describes concrete computations, memory accesses, and control flow, together with target platform and data layout information. | Generated by the CPU backend from optimized HLO, storage plans, and target information. After [LLVM optimization][src-cpu-ir-passes], [target code generation][src-cpu-machine-code] produces object files. The linked function library and thunk execution plan form part of `CpuExecutable`. |
| XLA GPU backend | [LLVM IR](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg#gpu_ir) | A [low-level intermediate representation][src-llvm-module] for GPU kernel code generation. It describes computations, memory accesses, and control flow within kernels, using target-specific instructions and conventions for thread cooperation, address spaces, and kernel entry points. | Generated by the [GPU backend][src-gpu-emit] from optimized HLO, storage plans, and target device information. Device bitcode is linked as needed; [LLVM optimization][src-gpu-ir-passes] and target code generation produce device code. The [NVIDIA path][src-nvptx-binary] generates PTX and compiles it into cubin. Device code, storage plans, and the thunk execution plan form a [`GpuExecutable`][src-gpu-backend], which is returned to PJRT. |
| TPU TensorCore backend (libtpu) | [Native LLO](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg#tpu_boundary) | The TC backend's low-level program representation, describing device computation through regions, loops, instructions, values, local storage, and synchronization dependencies. | For ordinary HLO, the TC backend produces native LLO directly. For Mosaic TC kernels, it first produces MLIR llo, then converts it to native LLO. The TC backend then optimizes LLO, schedules instructions, allocates registers, and packs instruction bundles to generate a TC program. After packaging and loading, upper layers invoke execution through a PJRT executable. |
| Target assembly representation | [ASM](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg#asm) | A readable program representation using target instructions and operands to describe computation, data access, and control transfers. It can target a hardware ISA or a virtual ISA such as PTX. | Produced by target code generation, then assembled or [compiled further for the target][src-nvptx-binary] into machine code for linking and loading. It can also be obtained by disassembling existing machine code to inspect code generation and analyze performance. |
| Device runtime and driver | [Execution submission](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg#submit) | A request and [execution context][src-cpu-submit] for one program run, associating the loaded program, input/output storage, execution parameters, and dependencies: what to execute, which data to use, and when execution may proceed. | Enters the backend through the [PJRT execution interface][src-pjrt-execute]. The backend [prepares storage][src-cpu-execute] using the executable and input buffers, then [schedules computation, communication, and data movement][src-thunk-execute] according to the execution plan and dependencies. Results are written to output buffers, while [completion or error status][src-ready] propagates upward. |
| Instruction set interface and hardware | [ISA](/blog/jax-software-stack/en/overview-software-stack-components-layered-en.svg#isa) | The instruction-level semantic contract between software and hardware: instructions and their encodings, visible state such as registers, and the computation, memory access, and control-transfer behavior of execution. | Defined by processor architecture specifications and developed through [architecture versions and extensions][doc-isa-evolution]. The compiler generates [machine code][src-cpu-machine-code] using the target's supported [instructions and features][src-cpu-target]; hardware implements those instruction semantics, updating registers, memory, and control state. |

The TPU row focuses on the TensorCore (TC) backend, with native LLO as its core abstraction. HLO, covered in the XLA row, is an upstream program representation. SparseCore (SC) follows a separate MLO/LLVM path, shown as an independent branch in the diagram and text. Mosaic TPU MLIR, the MLIR `llo` dialect used during its lowering, and native LLO are distinct representations.

JAX and jaxlib sit in the center of the diagram. IFRT/PJRT, XLA/compiler backends, and ASM/device runtime appear side by side, with hardware at the bottom. This arrangement expresses responsibilities and call relationships. Vertical position does not mean every request must pass through every box in order.

Filled backgrounds and thick borders identify core nodes. Blue lines carry compilation and programs; green lines carry execution and input data; purple lines indicate program transformations. Dashed orange lines return compilation results, solid pink lines return output objects, and dashed pink lines report completion. Dashed gray lines describe structure, ownership, or observation; dashed brown lines mark the limits of evidence about TPU internals. Bridges at line crossings mean the lines are not connected.

## 2. Compilation: from JAX code to an executable program

A JAX call that requires compilation first lowers JAX code through intermediate representations, then generates code that the target device can execute.

We follow the path from JAX through jaxlib, IFRT, and PJRT into the device backend. The device backend implementation provides the compilation, loading, and execution interfaces for the particular device.

### 2.1 Tracing and function transformations: constructing Jaxpr

When tracing a Python function, JAX creates Tracers from abstract input information such as shape and dtype. These act as placeholders for dynamic inputs during the function call, while JAX records primitive operations in a Jaxpr. The resulting Jaxpr records inputs, constants, primitive equations, outputs, and effects. Each variable's `aval` describes its abstract type; equations specify how those variables participate in computation. See the definitions of [`Jaxpr`][src-jaxpr], [`JaxprEqn`][src-eqn], and [`Var`][src-var]. One tracing entry point is [`trace_to_jaxpr_dynamic`][src-trace].

Jaxpr supports JAX's function transformations, each of which addresses a different concern:

| Transformation | What it does to the program | Source entry point |
|---|---|---|
| Automatic differentiation | For example, JVP constructs tangent propagation and computation alongside the original function's computation. | [`jvp_jaxpr`][src-jvp-jaxpr] |
| Batching | Rewrites primitive calls according to the batch dimensions of inputs, producing a program that handles a batch of inputs. | [`batch_jaxpr`][src-batch-jaxpr] |
| Partial evaluation | Splits computation according to which inputs are known and records intermediate results that must pass between the two parts. | [`partial_eval_jaxpr_nounits`][src-partial-eval-jaxpr] |
| Dead code elimination | Removes unnecessary computations based on output usage and effects. | [`dce_jaxpr`][src-dce] |

These transformations can participate in tracing or consume an existing Jaxpr. Their composition depends on which JAX transformations the function uses. Automatic differentiation and batching change the computation being performed; dead code elimination primarily removes unused work.

### 2.2 Lowering: constructing an MLIR Module from Jaxpr

[`lower_jaxpr_to_module`][src-lower] establishes the module, function entry point, and input/output constraints. [`jaxpr_subcomp`][src-subcomp] traverses Jaxpr equations and selects lowering rules by primitive and target platform. A rule receives existing MLIR input values, emits operations, and passes the resulting values to subsequent equations.

The MLIR Module is a program container. Ordinary tensor computation is expressed mainly in StableHLO; accompanying dialects and attributes record information such as sharding. jaxlib supplies the bindings used to construct these objects. Pallas kernels also produce specialized representations, whose integration is discussed in section 4. See the [Module definition][src-module-op] and [StableHLO operation definitions][src-stablehlo-add].

After building the module, JAX invokes the MLIR verifier to check structural, type, and operation constraints. When Shardy is enabled, it also runs `sdy-lift-inlined-meshes` to organize mesh representations. These steps still operate on the MLIR Module. Both verification and module pass invocation appear in [`lower_jaxpr_to_module`][src-lower].

### 2.3 Passing compilation requests through jaxlib, IFRT, and PJRT

jaxlib's [`PyClient::CompileAndLoad`][src-py-compile] receives the Module and compilation options, clones the Module, and wraps it in `ifrt::HloProgram`. The name is easy to confuse with `HloModule`: the [`HloProgram`][src-hlo-program] here still holds an MLIR Module.

jaxlib then invokes IFRT's default compiler. The IFRT implementation discussed here works through PJRT: [`PjRtCompiler::CompileAndLoad`][src-ifrt-compiler] extracts the Module and compilation options, passes them through IFRT's executable construction path, and ultimately calls the device implementation's [`PjRtClient::CompileAndLoad`][src-pjrt-compile].

| Caller → receiver | What is passed | What the receiver does next |
|---|---|---|
| JAX → jaxlib | MLIR Module, device information, and compilation options. | Clones the Module and constructs `HloProgram` and IFRT compilation options. |
| jaxlib → IFRT | `HloProgram` and IFRT compilation options. | Enters the appropriate compilation implementation through the default Compiler. |
| IFRT → PJRT | Module extracted from the Program, plus XLA compilation options. | Calls the device-specific compilation and loading entry point. |

This part primarily handles interface adaptation, object ownership, and compilation configuration. CPU/GPU paths construct XLA's `HloModule` later, during import.

### 2.4 XLA: importing HLO, optimizing computation, and planning storage

An `HloModule` represents a complete XLA program, containing the entry `HloComputation` and other called computations. Each computation consists of `HloInstruction` objects and their dependencies. Shape, layout, sharding, and compilation configuration further constrain the program. The [`HloModule` definition][src-hlo] specifies the relationship between the module and its entry computation.

The CPU compilation entry first exports HLO through MLIR compilation preparation, then constructs a module with `HloModule::CreateFromProto`. `JitCompile` subsequently calls `CpuCompiler::RunHloPasses` and `RunBackend` for HLO transformations and backend compilation. See the [CPU Module compilation entry][src-cpu-module], [HloModule construction][src-cpu-hlo-create], and [`JitCompile`][src-cpu-jit].

HLO passes perform computation simplification, dead code elimination, fusion, sharding, and layout assignment. The backend chooses passes and their order according to the target device and compilation configuration. These transformations progressively determine which computations run together, how data is distributed, and how tensors are laid out in memory. See [`HloPassPipeline`][src-hlo-passes] for pass organization, and the [CPU compiler][src-cpu-hlo-passes] and [GPU compiler][src-gpu-backend] for concrete pipelines.

Scheduling determines the execution order of computations. Storage planning then assigns space to inputs, outputs, and temporary values. [`BufferAssignment`][src-buffer-assignment] records compile-time allocations and slices for both code generation and the runtime. Objects that actually allocate and own device storage are created at runtime.

### 2.5 CPU and GPU code generation

The CPU backend generates LLVM IR from optimized HLO, storage plans, and target information. Functions, basic blocks, and instructions in `llvm::Module` describe concrete operations, memory accesses, and control flow. CPU emitters generate computation functions, while `ThunkEmitter` organizes those functions and library calls into the work sequence the runtime will execute. The relevant construction is centered in [`CompileCpuExecutable`][src-cpu-compile-body].

Next, [`IrCompiler::RunIrPasses`][src-cpu-ir-passes] performs LLVM optimization, and [`EmitMachineCode`][src-cpu-machine-code] produces object code. The current JIT path links this code into a callable function library, which joins `BufferAssignment`, the thunk sequence, constants, and other information in `CpuExecutable`. A thunk represents a concrete piece of work, such as calling a computation function or library, copying data, or handling control flow. See the [CPU LLVM IR detail diagram](https://github.com/0xaskr/jax-source-analysis/blob/5be019f9a76cd51d5b89d98fa4bb212e02056f70/research/software-stack/xla/cpu-llvm-ir-centered-hub.svg) for the object relationships.

The GPU backend selects kernel generation paths or device library implementations for HLO operations. Computations requiring generated kernels pass through the corresponding lowering to GPU LLVM IR, where target-specific conventions express address spaces, kernel entry points, and thread operations. Device library calls become part of the thunk execution plan. [`CompileModuleToLlvmIr`][src-gpu-emit] is the entry point for storage planning, the code generation context, and thunk construction.

GPU LLVM IR links device bitcode as needed and undergoes [LLVM optimization][src-gpu-ir-passes]. On NVIDIA, [`NVPTXCompiler::CompileTargetBinary`][src-nvptx-binary] generates PTX and invokes a compilation provider to compile it into cubin. Device code, constants, buffer allocation information, and the thunk execution plan form a [`GpuExecutable`][src-gpu-executable-create]. See the [GPU IR detail diagram](https://github.com/0xaskr/jax-source-analysis/blob/5be019f9a76cd51d5b89d98fa4bb212e02056f70/research/software-stack/xla/gpu-ir-centered-hub.svg) for additional branches.

### 2.6 Assembly, machine code, and the ISA

Assembly describes a concrete program using target instructions and operands. It can be a program exchanged between compilation tools or a view obtained by disassembling existing machine code. In the NVIDIA path above, PTX is passed to a subsequent compiler. On CPU, `EmitMachineCode` can emit object code directly; assembly can then be inspected with a disassembler. See the [PTX compilation implementation][src-nvptx-binary] and [LLVM disassembly tool][doc-llvm-objdump].

The ISA specifies instruction encodings, operational semantics, and changes to software-visible state such as registers and memory. It is defined by the processor architecture and evolves through architecture versions and extensions. The compiler emits machine code using the target's supported instructions and features, and hardware implements their semantics. The CPU backend selects target capabilities through [`TargetMachine`][src-cpu-target]; [Arm's ISA update notes][doc-isa-evolution] illustrate architectural evolution. PTX defines a virtual ISA, which still requires translation into the target GPU's machine code. The [PTX specification][doc-ptx] defines this layer.

### 2.7 TPU TensorCore backend: producing and compiling native LLO

JAX's [`make_tpu_client`][src-tpu-client] loads and initializes the PJRT plugin in `libtpu.so` as needed and obtains a Client. Upper layers continue to use IFRT/PJRT interfaces; the provider implements TPU-specific compilation, loading, and execution.

When the program reaches [`PjRtCApiClient::CompileAndLoad`][src-capi-compile], `InitializeArgsAndCompile` serializes the compilation options and calls `SerializeProgram` to process the program. MLIR inputs are serialized according to the StableHLO and Shardy versions reported by the plugin; `XlaComputation` inputs are serialized as HLO proto. Program bytes and a format identifier form a `PJRT_Program`, submitted through the plugin's `PJRT_Client_Compile` entry point. On success, the returned `PJRT_LoadedExecutable` is wrapped in `PjRtCApiLoadedExecutable` and passed back to IFRT and jaxlib. This path carries compilation inputs and execution capability; the arrays for a particular run have not yet been supplied. See [program serialization][src-capi-program] and [compilation argument construction and invocation][src-capi-compile-call].

**TPU internals still use XLA's HLO class hierarchy.** According to [libtpu-agent's account of the internal source][issue-tpu-answer], programs use `HloModule`, `HloComputation`, and `HloInstruction`, with configuration carrying TPU layout, window, memory-space, and execution-thread information. The backend performs canonicalization, SPMD partitioning, layout assignment, fusion, asynchronous rewriting, scheduling, and storage planning. Stages may repeat or vary with compilation options.

TC low-level programs have two sources. For ordinary tensor HLO, an emitter produces native LLO directly. Pallas's `tpu_custom_call` carries Mosaic TPU MLIR, which Mosaic first lowers to the MLIR `llo` dialect; a bridge builder then incorporates it into the outer native LLO program. Both paths converge on **native LLO** and continue through low-level optimization, instruction scheduling, register and local-storage allocation, instruction-bundle packing, and code generation.

```text
Ordinary HLO ── emitter ────────────────────────────┐
                                                   ↓
                                              Native LLO → TC program
                                                   ↑
Mosaic TPU MLIR ── lowering → MLIR llo ── bridge ────┘

SC HLO / Mosaic SC ── MLO / sparse_core → LLVM IR → SC program
```

Native LLO describes TC programs through regions, loops, instructions, values, local storage, and synchronization dependencies. It is distinct from the Mosaic input and from the MLIR `llo` representation used in the Mosaic path. The pinned [Pallas design document][src-pallas-mosaic-design] confirms that Mosaic produces LLO; the distinction between the two LLO representations and the separate SC path come from the internal account above. By default, Mosaic TC kernels become part of the outer device program, while SC programs cooperate with TC programs within the executable.

HLO scheduling and runtime scheduling must also be distinguished. A complete compiler-generated [`HloSchedule`][src-hlo-schedule] specifies HLO instruction order within each non-fusion computation; [`BufferAssignment`][src-buffer-assignment] describes the compile-time storage plan. The target backend must still arrange lower-level instructions, registers, and DMA operations. At execution time, the provider binds actual buffers to the compiled program and submits device work. One HLO operation can lower to many low-level instructions, so an HLO sequence cannot be read as a cycle-by-cycle hardware instruction schedule.

See the [native LLO notes](https://github.com/0xaskr/jax-source-analysis/blob/5be019f9a76cd51d5b89d98fa4bb212e02056f70/research/software-stack/libtpu/llo-boundary-centered-hub.md) for component roles, object definitions, and their production, transformation, and consumption. The internal account provides neither a source revision tied to a wheel nor private class names; “emitter” and “bridge builder” describe responsibilities. The diagram and text distinguish this account from the pinned public source.

### 2.8 Returning compilation results to JAX

As compilation returns, each layer retains the objects and metadata needed for later execution. CPU provides an example that can be traced end to end: `CpuExecutable` is wrapped in `PjRtCpuExecutable`, and `LoadInternal` constructs the loaded `PjRtCpuLoadedExecutable`. See [CPU executable wrapping][src-cpu-wrap] and [loading][src-cpu-load].

IFRT then wraps the device implementation's `PjRtLoadedExecutable` and stores metadata such as output types, shapes, sharding, and layouts. jaxlib waits for IFRT's compilation result and constructs `PyLoadedExecutable`. JAX's `ExecuteReplicated` retains that object and uses it for subsequent execution. See [IFRT wrapping][src-ifrt-wrap], [Python wrapping][src-py-wrap], and [`ExecuteReplicated`][src-dispatch].

```text
The device backend generates and loads the program
  → PJRT loaded executable
  → IFRT LoadedExecutable
  → jaxlib PyLoadedExecutable
  → retained and reused by JAX's execution entry point
```

These wrappers retain execution capability and its metadata. Array results are produced only when the program is actually called.

## 3. Execution: from input arrays to output arrays

Execution requires both the compiled program and the data for the current call. The program determines computation and storage arrangements; input arrays supply actual values. The runtime combines them, handles dependencies, and submits work.

### 3.1 How IFRT Array organizes device data

In this path, a Python-visible device array holds an IFRT Array through `PyArray`. An IFRT Array represents a complete logical array, whose `ArraySpec` describes element type, global shape, sharding, and layout. A logical array may reside on one device or be sharded or replicated across several devices. “Logical” refers to the complete-array perspective; it does not require multiple devices or hosts. See [`PyArray`][src-py-array] and [`ifrt::Array`][src-ifrt-array].

The PJRT-backed IFRT implementation groups the device buffers addressable by the current process with array metadata. `PjRtArray::Create` takes that information and constructs the array; `DisassembleIntoSingleDeviceArrays` retrieves arrays organized by device. This assembly and disassembly primarily manipulates array views and storage references. Whether data is copied depends on the operation and its copy semantics. See [array construction][src-array-create] and [per-device disassembly][src-array-disassemble].

### 3.2 What PJRT Buffer manages

`PjRtBuffer` represents storage associated with a device and memory space, with interfaces for shape, layout, readiness, and ownership. Host data can enter this storage management system through `BufferFromHostBuffer`; existing data can move to another memory space through copy interfaces, and execution returns new output buffers. Parameters such as `HostBufferSemantics` determine how host data is accepted. See [`PjRtBuffer`][src-pjrt-buffer], [host buffer ingestion][src-buffer-from-host], and [storage copying][src-buffer-copy].

Compile-time input/output aliasing and runtime donation contracts affect storage reuse. Where reuse is allowed, a backend can place outputs in input storage; callers must respect usage restrictions after input ownership has been consumed. When releasing a buffer, `Delete` first relinquishes its reference to device storage. Actual memory reclamation must wait until the relevant asynchronous operations and external references are finished. See the [execution contract][src-pjrt-execute] and [`Delete`][src-buffer-delete].

### 3.3 How execution requests reach the device implementation

JAX's `ExecuteReplicated.__call__` obtains arrays through its input handler, adds tokens for ordering effects where needed, and calls `execute_sharded`. jaxlib's `PyLoadedExecutable::ExecuteSharded` extracts the IFRT Array held by each `PyArray` and invokes IFRT's execution interface. See the [JAX execution entry][src-dispatch] and [jaxlib execution entry][src-py-execute].

IFRT receives inputs organized by array, whereas PJRT's execution interface needs buffers organized by the computation on each device. For example, if two input arrays are both spread across two devices, IFRT must turn “which shards belong to each array” into “which input shards each device needs for this computation.” The [IFRT implementation of `PjRtLoadedExecutable::Execute`][src-ifrt-execute] performs that adaptation and invokes the underlying PJRT executable.

| Layer | Execution inputs received | What is passed onward |
|---|---|---|
| JAX / jaxlib | Python arrays, execution options, and any required effects information. | IFRT Array list and execution options. |
| IFRT's PJRT adapter | Logical arrays and their device shards. | `PjRtBuffer` argument lists organized by device. |
| PJRT device implementation | Loaded program, input buffers, execution options, and data dependencies. | Concrete computation, communication, and data movement. |

The ownership relationships and argument rearrangement in these steps can reuse existing storage. The relevant implementation schedules copies when data must move.

### 3.4 How the runtime schedules an execution

The device implementation prepares output and temporary storage from the executable's storage plan and input buffers, then organizes work according to input readiness and execution dependencies. “Execution submission” here includes the program, storage, parameters, and dependencies: the concrete context for one run.

On CPU, `CpuPjRtRawLoadedExecutable::Execute` constructs a buffer table, then passes the function library, actual storage, thread pool, communication context, and other information through `Thunk::ExecuteParams`. `ThunkExecutor::Execute` chooses sequential execution or dependency-based scheduling according to the execution plan, invoking generated computation functions and libraries. See the [CPU execution entry][src-cpu-execute], [execution context construction][src-cpu-submit], and [`ThunkExecutor`][src-thunk-execute].

On GPU, the runtime and driver organize kernel launches, device library calls, communication, and data movement. In CUDA, streams order work and events can express dependencies between streams; together they determine when work may execute. See the [CUDA asynchronous execution documentation][doc-cuda-async] and [execution submission detail diagram](https://github.com/0xaskr/jax-source-analysis/blob/5be019f9a76cd51d5b89d98fa4bb212e02056f70/research/software-stack/stream-executor/submission-centered-hub.svg).

The public TPU call path can be followed into [`PjRtCApiLoadedExecutable::Execute`][src-capi-execute]. It converts inputs organized by device into `PJRT_Buffer` argument lists and passes them, together with execution options, to the plugin's `PJRT_LoadedExecutable_Execute`. The plugin returns output buffers. When the caller requests completion status, it also returns per-device completion events, which the C++ wrapper converts into Futures through `ConvertCEventToCppFuture`. The output lists prepared by the wrapper are handle containers; allocating device storage and submitting the actual work remain the provider's responsibility.

According to the [internal account][issue-tpu-answer], the TPU runtime gathers input dependencies, prepares output and temporary storage using the storage and aliasing plan, binds actual buffer addresses to the loaded program, and submits it to the device. The device program drives loops, DMA, computation, and waits. The host normally does not interpret individual HLO instructions, although host callbacks or offloaded work still use the corresponding host paths. Completion of one DMA operation, the return of output objects, and per-device execution completion are distinct observation points.

### 3.5 Returning outputs and establishing readiness

PJRT execution returns output buffers for each device. IFRT regroups the shards by output array, uses `PjRtArray::Create` to attach type, global shape, sharding, and layout information, and places the results in `ExecuteResult.outputs`. jaxlib then constructs Python-visible arrays through `PyExecuteResults` and output handlers. See [IFRT output array construction][src-ifrt-outputs] and [`ConsumeWithHandlers`][src-py-outputs].

```text
Output PjRtBuffers from each device
  → IFRT organizes logical arrays by output
  → jaxlib output handlers construct PyArray
  → the Python call receives jax.Array
```

An array object can return before the device has finished computing. It already carries shape, type, and storage associations, so subsequent computations can use it as an input while the runtime maintains dependencies. The relevant work must be confirmed complete when reading the data or explicitly waiting for the result.

Buffer readiness is queried through `GetReadyFuture()`. The PJRT execution interface can return a completion future; IFRT fills `ExecuteResult.status` only when `fill_status=true`. Computation failures propagate through the corresponding status mechanisms. See [`GetReadyFuture`][src-ready], the [PJRT execution interface][src-pjrt-execute], and [`ExecuteResult`][src-ifrt-result].

## 4. Integrating Pallas kernels into the outer program

Pallas lets a JAX function contain specialized kernels. The outer program organizes computation between arrays, while a kernel describes how a block of computation reads and writes data. Both can be expressed as Jaxpr, but their roles and lowering paths differ.

### 4.1 The kernel Jaxpr and the outer pallas_call

[`_trace_kernel_to_jaxpr`][src-kernel-trace] traces the inner kernel Jaxpr from the kernel function and abstract arguments. Kernels use Refs to express data reads and writes, normally producing results by writing output Refs. The current entry point requires the kernel function to return `None`.

The `GridMapping` constructed by [`get_grid_mapping`][src-grid-build] records the execution grid, block mappings, and information required for the call. The outer [`pallas_call_p.bind`][src-pallas-bind] carries the kernel Jaxpr, GridMapping, and other values as equation parameters, connecting the call to the outer dataflow through array inputs and outputs. Together, the inner program and its mapping determine how the kernel operates on the outer arrays.

### 4.2 Mosaic TPU lowering and custom calls

When outer Jaxpr lowering encounters `pallas_call`, [`_pallas_call_lowering`][src-pallas-lowering] selects the platform implementation. In the native Mosaic TPU path, `pallas_call_tpu_lowering_rule` invokes `lower_jaxpr_to_pipelined_module` with the kernel Jaxpr and GridMapping, producing a separate Mosaic TPU MLIR Module. See the [TPU lowering rule][src-tpu-lowering] and [Mosaic module construction][src-mosaic-module].

Next, [`lower_module_to_custom_call`][src-custom-call] organizes module processing and call configuration. Serialization clones the Mosaic Module, runs the `mosaic-serde` pass, and emits MLIR bytecode. This kernel program and its configuration enter the outer custom call's `backend_config`. Despite `asm` in its name, [`_lower_mosaic_module_to_asm`][src-mosaic-serialize] returns MLIR bytecode here.

The outer operation targets `tpu_custom_call`, receives IR values corresponding to outer arrays, and produces results used by later outer operations. The complete Module eventually reaches the compilation interface described earlier. See [custom call operation construction][src-custom-emit].

```text
Kernel function → kernel Jaxpr + GridMapping → Mosaic TPU Module
                                               ↓ serialize kernel and config
pallas_call in the outer Jaxpr → custom call in the outer Module
                                               ↓
                            submitted with the outer program for compilation
```

`backend_config` carries a compile-time kernel representation. Runtime inputs and outputs still travel through Array, Buffer, and execution interfaces. Outer StableHLO computation and the inner Mosaic program meet at the custom call. Mosaic TPU MLIR has its own definition and must not be called LLO.

### 4.3 Platform selection and interpretation

Pallas GPU paths select Mosaic GPU, Triton, or registered platform rules according to configuration. In `interpret` mode, the kernel goes through an interpreter implementation and then ordinary JAX lowering. The current CPU branch of `pallas_call` supports interpretation. [`_pallas_call_lowering`][src-pallas-lowering] explicitly distinguishes these branches; identify the path in use when examining a particular compilation.

For more detailed object relationships, see the [Pallas inner and outer program notes](https://github.com/0xaskr/jax-source-analysis/blob/5be019f9a76cd51d5b89d98fa4bb212e02056f70/research/software-stack/jax/jaxpr-centered-hub.md#pallas) and [Jaxpr detail diagram](https://github.com/0xaskr/jax-source-analysis/blob/5be019f9a76cd51d5b89d98fa4bb212e02056f70/research/software-stack/jax/jaxpr-centered-hub.svg#inner_jaxpr).

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
[src-hlo-schedule]: https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/ir/hlo_schedule.h#L153
[issue-tpu-answer]: https://github.com/elbertwang/libtpu-agent/issues/76#issuecomment-6064434151
[issue-tpu-followup]: https://github.com/elbertwang/libtpu-agent/issues/76#issuecomment-6064481581
[issue-tpu-corrections]: https://github.com/elbertwang/libtpu-agent/issues/76#issuecomment-6064767267

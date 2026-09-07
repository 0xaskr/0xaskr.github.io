---
title: 'What Are We Really Talking About When We Talk About Performance Optimization?'
description: 'A guide to the principles behind GPU, TPU, and accelerator performance: measurement, memory hierarchies, tiling, pipelines, parallelism, and the Roofline model.'
pubDate: '2026/9/7'
tags: ["Performance", "GPU", "TPU", "Kernel", "Profiling"]
draft: true
---

For people new to parallel computing and deep learning acceleration, performance optimization can seem mysterious: a program that once took days suddenly finishes in minutes after a few changes. What is the logic behind that transformation?

Performance optimization is not magic. Whether the hardware is a GPGPU, a domain-specific accelerator (DSA), or an ASIC, **the objective is the same: have the hardware complete the required computation in as little time as possible, then look for ways to do it with fewer operations and fewer resources.**

This article takes a general accelerator architecture perspective to explain the thinking behind performance optimization in heterogeneous computing.

---

## 1. No Measurement, No Optimization

Before discussing any particular optimization technique, establish one rule:

> **Do not guess where the bottleneck is. Measure it.**

If you cannot see what happens inside a program, you are like a pilot flying without instruments. Intuition may keep you moving, but it can easily take you off course.

### 1.1 Four Pillars of Observability

To make a program observable, engineers use four complementary approaches:

| Approach | How it works | Typical use |
| --- | --- | --- |
| **Logging** | Records individual events with timestamps | Records when an accelerator starts a convolution. Easy to understand, but potentially large in volume, with collection overhead of its own. |
| **Metrics** | Aggregates observations into time series | Tracks average utilization, bandwidth use, or power consumption over the past minute. |
| **Profiling** | Locates hotspots through sampling or instrumentation | Uses a flame graph to show which operators or lines of code consume the most time. |
| **Tracing** | Records execution as time spans | Shows on a timeline whether data transfers and computation overlap. |

**Profiling and tracing are two of the most useful tools for performance optimization.** Profiling tells you where time is spent; tracing shows how the stages relate in time.

### 1.2 Measure First, Optimize Second

> "We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil." — Donald Knuth

Often, 80% of performance problems come from 20% of the code. Before changing anything:

1. **Run a profile** to locate the most expensive **hotspots**.
2. **Weigh the return against the effort**: an optimization often adds complexity, so consider both the expected gain and the maintenance cost.
3. **Establish a baseline**: record the current performance, or you will have no way to judge whether a change helps.

### 1.3 The APOD Cycle

Performance optimization is an iterative process:

1. **Assess**: use a profiler to identify hotspots and Amdahl's law to estimate the maximum possible speedup.
2. **Parallelize**: move the expensive work to the accelerator, either through an existing library or a custom kernel.
3. **Optimize**: tune memory access, execution configuration, and data movement after introducing parallelism.
4. **Deploy**: validate the result in production. **You do not need to finish every optimization before deploying**; iterate and capture gains along the way.

After deployment, measure again, identify the next bottleneck, and repeat.

---

## 2. Understanding Accelerators: The Basics of Heterogeneous Computing

### 2.1 Why Do We Need Accelerators?

A conventional CPU uses a relatively small number of powerful cores and handles **complex serial logic and tasks that require low latency** well. As demand grows in deep learning, scientific simulation, and large-scale data processing, serial execution becomes a bottleneck.

Accelerators take a different approach: **use large-scale parallelism to maximize throughput**.

| Characteristic | General-purpose CPU | Accelerator: GPGPU / DSA / ASIC |
| --- | --- | --- |
| Core count | A few to a few dozen cores | Hundreds or thousands of cores, or many specialized compute units |
| Individual core capability | Powerful cores with complex pipelines, large caches, and branch prediction | Many simpler units: ALUs, systolic arrays, or specialized logic |
| Thread or task model | Heavyweight threads with expensive context switches | Lightweight execution units with almost no switching overhead |
| Suitable workloads | General-purpose logic, branching, and low-latency tasks | Large-scale data parallelism, high throughput, and regular computation |
| Latency hiding | Large caches and out-of-order execution | Many concurrent tasks and fast switching |
| Memory organization | Large, unified memory | Distinct levels of on-chip and off-chip memory |

**The central difference** is that a CPU aims to **minimize the latency of an individual task**, while an accelerator aims to **maximize the number of tasks completed per unit of time: throughput**.

Different accelerators make different tradeoffs:

- **GPGPUs**, such as NVIDIA and AMD GPUs, offer broad flexibility through large numbers of threads and a SIMT architecture.
- **DSAs**, such as Google TPUs and other AI accelerators, use specialized data paths for domains such as matrix computation or Transformer inference, trading flexibility for efficiency.
- **ASICs**, such as Bitcoin mining and video codec chips, optimize hardware for a narrow task, with very high efficiency and little general-purpose flexibility.

Despite their different implementations, **they share many performance challenges and optimization principles**.

### 2.2 Which Tasks Belong on an Accelerator?

Not every calculation benefits from offloading. A suitable task typically has:

1. **Large-scale data parallelism**: the same operation runs independently over thousands of elements, as in matrix operations, image processing, or vectorized computation.
2. **Enough computation per element** to justify host–device transfers. Matrix multiplication requires O(N³) operations but only O(N²) data transfer, so the benefit grows with matrix size.
3. **Regular access patterns**: neighboring execution units access neighboring memory locations, making good use of bandwidth.

**A counterexample** is sending two small matrices to a device, adding them, and copying the result back. Both the computation and the transfer scale as O(N²); moving the data may cost more than computing the answer.

---

## 3. Memory Hierarchies: The Cost of Accessing Data

To get the most out of an accelerator, work with its architecture. **Its compute units are fast, but without a steady supply of data, they sit idle.**

### 3.1 The Memory Pyramid

Architectures differ. GPUs commonly have an on-chip L2 cache, while DSAs such as TPUs often maintain a large shared on-chip SRAM or local memory. Yet they generally follow the same hierarchy: memory closer to the compute units is faster and smaller.

```text
Registers / register file
  Fastest; very small; private to a compute unit or systolic array
    ↓
Local on-chip cache / SRAM
  Very fast; shared within a thread group or compute unit
  Examples: shared memory, scratchpad, L1
    ↓
Chip-wide cache / SRAM
  Fast; shared across the chip
  Examples: GPU L2, DSA global SRAM
    ↓
Off-chip device memory
  Slower; HBM / GDDR / DDR; large and accessible to all units
    ↓
Host memory
  Slowest to reach from the device; accessed across PCIe / NVLink / CXL
```

The main points are:

- **On-chip storage**, including registers, shared memory, scratchpads, and SRAM, is fast but scarce.
- **Off-chip memory**, such as HBM and GDDR, offers more capacity but can take hundreds of cycles to access.
- **Host–device bandwidth** is often one or two orders of magnitude lower than device memory bandwidth.

> **Memory optimization means reducing accesses to slow storage and keeping data as close to the compute units as possible.**

### 3.2 Coalesced Access: Moving Data in Groups

When off-chip accesses are necessary, memory controllers favor **grouped transfers**.

On a GPGPU, a group of threads, such as an NVIDIA warp or an AMD wavefront, issues memory requests together. If those requests refer to **contiguous addresses**, the hardware can combine them into a larger memory transaction. This is **coalesced memory access**.

Strided or scattered access produces many fragmented transactions and can sharply reduce bandwidth utilization.

General principles include:

- Have neighboring execution units access neighboring addresses.
- Prefer a **structure of arrays (SoA)** to an **array of structures (AoS)** so that values of the same field are contiguous.
- Account for row-major or column-major storage when accessing multidimensional data, and make the innermost loop use a stride of one.

A DSA or ASIC may not have a warp abstraction, but its DMA engines and on-chip buffers also favor aligned, contiguous, sufficiently large transfers. **Regular, grouped data movement is useful across architectures.**

### 3.3 Conflicts in On-Chip Memory

Many accelerators divide fast on-chip memory, such as GPU shared memory, into **banks** to support concurrent accesses. Think of them as service counters:

- Requests to **different counters** can proceed in parallel.
- Requests to **different locations at the same counter** must wait, creating a **bank conflict** and reducing effective bandwidth.

**Optimization principle**: arrange data carefully in on-chip memory, for example through padding, so that concurrent accesses map to different banks.

---

## 4. Hiding Latency: Shopping While Cooking

If computation is cooking and data movement is shopping, a sequential schedule is wasteful: buy ingredients, wait for them to arrive, cook, then leave to buy the next batch. Much of the time is spent waiting.

### 4.1 Tiling

Fast on-chip storage is too small to hold an entire large matrix. **Tiling** divides a large iteration or data space into smaller pieces that fit in fast memory. Processing one tile at a time improves **temporal locality** and **data reuse**.

This idea appears across architectures:

- A GPGPU loads a matrix tile into shared memory.
- A DSA such as a TPU loads a tile into an on-chip accumulator buffer.
- An ASIC may incorporate a tiling strategy into its hardware scheduling logic because its buffer sizes are fixed.

### 4.2 Asynchronous Execution and Pipelines

Accelerators commonly have **independent transfer and compute engines** that can work at the same time. This enables **double buffering**, also called ping-pong buffering:

1. The transfer engine loads the first batch into **Buffer A**.
2. While the compute engine processes Buffer A, the transfer engine loads the second batch into **Buffer B**.
3. The compute engine switches to Buffer B while the transfer engine refills Buffer A.
4. The two buffers alternate, forming a pipeline.

By shopping while cooking, data movement can be **fully hidden behind computation**. In a trace, the transfer and compute spans overlap on the timeline.

### 4.3 Minimize Host–Device Transfers

Device memory bandwidth is much higher than host–device bandwidth. Typical orders of magnitude are:

| Transfer path | Typical bandwidth |
| --- | --- |
| Off-chip device memory: HBM2 | ~900 GB/s |
| PCIe Gen4 x16 | ~32 GB/s |
| PCIe Gen3 x16 | ~16 GB/s |

The gap can reach **30–60×**. Therefore:

- **Keep intermediate results on the device**. Even if the CPU can execute a particular step faster, leave it on the accelerator when the transfer cost exceeds the compute savings.
- **Combine small transfers**. Every transfer has a fixed cost; one larger request is usually more efficient than many small ones.
- **Overlap asynchronous transfers with computation** using independent DMA engines.

---

## 5. Maximizing Parallel Utilization

Different architectures emphasize two approaches: **many concurrent threads sharing execution resources over time**, as in GPGPUs, and **parallel computation distributed across spatial arrays**, as in TPUs and other DSAs.

### 5.1 Many Concurrent Threads: The GPGPU Approach

A central GPU design principle is to **hide latency with concurrency rather than relying on large caches**.

When one group of threads waits for memory, the hardware scheduler switches to another group at almost no cost. With enough work ready to execute, the compute units stay busy. **Occupancy** and utilization help assess this mechanism, while each task's on-chip resource requirements limit the number of execution contexts that can remain resident.

### 5.2 Spatial Parallelism: 1D and 2D Computation on DSAs and TPUs

Unlike a GPU with thousands of active threads, a DSA such as a TPU **does not use the warp or wavefront model**. It has few hardware instruction streams and much less hardware scheduling machinery. It relies on **wide parallel compute arrays and static software or compiler scheduling** to hide latency and improve throughput:

- **1D vector parallelism**: a wide vector ALU applies one arithmetic instruction to hundreds or thousands of elements at once.
- **2D matrix parallelism through a systolic array**: data flows through a two-dimensional network of interconnected ALUs and accumulates results in step with the clock. A large matrix operation produces many results across the array in parallel.

For these architectures, utilization depends on **matrix tiles that fit the systolic array**, such as a 128×128 layout, together with a carefully scheduled software pipeline that keeps the array supplied with data.

### 5.3 Multiple Levels of Parallelism

Parallelism exists at several levels:

- **Application level**: independent kernels can execute concurrently, and transfers can overlap with computation.
- **Device level**: the chip contains multiple compute modules, such as SMs, compute units, or core clusters, all of which need work.
- **Compute-module level**: enough execution contexts must remain resident to cover latency.
- **Instruction level**: compilers and hardware exploit instruction-level parallelism (ILP) by overlapping instructions in a pipeline.

---

## 6. Instruction and Control-Flow Optimization

Once memory bottlenecks are addressed, the next step is to make the compute units themselves more efficient.

### 6.1 Branch Divergence and Control-Flow Overhead

The penalties differ across architectures, but unpredictable control flow is costly:

- **GPGPU divergence**: under SIMT, threads in a warp execute together. If they take different `if` and `else` paths, the hardware executes the active paths serially, masking out threads that do not follow each path and leaving some capacity idle.
- **TPU and DSA control-flow bottlenecks**: these architectures do not use SIMT branch divergence. Their lower-level control logic can be much simpler, such as instruction queues or VLIW execution, and handles dynamic branching poorly. Complex branches can disrupt a pipeline and may even require intervention from the host CPU.

**Optimization principle**: eliminate dynamic branches where practical. For fine-grained conditions, consider computing both paths and selecting or masking the result. When conditions cannot be removed, operate at a coarser granularity with aligned shapes rather than splitting the work into many small pieces.

### 6.2 Loop Unrolling and Reordering

- **Loop unrolling** repeats the loop body to reduce loop-control overhead and give the compiler more freedom to schedule instructions.
- **Loop reordering** changes the order of nested loops so that the innermost accesses are contiguous, improving cache locality.

Although these may appear to be compiler responsibilities, explicit structure and hints can make a substantial difference in accelerator code.

### 6.3 Choosing Arithmetic Instructions

- **Prefer high-throughput instructions**. Single-precision operations are often several times faster than double-precision operations.
- **Use special-function hardware where suitable**. Accelerators may offer fast approximations for functions such as `sin`, `cos`, and `exp`, with a tradeoff in precision.
- **Use half or mixed precision**. For workloads such as deep learning that tolerate some numerical error, FP16 or BF16 can process twice as much work per instruction and benefit further from Tensor Cores or matrix engines.

### 6.4 Batch Parallelism

If one task is too small to fill the accelerator, combine multiple small tasks into a larger batch. Increasing batch size gives more compute units useful work and improves utilization.

---

## 7. Floating-Point Precision

Numerical work on accelerators, especially AI training and scientific computing, requires careful tradeoffs between precision and performance.

### 7.1 Precision Versus Performance

- **FP32** is a common reference for peak floating-point throughput.
- **FP64** can run at only 1/32 of FP32 throughput on some consumer GPUs.
- **FP16 and BF16**, designed for workloads such as deep learning, can deliver 2–8× FP32 throughput.
- Compare results with a **tolerance**, rather than requiring exact equality.

### 7.2 Floating-Point Addition Is Not Associative

**(A + B) + C ≠ A + (B + C)**

This is an inherent property of IEEE 754 floating-point arithmetic. **Parallel computation changes the order of operations**: a parallel reduction, for example, groups and sums elements differently across threads, producing different intermediate rounding errors. Small differences from serial execution are normal.

### 7.3 Fused Multiply-Add

Many accelerators support **FMA**, which computes `a × b + c` in one instruction with **a single rounding step**. It is faster and more accurate than separate multiplication and addition, but the result may differ slightly from a CPU implementation that performs the two operations separately.

---

## 8. Performance Metrics: Bandwidth and Computation

### 8.1 Theoretical Versus Effective Bandwidth

**Theoretical bandwidth** follows from hardware specifications: clock rate, data width, and channel count.

**Effective bandwidth** comes from what the program actually does:

**Effective bandwidth = bytes actually read and written / execution time**

**When effective bandwidth is far below the theoretical limit, memory access patterns are a primary area to investigate.**

### 8.2 The Roofline Model

The **Roofline model** is a standard way to distinguish a compute-bound operator from a memory-bound one:

- **Arithmetic intensity** is `FLOPs / Bytes`.
- Below the balance point, an operator is **memory-bound**: prioritize memory access efficiency.
- Above the balance point, it is **compute-bound**: prioritize instruction efficiency and compute utilization.

The shape of the roofline differs across accelerators, but the reasoning is the same.

### 8.3 Common Tools

| Tool | Typical use |
| --- | --- |
| **NVIDIA Nsight Systems** | System-level CPU–GPU interaction, stream timing, and transfer–compute overlap |
| **NVIDIA Nsight Compute** | Kernel-level occupancy, memory throughput, instruction throughput, and warp state |
| **AMD ROCProfiler / Omniperf** | Kernel- and system-level analysis on AMD GPUs |
| **Intel VTune / oneAPI** | Analysis of Intel GPUs and accelerators |
| **Vendor-specific profilers** | Toolchains for DSAs and ASICs |
| **Flame graphs** | General visualization of call stacks and time consumption |

---

## 9. The Path from 1 to 90

Optimization is gradual. A typical accelerator kernel, such as a convolution or matrix multiplication, progresses through these stages:

| Stage | What changes | Expected effect |
| --- | --- | --- |
| **Baseline** | Implement the correct computation | Performance may be only a few percent of peak |
| **Tiling** | Divide the work into tiles and use on-chip storage | Substantially improve data reuse |
| **Pipelining** | Add asynchronous execution and double buffering | Overlap transfers and computation |
| **Conflict removal** | Adjust layouts and reduce bank conflicts and divergence | Remove hidden hardware bottlenecks |
| **Final tuning** | Use reduced precision, sparsity, or specialized units | Approach peak hardware performance |

At every stage, **profiling data** determines what to optimize next.

---

## 10. Recap

When we talk about performance optimization, we are talking about:

- **Using measurements**: profile before optimizing.
- **Keeping the hardware supplied with data**: understand the memory hierarchy and reduce slow accesses.
- **Hiding latency**: overlap transfers and computation with asynchronous pipelines.
- **Maximizing parallelism**: provide enough work to keep compute units busy.
- **Understanding the hardware**: architectures differ, but memory, parallel utilization, and instruction efficiency matter across them.
- **Iterating**: every round begins with measurement.

These principles provide a starting point whether the target is an NVIDIA GPU, an AMD GPU, a Google TPU, or a new AI accelerator.

### Further Reading

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) — NVIDIA's guide to GPU performance optimization.
- [CUDA C++ Programming Guide — Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#performance-guidelines) — Performance guidance in the CUDA programming guide.
- [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model) — The distinction between compute-bound and memory-bound workloads.
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/) — Kernel-level performance analysis.

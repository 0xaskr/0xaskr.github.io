---
title: 'What Do We Mean by TPU/DSA Performance Optimization? A Guide for GPU Kernel Developers'
description: 'A GPU kernel developer’s guide to TPU performance: CUDA concept mappings, shape alignment, software pipelines, memory access, SPMD, and detailed profiling.'
pubDate: '2026/9/7'
tags: ["TPU", "Pallas", "Kernel", "Performance", "Profiling"]
draft: true
---

Developers used to tuning kernels in CUDA and on GPU hardware can encounter counterintuitive bottlenecks when moving to a TPU (Tensor Processing Unit) or a similar domain-specific accelerator, such as a DTU. This article helps GPU kernel developers build intuition for these systems, from their overall architecture to low-level LLO optimization.

---

## 1. The Overall Approach: Narrow the Search Space

From a search-space perspective, the TPU's performance sweet spot is narrow, much narrower than that of a GPU. The core idea is to **minimize the complexity and uncertainty presented to the hardware**.

A GPU relies on dynamic latency hiding through extensive warp scheduling; a TPU relies on static planning. To put an operation in the TPU's sweet spot, reduce complexity layer by layer, from the framework down to the compute library. For example, padding at the framework level can remove dynamic variable-length behavior before it reaches a kernel. A simpler instruction stream makes it easier to achieve high utilization.

---

## 2. A CUDA-to-TPU Concept Map

One way to understand a TPU is to map its components to familiar CUDA concepts:

| CUDA / GPU concept | TPU / DSA counterpart | Key difference |
| --- | --- | --- |
| **Streaming Multiprocessor (SM)** | **TPU Core** | A TPU core is more independent and typically follows a single instruction stream. |
| **Tensor Core** | **MXU: Matrix Multiply Unit** | A TPU MXU uses a large array, 128×128 or larger, and favors substantial matrix workloads over small fragments. |
| **Shared Memory** | **VMEM / SPMA: SRAM** | Both are explicitly managed local memory. On a TPU, the compiler schedules access statically, without the GPU-style bank-conflict problem. |
| **Registers** | **VR: vector registers / VACC: accumulator registers** | Register roles are specialized: input vectors reside in VR, matrix computation runs on the MXU, and accumulated results reside in VACC. |
| **Warp Scheduler: dynamic latency hiding** | **VLIW and software pipelining: static latency hiding** | A TPU relies on a compiler-generated instruction schedule rather than hardware thread-context switching. |
| **NCCL / NVLink** | **ICI: Inter-Core Interconnect** | A dedicated ICI network directly connects TPU chips in a 2D or 3D torus, with low latency and no host involvement. |
| **Triton / CUDA C++** | **JAX Pallas / LLO** | Pallas fills a role similar to Triton on TPUs, with explicit tiling and DMA scheduling. |

---

## 3. Align Work with the Systolic Array

The **systolic array** is central to TPU and DSA matrix computation. Making full use of it requires appropriate data sizes and layouts.

### 3.1 Spatial Alignment and Hidden Padding Costs

A GPU can accommodate a small 127×127 matrix through warp scheduling. A TPU's rigid array pads such a matrix to 128×128, spending compute resources and memory bandwidth on zeros.

- **Rule**: make tensor dimensions, such as batch size, hidden size, and sequence length, multiples of **128, 32, or 8**.

### 3.2 Larger Shapes and Higher CMAR

A GPU can perform well at small shapes, while a TPU needs substantial work to reach high utilization.

- **Empirical recommendation**: for a matrix multiplication `[M, K] × [K, N]`, use **M ≥ 512, K ≥ 1024, and N ≥ 512**.
- **Dimension priority**: **N ≥ K > M**. Larger workloads improve utilization of both the MXU and the 1D vector engine.

---

## 4. Hiding Latency: From Occupancy to Software Pipelines

This is a major change in intuition for GPU developers. In CUDA, higher **occupancy** can hide latency: when one warp stalls, the hardware switches to another.

**A TPU has no equivalent hardware warp scheduler or thread-context switching.**

### 4.1 Software Pipelining

A TPU hides latency through **instruction scheduling, performed manually or by the compiler**.

- **Asynchronous DMA**: while the MXU computes tile `i`, the DMA engine should already be loading tile `i+1` into VMEM.
- **VLIW packing**: an instruction bundle combines work such as `Vector Load`, `Scalar Add`, and `Matrix Multiply` so that different units operate together.
- **Loop unrolling**: beyond reducing loop overhead, unrolling gives the scheduler more instructions with which to arrange asynchronous loads and keep the pipeline moving.

---

## 5. Memory Access and the Innermost Two Dimensions

TPU memory operations impose rigid constraints, so the implementation needs to fit them.

### 5.1 Protect the Innermost Two Dimensions

Where possible, avoid complex `transpose`, `gather`, or `scatter` operations on the **innermost two dimensions** of a tensor.

- **Prefer linear copies**, which are the most efficient transfer pattern.
- **Express higher-dimensional transformations through address arithmetic** in the scalar engine where possible, rather than physically rearranging data.
- Operations on the innermost dimension cost more than operations on the second-innermost dimension.

### 5.2 Trading Computation for Bandwidth: Rematerialization

When training large models, **activation recomputation through `jax.remat`** can be particularly useful on a TPU. The MXU has abundant compute capacity, while HBM bandwidth is often the bottleneck. Recomputing an intermediate value can be preferable to repeatedly storing and loading it in HBM.

---

## 6. Distributed Computation with SPMD

Compared with manually implementing distributed operators and NCCL communication in a GPU stack such as Megatron, TPU programming offers **SPMD: Single Program, Multiple Data**.

- **Direct ICI connections**: communication within a TPU pod uses its dedicated ICI network rather than passing through PCIe or the CPU.
- **Automatic partitioning**: define global tensors and a physical device mesh, and the XLA compiler inserts communication operations such as `AllGather` and `ReduceScatter` into distributed matrix multiplication.

---

## 7. Validate with Detailed Profiling and Busy Engines

Ultimately, tuning needs to be checked in a **trace viewer with nanosecond-level timing**.

### 7.1 Look for Busy Engines

A TPU contains scalar (0D), vector (1D), matrix (MXU / 2D), and transpose (XLU) engines.

**The objective** is for at least one engine to remain continuously busy on the timeline. Large gaps across every engine indicate that execution is blocked.

### 7.2 Locate the Core Loop by Counting Iterations

To connect low-level LLO instructions to source code:

1. **Estimate the loop counts** from the size of the input and output data.
2. **Count repetitions in the trace** to identify the outer loop, then locate the inner compute loop.
3. **Recognize instruction patterns**: use the frequency of VR and VACC loads, stores, and spills to check whether an instruction sequence corresponds to the expected kernel logic.

---

## 8. Three Practical Habits

1. **Use static, aligned shapes**: fix shapes, increase useful work per operation, and express conditional logic through selection or predication where possible.
2. **Hide latency through scheduling and unrolling**: use software pipelines to preload the next tile while computing the current one.
3. **Protect the innermost dimensions**: keep memory access linear and use scalar address calculations for higher-dimensional transformations.

A TPU is a powerful machine with a rigid execution model. Narrow the search space, align the work, and keep data movement regular to make the most of its compute capacity.

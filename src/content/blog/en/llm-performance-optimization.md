---
title: 'What Do We Mean When We Talk About LLM Performance Optimization?'
description: 'A TPU-based view of LLM performance optimization through data movement, computation, communication, scheduling, and computation graphs.'
pubDate: '2026/9/4'
tags: ["LLM", "TPU", "Performance", "Cost Model", "RCPSP"]
---

# What Do We Mean When We Talk About LLM Performance Optimization?
A TPU-based perspective on model performance optimization.

This article was formatted and polished with AI assistance; the author takes full responsibility for its content.

Summary:
LLM performance optimization can be reduced to improving a model's hardware utilization on a given system, while information entropy provides a way to quantify the complexity of the optimization problem.

## Contents

- Goal
- Decomposing time: five factors and entropy
- From global optimality to high utilization
- Method: cost model + RCPSP + graph-rewrite search
- Optimization directions
- Quantification: bottlenecks and critical paths (two formulas)
- Open questions

## Goal

The goal of LLM performance optimization is to maximize throughput, subject to acceptable accuracy and a fixed resource budget.

> 1. Acceptable accuracy means that the final model meets its evaluation target, such as on the ARC-AGI-2 benchmark.
> 2. Numerical differences caused by changes in execution order—fusion, splitting, reductions, and so on—are assumed to remain within the accepted accuracy tolerance.
> 3. A fixed resource budget means that the TPU hardware is given. Dynamically configurable choices such as network topology are not treated as fixed.

In practice, throughput is measured in tokens/s or train steps/s. This objective is clear and measurable, but it does not yet tell us where to optimize, so it needs to be decomposed further.

In real inference and training workloads, throughput must also account for concurrency, TTFT, TPOT, P99 latency, memory limits, and related constraints. These can be summarized by one bound: **the maximum degree of concurrency achievable within the allowed operating envelope**.

Under these conditions, the objective can be written as **throughput = concurrency / time**.

None of this depends on a particular model.

The degree of concurrency is not an independent, one-dimensional tuning knob. The parallelization strategy—the partitioning choices for TP, PP, EP, and FSDP—is itself a computation-graph transformation and is tightly coupled to the graph. I therefore include it under the computation-graph factor here and leave its details outside the scope of this article.

## Decomposing Time: Five Factors and Entropy

Model execution time is the result of many interacting factors. Here I borrow the information-theoretic concept of entropy to measure their complexity and define it formally as follows:

**Entropy = the average number of bits needed, under a fixed grammar, to identify one candidate among all legal arrangements (= log₂|S|).**

- `S` is the set of all legal arrangements—code permutations—under that grammar.
- Intuitively, it measures our prior uncertainty about which arrangement is the solution before the search begins.
- The grammar must be explicit about what constitutes an arrangement: a graph-rewrite sequence, tile configuration, instruction schedule, and so on. Otherwise, `|S|` cannot be calculated and the ranking cannot be falsified.
- Higher entropy means a larger search space and a harder optimization target.

The five broad factors that affect model execution time on a TPU, ordered from lower to higher conditional entropy, are:

1. **Data-movement time between HBM (DRAM) and VMEM (SRAM).** Given the **input and output parameters**, relatively few tile configurations can reach peak bandwidth, and the set of legal tile configurations is also comparatively small.
   1. Other hardware may use different names, but the underlying issue is the time required to move data between slower and faster memory.
   2. On another microarchitecture, memory movement may be considerably more complex, depending on how many hardware IP blocks participate in the transfer.
   3. If we consider every legal tile configuration that preserves mathematical equivalence across the entire model, the complexity grows dramatically.
2. **Computation time between VMEM and registers.** Given the **input and output parameters, the formula, and the hardware ISA**, relatively few instruction arrangements reach peak compute utilization; there are usually only a handful of effective patterns. Although many arrangements are legal, the ISA makes it possible to converge quickly.
   1. If only the mathematical formula and hardware ISA are fixed, while input and output shapes remain free to change through fusion or splitting, the complexity again grows dramatically.
3. **Communication time.** For communication itself, the shortest time is often deterministic under a **fixed network topology and fixed communication algorithm**. But there are many topology choices, the combinatorial space of communication algorithms is large, and both are tightly coupled to the parallelization strategy. Different combinations may also be optimal over different performance regimes.
4. **Stall time caused by synchronization and scheduling.** Finding the global minimum stall time is generally **intractable**, because stall time is coupled to the parallelization strategy and to the data-movement, compute, and communication time of each concurrent stream. Stall time is an outcome of those interactions.
5. **The computation graph.** The time directly attributable to the graph itself—memory management, resource contention, and similar overheads—is small. Its main effect is that graph choices reshape the other four factors. In a certain sense, the space of arrangements is unbounded: operations can be split and parallelized in many ways, and their execution order can be rearranged freely. That is why it comes last in this ordering.

Other factors either contribute less to LLM performance optimization or can be subsumed under these five:

- runtime and kernel-launch overhead
- host-side computation overhead, counted as computation and data-movement time
- memory allocation and deallocation, counted as runtime overhead
- workload imbalance, counted as synchronization and scheduling
- resource contention, counted under the computation graph
- memory management, counted under the computation graph
- cache, counted as computation time; TPU has no user-visible cache hierarchy
- warm-up, counted as host-side overhead

> The five main factors are coupled, so discussing the entropy of any one factor in isolation is meaningless. At whole-model scale, all five have high entropy; there is no meaningful absolute ranking among them. In practice, however, we reduce complexity and uncertainty layer by layer. The result can be written as a conditional-entropy decomposition using the chain rule:

```
H(total) = H(graph) + H(communication | graph) + H(stall | graph, communication) + H(compute | first three) + H(data movement | first four)
```

Each term is the number of decision bits still required after the other factors have been fixed. In other words, a term has low entropy because more of its context has already been fixed. The order of this chain-rule decomposition can be changed, so this ordering does not imply relative importance.

## From Global Optimality to High Utilization

The five factors above lead to several conclusions:

1. Each category affects the **execution time of different hardware units** from a different angle. Ultimately, the model has to run on hardware.
2. A factor has lower entropy when other parts of the real system have already removed more of its uncertainty—precisely the meaning of conditional entropy. Higher-entropy factors have fewer constraints. Because every factor interacts with the others in a real workload, they cannot be assigned a meaningful absolute ranking.
3. Model optimization contains many mutually dependent factors, each with high entropy and complexity. Finding the global optimum is generally out of reach.
4. Regardless of how the computation graph is transformed while preserving mathematical equivalence, there is a lower bound on computation and data movement, while every hardware unit has a maximum capability.
5. Combining points 1, 3, and 4: the global optimum is effectively unattainable, every factor influences hardware execution time in a different way, and computation and data movement have lower bounds. From an engineering perspective, we should therefore aim not for a whole-model global optimum, but for high utilization of a specific model on specific hardware, benchmarked against minimum necessary work and maximum peak capability. This turns an effectively unsolvable global problem into a utilization problem under progressively tighter constraints.

Returning to this perspective, hardware utilization alone cannot compare two approaches. Wall-clock time remains the practical standard: minimize useless work, saturate the bottleneck hardware unit, hide the remaining work behind it whenever possible, and approach the bound imposed by constraints such as memory capacity and latency.

**Corollary (difficulty of reaching a target):** Let the target be a utilization threshold, such as 60%, and let `A` be the set of legal arrangements that meet it, with `H(A) = log₂|A|`. A uniform random search then needs an expected `|S|/|A| = 2^(H(total) − H(A))` trials to hit an acceptable solution. The difficulty of reaching the target is `H(total) − H(A)` bits. With a relaxed target, `A` is large and the problem is easier; as the target approaches a capacity constraint, `A` collapses and the difficulty rises again. The difficulty of each factor is therefore not a class property such as NP-hardness, but an instance property captured by `H(total) − H(A)`.

### Reference Lower Bounds (Idealized)

The lower bounds on computation and data movement can be derived from a purely mathematical model and the target hardware under these idealized assumptions:

- With no recomputation, the compute lower bound is the algorithm's exact FLOP count and can be calculated directly.
- With infinite HBM and VMEM capacity, the data-movement lower bound is compulsory traffic—each byte read and written exactly once—and can also be calculated directly.

These assumptions remove capacity constraints, so the result is only a theoretical lower bound—a denominator for utilization—not an attainable target. No real system can reach 100% of it. An attainable target should instead use the empirical ceiling observed in measurements, such as 85%–90% HBM utilization.

## Method: Cost Model + RCPSP + Graph-Rewrite Search

The basic idea is quantification: use a cost model to measure how each code segment affects each hardware unit. A kernel can be divided into segments, such as 1D code executed by a vector unit and 2D code executed by a matrix unit. Once the time of each segment is quantified, the segments can be rearranged according to the computation graph.

The next step is to include fusion and splitting. While preserving the graph's mathematical equivalence, these transformations add or remove code segments: fusion can be viewed as removing an HBM → VMEM transfer, while splitting can be viewed as adding an HBM ↔ VMEM transfer.

Some segment properties directly affect time, such as 1D and 2D operation counts; others have indirect effects, such as VMEM and HBM occupancy. Combining multiple segments also requires modeling their interactions. A segment that exceeds memory capacity introduces additional HBM ↔ VMEM traffic. If a segment fits in VMEM and has no dependency between MXU and VPU work, those units can overlap and reduce total time.

**Problem formulation:** this can be modeled as **resource-constrained project scheduling (RCPSP) plus graph-rewrite search**:

- Code segments have dependency constraints and cannot be freely rearranged.
- Time is not additive: concurrent segments take the maximum, while serial segments take the sum.
- Resource constraints are multidimensional—VMEM, registers, bandwidth, DMA slots—and the segments contend for them.

Under this formulation, whole-model execution time is the makespan of an RCPSP instance. RCPSP is NP-hard, so "solving under progressively tighter constraints" becomes "heuristic search under progressively tighter constraints," which is consistent with the goal of high utilization rather than exact global optimality.

NP-hardness describes the worst case for exact solutions. Engineering only needs to reach sufficiently high utilization, and the practical search difficulty is determined by the density of acceptable solutions. When the target is relaxed, acceptable solutions are dense and the search is easy; as the target approaches a capacity constraint, they become sparse and the search becomes hard. In practice, autotuners can often find a "good enough" kernel in a vast search space with only hundreds of evaluations. This suggests that acceptable solutions are not exceptionally sparse for this family of instances. The hard cases concentrate in pathological tail shapes—non-divisible dimensions and ragged batches—where `|A|` is smallest.

The accuracy of the cost model rests on the following assumptions:

1. TPU hardware is simple enough: it has no user-visible cache hierarchy, runs at a fixed frequency, has no warp divergence, and exposes a fixed ISA.
2. All source code is available, from XLA's HLO optimization, fusion, layout, and SPMD partitioning through final code generation in libtpu/ptxas. Compiler decision rules can therefore be reproduced completely rather than inferred from samples.
3. From the chip vendor's perspective, hardware constants are available. HBM bank conflicts, read/write turnaround behavior, power-wall effects, and similar parameters can be derived from RTL, cycle-accurate simulation, and post-silicon characterization.
4. AI can accelerate evaluation. Assumptions 1–3 form a cheap and accurate label generator—full source plus a hardware model produces a large offline set of precise labels. A learned cost-model proxy can then be trained on those labels, amortizing the cost of compiling or simulating each candidate to nearly zero.

Under these assumptions, both the compiler and hardware sides of the cost model are closed, while AI amortizes the evaluation cost. The cost model can therefore be treated as solved. A learned proxy still introduces modeling error, but assumptions 1–3 provide an unlimited supply of precise offline training labels, allowing that error to be reduced to an engineering-acceptable level. The accuracy-cost tradeoff converges toward an offline-trained proxy, leaving proxy drift after compiler upgrades as the main residual risk; see Open Question 1.

## Optimization Directions

1. **Bottom-up:** optimize from lower-entropy factors toward higher-entropy ones. Computation, for example, can reach high hardware utilization relatively easily under a specific set of constraints. It can then constrain the layers above: only admit parameters that preserve high utilization, and use those constraints to optimize the computation graph and communication or data-movement strategies. More precisely, fixed information such as the hardware ISA is used to eliminate uncertainty in the other factors.
2. **Top-down:** optimize from higher-entropy factors toward lower-entropy ones. Entropy represents both complexity and freedom. By changing the computation graph, a top-down approach reshapes data movement and computation, using the larger degree of freedom to find profitable transformations.

The two approaches complement each other and should be iterated together rather than treated as alternatives.

## Quantification: Bottlenecks and Critical Paths (Two Formulas)

This objective can be quantified primarily through two views: the compute bottleneck and the critical path.

Pure calculation combined with a sufficiently accurate cost model can estimate whole-model time under different strategies. Those estimates then become the optimization target for improving hardware utilization through approaches such as the two directions above.

### Formula 1: Bottleneck Form (Sufficient Concurrency)

When there is enough independent concurrency for different streams to overlap until one hardware IP reaches high whole-model utilization—say 80%—time can be simplified as:

```
time = max(data movement / bandwidth / movement efficiency,
           computation / peak compute / compute efficiency,
           communication volume / communication capacity / communication efficiency,
           ...)
```

On TPU today, time is primarily determined by the slowest-running component among all hardware IP blocks that can operate in parallel. The main overlapping factors are data movement, computation, and communication. Other parallel factors can be added as needed, such as host-memory ↔ device-memory transfers.

Applicability: enough independent concurrency must be available to sustain overlap. This form does not apply to serially dependent workloads such as decoding when no useful overlap is available.

(The 80% figure here is a utilization threshold for entering an overlap steady state. It is distinct from the 60% target in the earlier corollary on acceptable-solution difficulty.)

### Formula 2: Critical-Path Form (Insufficient Concurrency)

When concurrency is insufficient, time is primarily described by:

```
time = sum(work on each hardware IP along the critical path / peak capability / utilization)
```

Work on hardware IP blocks outside the critical path should be hidden behind it. If it cannot be hidden, the longest resulting path becomes the new critical path.

This is a descriptive formula for a schedule that has already been chosen. Finding the schedule with minimum makespan is precisely the RCPSP problem and is itself NP-hard.

**Unified view:** these formulas describe two extreme regimes of RCPSP makespan. With abundant resources and full overlap, makespan approaches `max(...)`; with scarce resources, it approaches the sum along the critical path.

## Open Questions

1. **Proxy drift and retraining:** after a compiler upgrade, a learned cost model requires regenerated labels and retraining. The retraining cost determines how quickly the framework can adapt to upstream changes.
2. **The interface between concurrency and computation-graph search:** once the parallelization strategy is incorporated into the computation graph, how should outer-loop tuning—such as the peak-memory budget—and inner-loop search—such as the time budget—alternate?
3. **Can model-generated candidates replace heuristic search?** The current objective is to optimize computation graphs. Can a large model learn to find the arrangements directly, generating candidates in place of heuristic search?

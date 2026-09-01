---
title: 'ECC Partial Write and Model Design'
description: 'Why does a few-byte partial write become a full read-modify-write once ECC is on? A walk through RMW using a BF16[8,130] DMA deslice: the principle, the cost, and how software avoids it.'
pubDate: '2026/8/31'
tags: ["ECC", "HBM", "Memory", "Kernel", "DMA", "Performance"]
---

# ECC Partial Write and Model Design

This article was written with AI assistance; the author takes full responsibility for its content.

## 1. Summary

- **The problem**: the smallest unit of ECC checking and computation is a group of `G` bytes, not a single byte. A partial write cannot produce the new ECC directly, so it becomes an RMW: read old → merge → write full.
- **The principle**: RMW carries a very large performance overhead: to guarantee a whole-block write, it adds a read of the old data, a full write, and the potential read/write turn-around cost. Unaligned writes, cross-boundary writes, stride drift, and scattered small writes all keep inflating the partial-write count `P`.
- **Software avoidance**: reduce `P` before the write is issued: align base, `dst_stride`, and `last_dim_size`; align to the smallest unit whenever possible.

> Note: `last_dim_size` is the size of the lowest (contiguous) dimension.

### 1.1 Quick reference

| Design item | Friendly range | Unfriendly range |
| --- | --- | --- |
| last dim size | `last_dim_size × dtype_bytes` is a multiple of `G` | a tail of a few bytes |
| store_stride | `dst_stride` is a multiple of `G` | tight stride equals the row bytes but is not a multiple of `G` |
| Write shape | contiguous, aligned, block writes | per-row small DMAs, scatter, partial slices |
| Update merging | updates in the same `G` merged into one write | the same `G` repeatedly updated by many small stores |
| Value of `G` | confirmed from the target chip's documentation | treating the 128 B bus width as `G` |

## 2. The symptom: same tile, some layouts are visibly slower

If your model shows any of the following, suspect ECC-RMW first:

- a `BF16[8,130]` tile written back to HBM via DMA deslice writes 2080 B logically, yet switching to a contiguous or aligned layout changes the time noticeably;
- the effective bytes written stay the same, but read traffic on the memory side goes up;
- shifting the base offset by a few dozen bytes makes performance fluctuate periodically;
- padding the allocation's `dst_stride` from 260 B to 512 B doesn't help, but actually writing the full 512 B does;
- per-row small DMAs are much slower than one contiguous DMA.

None of these prove ECC is the cause — the counters and control experiments in section 5 are still needed. But they are typical signals pointing at RMW.

## 3. Cause and principle: the BF16[8,130] stride store

The `G`, the bus width, and the 32 B HBM operation figures in this article come from public models or examples; for a concrete chip, trust its documentation and counters.

ECC check bits protect a group of `G` bytes. When a write covers the whole `G`, the controller gets all the new data and computes the new ECC directly; when it covers only part, the controller reads the old `G` back, checks and corrects it, merges in the new bytes, recomputes the ECC for the whole group, and writes it back. Intel's public memory-controller documentation describes the same flow [1].

![Full write vs partial write inside an ECC controller](/blog/ecc-partial-write/en/ecc-partial-write-flow-en.svg)

Figure 1: A full coverage of `G` produces the new ECC directly; a partial coverage reads the old data first, then merges, recomputes, and writes back.

A model designer only needs three hardware concepts:

1. **128 B cycle (bus width)**: 1024 bit = 128 B, about two common cache lines. Hardware views a write as a number of 128 B transfer units.
2. **Byte mask**: within each transfer unit, which bytes are the new values being written; the rest must keep their old values.
3. **Protection granule `G`**: the number of bytes the ECC check bits actually protect.

Back to the `BF16[8,130]` deslice: the DMA issues 8 row writes; `dst_stride` decides where the next row starts, and `last_dim_size` decides how much each DMA actually writes.

| deslice config | dst_stride | written per row | 128 B cycles cut out of row 0 |
| --- | ---: | ---: | --- |
| `rows128_tight` | 256 B | 256 B | 2 full |
| `rows130_tight` | 260 B | 260 B | 2 full + one 4 B tail; row starts drift afterwards |
| `rows130_padded_valid` | 512 B | 260 B | 2 full + one 4 B tail; row starts realign |
| `rows130_padded_full` | 512 B | 512 B | 4 full |

![2-D DMA deslice: one tile, many stride stores](/blog/ecc-partial-write/en/ecc-dma-deslice-stride-store-en.svg)

Figure 2: The same tile, different `dst_stride` and `last_dim_size` — the number of partials seen by the controller is completely different.

A `rows130_tight` row is 260 B: the first two cycles carry 128 B each, and the third carries only 4 B. With `dst_stride = 260 B`, the start of each next row shifts by 4 B relative to the 128 B boundary:

```text
row_i_offset = (base_offset + i × dst_stride_bytes) mod 128
```

The offsets of the 8 rows are `0, 4, 8, …, 28`. The row at offset 0 has only one tail partial; every other row has both a head and a tail partial. With more rows, the offsets cycle `0, 4, …, 124, 0` every 32 rows.

The difference between ECC off and on exists only for partial updates:

| Mode | Partial update path |
| --- | --- |
| ECC completely off | the model assumes a physical WDM is available: masked write, no read of old data for ECC |
| ECC enabled | read old → merge → write full |
| ECC bypass | semantics vary by IP; may still occupy ECC pins and disable WDM, and does not necessarily avoid RMW |

![Partial-update paths with ECC completely off vs ECC enabled](/blog/ecc-partial-write/en/ecc-partial-write-mode-matrix-en.svg)

Figure 3: Full coverage shows no RMW difference between the two; a partial update adds one old-data read when ECC is enabled.

The one-line conclusion: for the 8-row deslice, `rows130_tight` is the worst, `rows130_padded_valid` still has tails, and `rows130_padded_full` and `rows128_tight` have no partials. The numbers are in section 5.

## 4. What model designers should watch, and how to avoid it

![How software shapes a write: the decision order](/blog/ecc-partial-write/en/ecc-software-write-strategy-en.svg)

Figure 4: Write fully and aligned first; if that's impossible, merge or pad safely; failing that, write the bulk fully and leave only the head/tail to the slow path.

When designing a layout, check in this order:

1. **Row bytes aligned to `G`.** A BF16 row of 128 elements is 256 B; 130 elements is 260 B, which immediately adds a 4 B tail.
2. **`dst_stride` aligned to `G` too.** Aligning the base address is not enough; `dst_stride = 260 B` keeps drifting every row from the second row on.
3. **Padding counts only when written.** `rows130_padded_valid` still keeps one tail partial per row; only `rows130_padded_full` pushes `P` to 0. The price is 252 B extra per row — weigh it against the RMW cost when designing.
4. **Merge small updates in the same `G`.** Accumulate them on-chip or in a temporary buffer first, then write once, fully. A software read-merge-write has no atomicity; add your own synchronization.
5. **Do not hard-code `G = 128 B`.** HBM-class controllers may split operations into 32 B chunks; the same layout behaves differently on different chips. Ask the hardware side before designing.

A load-modify-store in the algorithm is not a controller RMW: what matters is always the address, length, and byte mask of the final requests.

Two more pitfalls:

- ECC bypass is not "ECC off": PG276's bypass also disables Write Data Mask, so partial writes may still RMW [2]. Disabling ECC also forfeits error detection and correction, so it must not become a default optimization.
- Before the first partial write, the target region must already be initialized; otherwise the first RMW reads an invalid old ECC state [3].

To verify a design, run the same `BF16[8,130]` deslice across the four configurations and watch the `AM_RMW_CYCLE` counter [2]. If ECC-RMW is the main cause, the RMW density should look like: `rows130_tight` highest, `rows130_padded_valid` next, `rows130_padded_full` and `rows128_tight` close to 0.

## 5. Putting numbers on it

> This section only quantifies the conclusions of sections 3 and 4. Skipping the formulas does not change the design conclusions.

Cycles on the 128 B interface touched by one update:

```text
N_AXI = ceil(((address mod 128) + update_bytes) / 128)
```

Let `F` be the number of full granules, `P` the number of partial granules, and `U` the number of effective update bytes. The teaching model takes `G = 128 B`:

```text
ECC off:  D_off ≈ G × (F + P)
ECC on:   D_on  ≈ G × (F + 2P)
amplification = D / U
```

`D` is a simplified model of the data volume the controller services, not real DRAM current. One 260 B row takes 3 write slots (384 B) with ECC off; with ECC on it becomes 2 full writes plus 1 RMW (512 B).

Accumulated over 8 rows:

| Config | `P` | Cost |
| --- | ---: | --- |
| `rows128_tight` | 0 | no extra RMW |
| `rows130_padded_valid` | 8 | one 4 B tail kept per row |
| `rows130_tight` | 15 | most rows have both head and tail partials |
| `rows130_padded_full` | 0 | 252 B of padding written extra per row |

What model design must control is `P`, not the total bytes written.

## Closing

ECC-RMW is not about the checksum being slow to compute; it is about partial writes forcing the controller to read old data first. Align the row bytes and `dst_stride` up front, use only padding that is actually written, and merge small updates — then most writes stay in the hardware-friendly range.

## References

1. Intel, [External Memory Interface Handbook](https://cdrdv2-public.intel.com/654635/emi_archive_101.pdf) — the "Partial Writes" section describes the read, correct, merge, and write-back flow of ECC partial writes.
2. AMD, [AXI HBM Controller PG276](https://docs.amd.com/r/en-US/pg276-axi-hbm/AXI-Considerations) — 32 B HBM operations, unaligned writes, and ECC RMW; [Reliability Options](https://docs.amd.com/r/en-US/pg276-axi-hbm/Reliability-Options-Tab) — bypass disables Write Data Mask; [Register Map](https://docs.amd.com/r/en-US/pg276-axi-hbm/Memory-Controller-Register-Map) — the `AM_RMW_CYCLE` counter.
3. AMD, [Versal Adaptive SoC Soft DDR4 Memory Controller: ECC](https://docs.amd.com/r/en-US/pg353-versal-acap-soft-ddr4-mem-ip/ECC) — the different partial write paths with ECC on and off, and the initialization requirement before RMW.
4. Micron, [Integrating and Operating HBM2E Memory (local copy)](/blog/ecc-partial-write/micron-hbm2e-memory-wp.pdf) — HBM2E channels and pseudo channels, and DM pins shared between Write Data Mask and ECC data.
5. Arm, [AMBA AXI and ACE Protocol Specification, IHI 0022H](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/IHI0022H_amba_axi_protocol_spec.pdf) — AXI data width and byte-lane write strobes.

A local copy of the [AXI High Bandwidth Memory Controller PG276](/blog/ecc-partial-write/axi-hbm-controller-pg276.pdf) is also attached.

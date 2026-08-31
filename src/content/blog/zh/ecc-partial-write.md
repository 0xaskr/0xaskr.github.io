---
title: 'ECC 为什么会让一次“小写入”变成“大动作”？'
description: '为什么 ECC 开启后，几字节的局部写会变成一次完整的读-改-写？从 BF16[8,130] DMA deslice 讲起：RMW 的原理、代价，以及软件侧的规避方法。'
pubDate: '2026/8/31'
tags: ["ECC", "HBM", "Memory", "Kernel", "DMA", "Performance"]
---

# ECC Partial Write 如何影响模型设计

本文使用 AI 辅助写作，但作者对文章内容负责。

## 1. 总结

- **问题**：ECC 校验与计算最小单位是一组数据 `G`，而不是单个字节。Partial Write 没有办法直接生成新 ECC，于是变成 RMW：read old → merge → write full。
- **原理**：RMW会带来非常大的性能开销，其原因是为保证整块写入，RMW中新增的read old和 wirte full 以及潜在的读写切换带来的开销。非对齐写入、跨边界写入、stride 漂移和小数据分散写入，都会让 Partial Write 数量 `P` 反复增加。
- **软件规避**：在写入前减少 `P`：对齐 base、`dst_stride`、`last_dim_size`；尽量对齐到最小单位；

> note: last_dim_size 表示为最低维度的大小(连续存储的维度)

### 1.1 速查

| 设计项 | 友好区间 | 不友好区间 |
| --- | --- | --- |
| last dim size | `last_dim_size × dtype_bytes` 是 `G` 的整数倍 | 差几个 byte 的 tail |
| store_stride | `dst_stride` 是 `G` 的整数倍 | tight stride 等于行字节数，但不是 `G` 的整数倍 |
| 写入形态 | 连续、对齐、整块写入 | 逐行小 DMA、scatter、partial slice |
| 更新合并 | 同一 `G` 内的更新合并后一次写出 | 同一 `G` 被多次小 store 反复更新 |
| `G` 的取值 | 按目标芯片文档确认 | 把总线宽度 128 B 直接当成 `G` |

## 2. 现象：同一个 tile，为什么有的 layout 明显更慢

模型里出现下面这类情况时，优先怀疑 ECC-RMW：

- 一个 `BF16[8,130]` tile 用 DMA deslice 写回 HBM，逻辑上写 2080 B，但改成连续写或对齐 layout 后，耗时明显不同；
- 有效写入字节数没变，内存侧读流量却上升；
- base offset 微调几十字节，性能出现周期性波动；
- 把 allocation 的 `dst_stride` 从 260 B pad 到 512 B 没用，但真正写满 512B 后变快；
- 逐行小 DMA 比一次连续 DMA 慢得多。

这些现象不等于一定由 ECC 引起，还需要第 5 节的计数器和对照实验确认；但它们是指向 RMW 的典型信号。

## 3. 原因和原理：以 BF16[8,130] stride store 为例
本文中的 `G`、总线宽度 和 32 B HBM operation 均来自公开模型或示例；落到具体芯片，以目标文档和计数器为准。

ECC 校验位保护一组 `G` 字节。一次写覆盖整个 `G` 时，控制器拿到全量新数据，直接生成新 ECC；只覆盖一部分时，它先读回旧 `G`，检查纠错后合并新字节，再重算整组 ECC 并写回。Intel 公开的内存控制器文档描述了同样的流程 [1]。

![完整写与局部写在 ECC 控制器里的分叉](/blog/ecc-partial-write/ecc-partial-write-flow.svg)

图 1：完整覆盖 `G` 直接生成新 ECC；局部覆盖先读旧数据，再合并、重算并写回。

模型设计者只需要三个硬件概念：

1. **128 B cycle(总线宽度)**：1024 bit = 128 B，约等于两个常见 cache line。硬件把一次写入看成若干个 128 B 的搬运单位。
2. **byte mask**：每个搬运单位里，哪些字节是本次要写的新值；其余字节必须保留旧值。
3. **保护粒度 `G`**：ECC 校验位实际保护的字节数。

回到 `BF16[8,130]` deslice：DMA 按行生成 8 条写请求；`dst_stride` 决定下一行起点，`last_dim_size` 决定每条 DMA 实际写多少。

| deslice 配置 | dst_stride | 每行实际写入 | 第 0 行切出的 128 B cycles |
| --- | ---: | ---: | --- |
| `rows128_tight` | 256 B | 256 B | 2 full |
| `rows130_tight` | 260 B | 260 B | 2 full + 1 个 4 B tail，之后行起点漂移 |
| `rows130_padded_valid` | 512 B | 260 B | 2 full + 1 个 4 B tail，行起点重新对齐 |
| `rows130_padded_full` | 512 B | 512 B | 4 full |

![二维 DMA deslice：一个 tile，多条 stride store](/blog/ecc-partial-write/ecc-dma-deslice-stride-store.svg)

图 2：同一个 tile，`dst_stride` 和 `last_dim_size` 不同，落到 controller 的 partial 数量完全不同。

`rows130_tight` 一行是 260 B：前两个 cycle 各 128 B，第三个 cycle 只有 4 B 有效。`dst_stride = 260 B` 时，下一行起点相对 128 B 边界每次移动 4 B：

```text
row_i_offset = (base_offset + i × dst_stride_bytes) mod 128
```

8 行的偏移是 `0, 4, 8, …, 28`。偏移 0 的行只有 1 个 tail partial；其余行同时有 head 和 tail partial。行数更多时，偏移会按 `0, 4, …, 124, 0` 每 32 行循环。

ECC off 与 on 的差别，只在 partial update 上：

| 模式 | partial update 路径 |
| --- | --- |
| ECC completely off | 模型假设 physical WDM 可用：masked write，不因 ECC 读旧数据 |
| ECC enabled | read old → merge → write full |
| ECC bypass | 语义因 IP 而异；可能仍占 ECC pins 并关闭 WDM，不一定避开 RMW |

![ECC completely off 与 ECC enabled 下 partial update 的性能路径](/blog/ecc-partial-write/ecc-partial-write-mode-matrix.svg)

图 3：完整覆盖时两者没有 RMW 差异；partial update 在 ECC enabled 时多出一次旧数据读取。

一句话结论：对 8 行 deslice，`rows130_tight` 最差，`rows130_padded_valid` 仍有 tail，`rows130_padded_full` 和 `rows128_tight` 没有 partial。具体数字见第 5 节。

## 4. 模型设计者的注意和规避方法

![软件侧整理写入形态的决策顺序](/blog/ecc-partial-write/ecc-software-write-strategy.svg)

图 4：先完整对齐写；做不到再安全合并或填充；仍不行就让主体完整写，只把头尾留给慢路径。

设计 layout 时按下面顺序检查：

1. **行字节数对齐 `G`**。BF16 一行 128 个元素是 256 B；130 个元素是 260 B，立刻多出一个 4 B tail。
2. **`dst_stride` 也要对齐 `G`**。只对齐首地址不够；`dst_stride = 260 B` 会让第二行起持续漂移。
3. **padding 只有被写入才算数**。`rows130_padded_valid` 每行仍有 1 个 tail partial；`rows130_padded_full` 才把 `P` 压到 0。代价是每行多写 252 B，设计时要和 RMW 成本比较。
4. **合并同一 `G` 内的小更新**。先在片上或临时 buffer 中累积，再一次完整写出。软件自己做 read-merge-write 时没有原子性，需要自行同步。
5. **不要写死 `G = 128 B`**。HBM 类控制器可能按 32 B 拆 operation；同一个 layout 在不同芯片上表现不同，设计前先问硬件侧。

算法里的 load-modify-store 不等于 controller RMW：判断依据永远是最终请求的 address、length 和 byte mask。

还有两个坑：

- ECC bypass 不等于完全关闭：PG276 的 bypass 会同时关闭 Write Data Mask，局部写仍可能 RMW [2]。关闭 ECC 还会损失检错纠错能力，不能作为默认优化。
- 局部写之前，目标区域必须已经初始化；否则第一次 RMW 读到的旧 ECC 是无效状态 [3]。

验证设计时，用同一个 `BF16[8,130]` deslice 对比四种配置，并查看 `AM_RMW_CYCLE` 计数器 [2]。若 ECC-RMW 是主因，RMW 密度应大致为：`rows130_tight` 最高，`rows130_padded_valid` 次之，`rows130_padded_full` 和 `rows128_tight` 接近 0。

## 5. 量化

> 本节只把第 3、4 节的结论量化。不看公式也不影响设计结论。

一次更新触及的 128 B 接口 cycle 数：

```text
N_AXI = ceil(((address mod 128) + update_bytes) / 128)
```

设 `F` 为 full granule 数，`P` 为 partial granule 数，`U` 为有效更新字节数。教学模型取 `G = 128 B`：

```text
ECC off：  D_off ≈ G × (F + P)
ECC on：   D_on  ≈ G × (F + 2P)
放大倍数： amplification = D / U
```

`D` 是控制器服务数据量的简化模型，不是真实 DRAM 电流。260 B 一行在 ECC off 下占 3 个 write slot（384 B）；ECC on 下是 2 个 full write 加 1 个 RMW（512 B）。

按 8 行累计：

| 配置 | `P` | 代价 |
| --- | ---: | --- |
| `rows128_tight` | 0 | 无额外 RMW |
| `rows130_padded_valid` | 8 | 每行保留一个 4 B tail |
| `rows130_tight` | 15 | 大部分行同时有 head 和 tail partial |
| `rows130_padded_full` | 0 | 每行多写 252 B padding |

模型设计时真正要控制的是 `P`，而不是只看总写入字节数。

## 结语

ECC-RMW 的核心不是校验算得慢，而是局部写迫使控制器先读旧数据。模型设计时提前对齐行字节数和 `dst_stride`、只使用真正写满的 padding、合并小更新，就能让大部分写入留在硬件友好区间。

## 公开资料

1. Intel, [External Memory Interface Handbook](https://cdrdv2-public.intel.com/654635/emi_archive_101.pdf)，其中 “Partial Writes” 小节说明 ECC partial write 的读取、纠错、合并和回写流程。
2. AMD, [AXI HBM Controller PG276](https://docs.amd.com/r/en-US/pg276-axi-hbm/AXI-Considerations)：说明 32 B HBM operation、非对齐写与 ECC RMW；[Reliability Options](https://docs.amd.com/r/en-US/pg276-axi-hbm/Reliability-Options-Tab) 说明 bypass 会关闭 Write Data Mask；[Register Map](https://docs.amd.com/r/en-US/pg276-axi-hbm/Memory-Controller-Register-Map) 提供 `AM_RMW_CYCLE`。
3. AMD, [Versal Adaptive SoC Soft DDR4 Memory Controller: ECC](https://docs.amd.com/r/en-US/pg353-versal-acap-soft-ddr4-mem-ip/ECC)，说明 ECC 开关下 partial write 的不同路径及 RMW 前初始化要求。
4. Micron, [Integrating and Operating HBM2E Memory（本地副本）](/blog/ecc-partial-write/micron-hbm2e-memory-wp.pdf)，说明 HBM2E channels、pseudo channels，以及 DM pins 在 Write Data Mask 与 ECC data 之间复用。
5. Arm, [AMBA AXI and ACE Protocol Specification, IHI 0022H](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/IHI0022H_amba_axi_protocol_spec.pdf)，说明 AXI data width 与 byte-lane write strobes。

另附本地副本：[AXI High Bandwidth Memory Controller PG276](/blog/ecc-partial-write/axi-hbm-controller-pg276.pdf)。

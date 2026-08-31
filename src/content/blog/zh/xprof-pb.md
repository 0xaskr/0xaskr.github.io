---
title: 'JAX Profiler 生成的 .pb 文件如何解析'
description: '一次讲清 JAX Profiler 导出的 *.xplane.pb 怎么读：从 XSpace/XPlane 结构到 kernel 定位和 device time。'
pubDate: '2026/8/31'
tags: ["XProf", "XLA", "JAX", "TPU", "Profiling"]
---

# JAX Profiler 生成的 .pb 文件如何解析

写这篇是因为前阵子排查一个 Pallas kernel 的耗时，发现无法在cli中去读取到kernel的耗时， 也说不清它是怎么来的、该怎么读。后来从 `jax.profiler.trace()` 一路翻到 XLA/TSL profiler，才把整条链路串起来。下面这些问题，是我中途卡住过的：

- 被 profiling 的代码在跑的时候，JAX 做了什么？
- kernel 的 host/device 活动是谁采集的，存在哪里？
- `*.xplane.pb` 怎么生成？为什么它比 `*.trace.json.gz` 更适合当原始入口？
- 怎么在 `.pb` 里定位一个 kernel、算 device time，并导出成能看的 JSON？
- XProf 在原始 XPlane 上又加了什么处理？

整条链路大致是：

```text
jax.profiler.trace()
  -> ProfilerSession
  -> collectors
  -> XSpace
  -> XPlane / XLine / XEvent
  -> *.xplane.pb
  -> protobuf parser / resolved JSON
  -> kernel device time
  -> XProf derived views
```

## 0. 分析范围与源码版本

下面的结论都基于 PR #533 固定的源码版本，省得以后版本变了对不上：

| 组件  | 固定版本 | 用到的主要源码 |
|-----|------|-----------|
| JAX | `6855c5bdd6a30d2f2bc7bfc13bcd211c44232362` | `third_party/jax/jax/_src/profiler.py` |
| XLA / TSL | `7c3dd1936addd297d7c6fa46f6183986fc4160c3` | `xla/python/profiler.cc`、`profiler_session.cc`、`xplane.proto` |
| XProf | `403109fc2d5b6e1f87cdd26379adefbcb94b89be`，tag `xprof-v2.21.1` | `xplane_to_tools_data.cc`、`xplane_to_hlo.cc` |
| Falcon | `7e58a8e30a599d11ca9f452edb03e5893a5804d9` | `falcon_xprof_summary.py`、`falcon_profile_overlap.py` |

平时说的 “XProf `.pb`”，一般指 `*.xplane.pb`。它其实是 JAX 调 jaxlib/XLA/TSL profiler 写出来的，XProf 只是下游消费者。实际采到什么，还取决于当时的 jaxlib、libtpu/PJRT plugin、ProfileOptions 和硬件；

## 1. 从一次 kernel profiling 开始

最常用的入口长这样：

```python
import jax


# 先完成编译和 warm-up，避免把首次编译混入 steady-state kernel 时间。
kernel(x).block_until_ready()

with jax.profiler.trace("profile-data"):
  y = kernel(x)
  y.block_until_ready()
```

退出 `with` 之后，默认能在这样的目录下看到产物：

```text
profile-data/plugins/profile/<timestamp>/
  <hostname>.xplane.pb
  <hostname>.trace.json.gz
```

这两个文件都从 `trace()` 的调用产生，先看这个 context manager 做了什么。

## 2. `jax.profiler.trace()` 只是个生命周期控制器

公共 API 最终落在 `third_party/jax/jax/_src/profiler.py`，剥开看就是一段很短的 context manager：

```python
@contextmanager
def trace(log_dir, ..., profiler_options=None):
  start_trace(log_dir, ..., profiler_options)
  try:
    yield
  finally:
    stop_trace()
```

无非两个动作：

* `start_trace()`：创建并启动一次 profiler session。
* `stop_trace()`：停止本次 session，收集数据并导出文件。

它既不去读 TPU 计数器，也不自己拼 protobuf；采集、合并、序列化都在 jaxlib 自带的 XLA/TSL C++ profiler 里完成。

因为是 `finally`，`with` 里的代码哪怕抛异常，JAX 也会尽量把 trace 停下来并导出去。

## 3. `start_trace()`：启动 ProfilerSession

### 3.1 为什么先初始化 backend

`start_trace()` 按顺序做这些事：

1. 获取 `_profile_state.lock`，拒绝同一进程中的第二个 JAX programmatic trace。
2. 清空上一次 profiler metadata。
3. 调 `xla_bridge.get_backend()` 初始化 backend（backend 是自己写的，理论上可以脱离 XLA/XProf 这套）。
4. 创建或采用调用者传入的 `ProfileOptions`。
5. 注册 `jax_version`、`jaxlib_version` 和每个 backend 的 `<platform>_version` metadata。
6. 构造 `_profiler.ProfilerSession(options)`。
7. 保存输出目录和 Perfetto 选项。

第 3 步经常被忽略。TPU backend 初始化会让 libtpu/PJRT 的 profiler extension 赶在 session 创建前注册；少了这一步，factory 列表里可能没有 TPU tracer，最后 `.pb` 是写出来了，里面却只有 host 数据，device plane 是空的。

### 3.2 ProfilerSession 创建后立即开始采集

`_profiler.ProfilerSession(options)` 会进到 `third_party/xla/xla/python/profiler.cc`，创建 TSL 的 `ProfilerSession`：

```text
ProfilerSession::Create(options)
  -> 获取进程级 ProfilerLock
  -> 记录 start_time_ns
  -> CreateProfilers(options)
  -> 构造 ProfilerCollection
  -> 对每个 profiler 调用 Start()
```

`ProfilerSession` 就是一次有明确开始和结束的采集会话。它一构造就开始采，不用等第一条 JAX op。

Python binding 这边的关键默认值：

| 选项  | 当前固定源码默认值 | 对 kernel profile 的影响 |
|-----|----------:|----------------------|
| `host_tracer_level` | 2         | 收集 TraceMe、dispatch 和较高层 host activity |
| `python_tracer_level` | 1         | 收集 Python function tracing |
| `device_tracer_level` | 1         | 允许适用的 GPU/TPU device tracer 工作 |
| `enable_hlo_proto` | `true`    | 收集 HLO metadata 并嵌入 XSpace |
| `raise_error_on_start_failure` | `false`   | collector 启动失败默认不一定抛回 Python |

这些都是当前固定源码里的值，旧 jaxlib 不一定一样。真要较真，还是以 `.pb` 里的 plane、warning 和 `Task Environment` 为准。

## 4. 谁真正记录 kernel：collectors

`ProfilerSession` 只负责把采集组织起来，真正记数据的是这些 collectors：

```text
Host TraceMe collector
Python tracer
HLO metadata collector
GPU / TPU device tracer
PJRT plugin profiler
其他已注册 profiler factory
```

实际最常打交道的几个：

| Collector | 典型内容 | 回答的问题 |
|-----------|------|-------|
| Host tracer | host thread、dispatch、runtime、TraceMe | Python/JAX 何时提交工作？ |
| Python tracer | Python call/return | host 端调用栈花了多久？ |
| Metadata collector | JAX/backend 版本、序列化 HLO proto | 这段 device event 对应什么编译程序？ |
| TPU/libtpu tracer | TPU TensorCore、SparseCore、runtime events | kernel 在设备上何时执行、持续多久？ |
| PJRT plugin tracer | plugin 返回的 serialized XSpace planes | 外部 PJRT backend 提供了哪些 device 数据？ |

TPU/PJRT 这条路径还会多一次 protobuf 交换：

```text
libtpu 或 PJRT plugin
  -> 构造 plugin-local XSpace
  -> 序列化到内存 buffer
  -> outer TpuTracer / PluginTracer 解析 buffer
  -> 把其中的 XPlanes 合入当前 ProfilerSession
```

所以磁盘上的 `.xplane.pb` 不一定由 Python 进程从硬件 packet 逐条编码；TPU device 数据可能已经在 plugin 里过了一次 protobuf ABI。

### 4.1 为什么 `block_until_ready()` 必须在 trace 内

JAX dispatch 是异步的：Python 返回只表示任务已提交，不代表 TPU 上已经跑完。

```python
with jax.profiler.trace("profile-data"):
  y = kernel(x)
  y.block_until_ready()  # 在 collector 停止前等待设备完成
```

把 `block_until_ready()` 放到 `with` 外面，host enqueue 可能记下来了，device collector 却可能在 kernel 跑完前就停了，device event 会缺一段。

测 steady-state，warm-up 放 trace 外；想研究编译开销，就把首次调用放进 trace。两件事别混。

### 4.2 `StepTraceAnnotation` 标记的是采集窗口

要跑多轮，可以给每次 host 侧调用加 step annotation：

```python
with jax.profiler.trace("profile-data"):
  for step in range(3):
    with jax.profiler.StepTraceAnnotation("kernel_step", step_num=step):
      y = kernel(x)
      y.block_until_ready()
```

`StepTraceAnnotation` 本质上是带 `_r=1` 的 host `TraceAnnotation`，用来让 XProf 的 step grouping 识别窗口。它不会把 TPU device event 改名成 `kernel_step`，也不能代替 `XLA Modules` 或 device event 算 kernel 时间；后面写结果时，要说明白自己用的是 host step window 还是 device module window。

## 5. Collectors 把数据汇总到 XSpace

退出 `with` 时，`stop_trace()` 会调：

```text
profile_session.stop_and_export(log_dir)
  -> ProfilerSession::CollectData(&xspace)
```

`XSpace` 是一次 session 的顶层容器，每个 collector 把自己负责的时间线或 metadata 往里面塞。

```text
Host collector ──────┐
Python collector ────┤
Metadata collector ──┼─> XSpace
TPU collector ───────┤
PJRT collector ──────┘
```

先记住最小结构：

```text
XSpace
└── XPlane
    └── XLine
        └── XEvent
```

* `XSpace`：一次 profile 的总容器。
* `XPlane`：一个设备或逻辑域。
* `XLine`：plane 内的一条时间线。
* `XEvent`：时间线上的一次事件实例。

后面找 Pallas kernel，通常是沿着这条路径：

```text
XPlane: /device:TPU:0
XLine:  XLA Modules
XEvent: jit_pallas_gemm(...)
```

## 6. `CollectData()` 整理 XSpace

`CollectData()` 不是把 collector 的结果简单拼起来。单 host 的普通流程会做这些事：

1. 向 `XSpace.hostnames` 添加本机 hostname。
2. 停止所有 collectors。
3. 调用各 collector 的 `CollectData(space)`，向同一个 XSpace 追加 planes。
4. 给缺少 PID 的 plane 添加 process ID。
5. 调用 `PostProcessSingleHostXSpace()`。
6. 把完整 `ProfileOptions` 写入 `Task Environment` plane。

`PostProcessSingleHostXSpace()` 还会：

* 合并 host、Python、TPU runtime 等 host planes；
* 合并同名 planes；
* 把各 line 的时间轴平移到 profiling session 起点；
* 在 `Task Environment` 保存原始 session start/stop wall time；
* 排序 lines 和 events。

这会影响后面读时间：`xplane.proto` 注释里写 `XLine.timestamp_ns` 是 UNIX epoch ns，但 JAX 导出的 XSpace 已经归一化，文件里的值通常相对 session 起点。

## 7. XSpace 怎么变成两个文件

`CollectData()` 结束后，普通模式会调用：

```text
ExportToTensorBoard(xspace, log_dir, ..., also_export_trace_json=true)
```

落盘链路：

```text
post-processed XSpace
  ├─ SaveXSpace()
  │   └─ WriteBinaryProto()
  │       └─ <hostname>.xplane.pb
  │
  └─ ConvertXSpaceToTraceContainer()
      └─ TraceContainerToJson()
          └─ gzip
              └─ <hostname>.trace.json.gz
```

三件事别混：

* `ProfilerSession::CollectData()` 生成并后处理内存中的 XSpace，但不负责文件落盘。
* `.xplane.pb` 的落盘点是 `ExportToTensorBoard()` → `SaveXSpace()` → `WriteBinaryProto()`。
* `.pb` 直接来自 XSpace 的 protobuf 序列化，不经过 JSON 中间态。

默认目录布局：

```text
<log_dir>/plugins/profile/<session-id-or-timestamp>/
  <hostname>.xplane.pb
  <hostname>.trace.json.gz
```

### 7.1 为什么后面主要看 `xplane.pb`

两个文件出自同一份处理后的 XSpace，但定位不同：

| 文件  | 生成方式 | 适用场景 |
|-----|------|------|
| `*.xplane.pb` | XSpace 直接 protobuf 序列化 | 原始事件、metadata、后续 XProf 分析 |
| `*.trace.json.gz` | 选择部分 planes/events 后转为 Chrome Trace JSON | Trace Viewer、Perfetto、有限 fallback |

这版 XLA 只导出 host plane 和匹配到的 GPU/TPU/custom device family，viewer events 上限是 5,000,000，旧 jaxlib 还可能更低。所以 `trace.json.gz` 会缺 plane、缺 internal event、丢 stat，也躲不开 event cap。

所以后面我都只拿 `xplane.pb` 找 kernel；`trace.json.gz` 可以当 fallback，但不把它当成完整统计的唯一来源。

### 7.2 其他可能出现的产物

* `create_perfetto_trace=True` 时，JAX 会读 `trace.json.gz`，去掉顶层 `metadata`，再生成 `perfetto_trace.json.gz`。
* continuous profiling 走 `Stop()` → `SerializeChunks()`，产物主要是 `*.xplane.riegeli`，不是这里说的单个 `xplane.pb`。
* `*.hlo_proto.pb` 默认不会单独写出来：HLO proto 先嵌在 `/host:metadata` plane 里，XProf 之后可以抽取、去重并缓存成独立文件。

### 7.3 “主要数据源”不等于硬件活动绝对无损

`xplane.pb` 是 collectors 交给 session 的 canonical 数据，但不代表硬件活动一点不丢。常见的限制有：

* collector 自己的 event/buffer 上限；
* libtpu/PJRT trace mode 和实现版本；
* collector start/stop/collect 错误；
* trace 结束时仍未完成的异步 device work；
* 进程中断、磁盘写入失败造成的截断文件。

判断采集完不完整，别只看文件在不在。`XSpace.errors/warnings`、plane 集合、目标 kernel 是否出现、时间窗口是否合理，都要看。

## 8. 在 XSpace 中定位一个 kernel

`.pb` 里装的是 XSpace。找 kernel 就是逐层定位：在哪个设备、哪条线、叫什么、何时开始、持续多久。

TPU profile 里常见的入口如下，实际 plane/line 会随 libtpu 版本和采集配置变化：

| 想看什么 | 常见 XPlane | 常见 XLine | 典型 event |
|--------|-----------|----------|----------|
| 整个 JIT module 的 device 窗口 | `/device:TPU:0` | `XLA Modules` | `jit_pallas_gemm(...)` |
| HLO/编译算子 | `/device:TPU:0` | `XLA Ops` | dot、fusion、custom-call |
| kernel 内命名区域 | `/device:TPU:0` | `XLA TraceMe` | `pallas_gemm_dot` |
| SparseCore 工作 | `/device:TPU:<chip> SparseCore <die>` | `Sparse Core Ops` | collective 或 SC op |
| host dispatch/runtime | `/host:CPU` | host thread | dispatch、PJRT runtime |
| HLO 与版本 metadata | `/host:metadata` | 通常无普通 timeline | HLO proto、JAX/backend 版本 |

这里的 `pallas_call(name="pallas_gemm")`、JIT module 名和 `jax.named_scope("pallas_gemm_dot")` 分属不同抽象层，别指望它们出现在同一条 line。分析前先想清楚，要的是整个 module、单个 HLO op，还是 kernel 里的某个 region。

### 8.1 让 Pallas kernel 更容易被找到

采集程序可以同时设置稳定的 `pallas_call` 名称、`jax.named_scope` 区域和 TPU profiling 参数，让 kernel 更容易辨认：

```python
import os


# 必须在 import jax 和 backend 初始化之前设置。
os.environ["LIBTPU_INIT_ARGS"] = (
  os.environ.get("LIBTPU_INIT_ARGS", "")
  + " --xla_enable_custom_call_region_trace=true"
  + " --xla_xprof_register_llo_debug_info=true"
)

import jax
from jax.experimental import pallas as pl


def kernel(lhs_ref, rhs_ref, out_ref):
  with jax.named_scope("pallas_gemm_dot"):
    ...


call = pl.pallas_call(
  kernel,
  out_shape=...,
  name="pallas_gemm",
)
```

`pallas_call(name=...)` 给 kernel call 一个稳定的调试名，但最终 JIT/HLO event 叫什么，还是 XLA lowering 说了算。Mosaic 会在 `jax.named_scope(...)` 边界插 TPU `trace_start` / `trace_stop`，所以 named scope 可以当成更细的 device region。上面的两个 `LIBTPU_INIT_ARGS` 参数用来保留 custom-call region trace 和 Mosaic/LLO debug metadata，必须在 JAX 初始化 libtpu 之前设置。

这些设置让数据更全，但不保证某个名字一定出现在固定 line。稳妥的做法是先枚举实际采集到的 plane、line 和 event，再挑稳定的 selector。

## 9. 事件名都藏在 metadata 里

`XEvent` 一般不直接存长名字，只放一个 `metadata_id`：

```text
event.metadata_id
  -> 当前 XPlane.event_metadata[metadata_id]
  -> XEventMetadata.name / display_name
```

stat 也一样：

```text
stat.metadata_id
  -> 当前 XPlane.stat_metadata[metadata_id]
  -> XStatMetadata.name / description
```

metadata ID 只在所属 XPlane 内有意义，拿 TPU:0 的 ID 去查 TPU:1、SparseCore 或 host plane 的 map，查到的是别的东西。

XProf 的 C++ `XPlaneVisitor` 会同时给 `Name()` 和 `DisplayName()`，Trace Viewer 一般优先显示非空的 `display_name`；JAX `ProfileData` 返回的则是 metadata 里的 `name`。如果 UI 短名和 parser 长名对不上，先看这两个字段，别急着下结论说 event 丢了。

## 10. 算时间要分清 ps 和 ns

普通 timeline event 的绝对时间，就是 line 基准时间加 event 偏移：

```text
start_ps = line.timestamp_ns * 1000 + event.offset_ps
end_ps   = start_ps + event.duration_ps

start_ns    = line.timestamp_ns + event.offset_ps / 1000
duration_ns = event.duration_ps / 1000
end_ns      = start_ns + duration_ns
```

比如：

```text
line.timestamp_ns = 100
event.offset_ps    = 2000
event.duration_ps  = 3000
```

解析出来是：

```text
start_ns    = 102.0
duration_ns = 3.0
end_ns      = 105.0
```

`XEvent.offset_ps` 和 `XEvent.num_occurrences` 是同一个 protobuf `oneof data`。如果 event 是聚合记录，选了 `num_occurrences`，就别把默认的 `offset_ps=0` 当成真实 timeline 起点。

## 11. XPlane protobuf 结构

前面反复出现的 XSpace、XPlane、XLine、XEvent、metadata，核心 schema 位于 XLA checkout 的：

```text
third_party/tsl/tsl/profiler/protobuf/xplane.proto
```

protobuf package 是 `tensorflow.profiler`，顶层消息是 `XSpace`：

```protobuf
syntax = "proto3";

package tensorflow.profiler;

message XSpace {
  repeated XPlane planes = 1;
  repeated string errors = 2;
  repeated string warnings = 3;
  repeated string hostnames = 4;
}

message XPlane {
  int64 id = 1;
  string name = 2;
  repeated XLine lines = 3;
  map<int64, XEventMetadata> event_metadata = 4;
  map<int64, XStatMetadata> stat_metadata = 5;
  repeated XStat stats = 6;
}

message XLine {
  int64 id = 1;
  string name = 2;
  int64 timestamp_ns = 3;
  repeated XEvent events = 4;
  int64 duration_ps = 9;
  int64 display_id = 10;
  string display_name = 11;

  reserved 5, 6, 7, 8;
}

message XEvent {
  int64 metadata_id = 1;

  oneof data {
    int64 offset_ps = 2;
    int64 num_occurrences = 5;
  }

  int64 duration_ps = 3;
  repeated XStat stats = 4;
}

message XStat {
  int64 metadata_id = 1;

  oneof value {
    double double_value = 2;
    uint64 uint64_value = 3;
    int64 int64_value = 4;
    string str_value = 5;
    bytes bytes_value = 6;
    uint64 ref_value = 7;
  }
}

message XEventMetadata {
  int64 id = 1;
  string name = 2;
  bytes metadata = 3;
  string display_name = 4;
  repeated XStat stats = 5;
  repeated int64 child_id = 6;
}

message XStatMetadata {
  int64 id = 1;
  string name = 2;
  string description = 3;
}
```

结构对应关系：

```text
XSpace
├── repeated XPlane planes
│   ├── repeated XLine lines
│   │   └── repeated XEvent events
│   │       ├── metadata_id
│   │       ├── offset_ps 或 num_occurrences
│   │       ├── duration_ps
│   │       └── repeated XStat stats
│   ├── event_metadata[id] -> XEventMetadata
│   ├── stat_metadata[id] -> XStatMetadata
│   └── plane-level XStats
├── errors
├── warnings
└── hostnames
```

## 12. 用 JSON 看懂一个 kernel event

接下来会看到三种 JSON，别把它们当成同一种东西。

### 12.1 标准 protobuf JSON

标准 protobuf JSON 保留原始 ID、metadata map、oneof 和 ps 字段：

```json
{
  "planes": [
    {
      "id": "1",
      "name": "/device:TPU:0",
      "lines": [
        {
          "id": "1",
          "name": "XLA Modules",
          "timestampNs": "100",
          "events": [
            {
              "metadataId": "7",
              "offsetPs": "2000",
              "durationPs": "3000",
              "stats": [
                {
                  "metadataId": "9",
                  "int64Value": "42"
                }
              ]
            }
          ]
        }
      ],
      "eventMetadata": {
        "7": {"id": "7", "name": "jit_pallas_gemm"}
      },
      "statMetadata": {
        "9": {"id": "9", "name": "run_id"}
      }
    }
  ],
  "hostnames": ["demo-host"]
}
```

标准 mapping 有几个容易踩的点：

* field name 默认由 snake_case 转为 lowerCamelCase；
* 64-bit integer 使用十进制字符串，避免 JavaScript number 精度损失；
* integer map key 在 JSON object 中成为字符串 key；
* `bytes` 使用 base64；
* oneof 只输出被选择的字段；
* proto3 默认值可能被省略。

### 12.2 metadata 已解析的 resolved JSON

日常分析更想要的是解析好的名字和时间：

```json
{
  "plane": "/device:TPU:0",
  "line": "XLA Modules",
  "name": "jit_pallas_gemm",
  "start_ns": 102.0,
  "duration_ns": 3.0,
  "end_ns": 105.0,
  "stats": [
    {"name": "run_id", "value": 42}
  ]
}
```

这是解析后的视图，不是 raw protobuf 的无损 JSON：metadata ID 换成了名字，时间算好了，单位也统一了；raw ID、metadata bytes、oneof presence、`errors/warnings` 这些字段可能都不在。

### 12.3 XProf 派生 JSON

XProf 的 `hlo_stats`、`roofline_model`、`memory_profile` 这些 JSON 又是第三类：它们在 XSpace/HLO 上做了预处理、关联和聚合，不是 XPlane schema 的简单 JSON 映射。

## 13. 用 JAX visitor API 读取 `xplane.pb`

JAX 提供了 `ProfileData.from_file()`，可以直接读取 `xplane.pb`：

```python
from pathlib import Path

import jax


path = Path("path/to/host.xplane.pb")
profile = jax.profiler.ProfileData.from_file(str(path))
```

底层 C++ 会把整个文件读进来，调 `XSpace::ParseFromArray()`，再暴露：

```text
profile.planes
  -> plane.name / plane.stats / plane.lines
     -> line.name / line.events
        -> event.name / start_ns / duration_ns / end_ns / event.stats
```

### 13.1 查看 plane 和 line 结构

```python
for plane in profile.planes:
  print(f"plane: {plane.name}")
  for line in plane.lines:
    print(f"  line: {line.name}")
```

### 13.2 定位 Pallas 相关 event

```python
for plane in profile.planes:
  if not plane.name.startswith("/device:TPU:"):
    continue
  for line in plane.lines:
    for event in line.events:
      name = event.name.lower()
      is_pallas_custom_call = (
        "custom_call_target" in name and '"tpu_custom_call"' in name
      )
      if "pallas_gemm" in name or is_pallas_custom_call:
        print(
          plane.name,
          line.name,
          event.name,
          event.start_ns,
          event.duration_ns,
          dict(event.stats),
        )
```

实际 event 名由 lowering 和运行时决定。整个编译模块通常在 `XLA Modules`，单个 Pallas custom call 更可能出现在 `XLA Ops`；应先枚举当前 capture，再按实际名称筛选，不要把示例名称当成固定 ABI。

用下面的 synthetic XSpace 时间字段验证 visitor 的换算：

```text
timestamp_ns = 100
offset_ps    = 2000
duration_ps  = 3000
```

能正确得到：

```text
start_ns    = 102.0
duration_ns = 3.0
end_ns      = 105.0
```

`ProfileData` 暴露的是单个 event。计算 duration sum 或 interval union 前，仍要先采用第 14 节的口径，并明确筛选范围和时间窗口。

### 13.3 想拿 raw protobuf 字段时

JAX `ProfileData` 不会把 raw ID、oneof presence、`errors/warnings`、metadata bytes 全部暴露出来。要完整字段，可以从文章固定版本的 XLA checkout 生成 Python binding。假设源码位于 `/path/to/xla`：

```bash
mkdir -p /tmp/xplane-py

protoc \
  --proto_path=/path/to/xla/third_party/tsl \
  --python_out=/tmp/xplane-py \
  tsl/profiler/protobuf/xplane.proto
```

然后按标准 protobuf JSON 输出：

```python
from pathlib import Path

from google.protobuf.json_format import MessageToJson
from tsl.profiler.protobuf import xplane_pb2


path = Path("path/to/host.xplane.pb")
space = xplane_pb2.XSpace()
space.ParseFromString(path.read_bytes())
print(MessageToJson(space, preserving_proto_field_name=False))
```

运行这段代码时，需要把 `/tmp/xplane-py` 加入 Python 模块搜索路径，并安装与生成代码兼容的 `protobuf` runtime。

`ProfileData.from_file()` 和 generated-protobuf 解码都会把整个文件读进内存。XSpace 超过 1 GB 时，内存占用会明显高于文件大小；内存紧就别把整份文件展开成 Python object 或 JSON，应改用经过验证的流式或 mmap 读取方式，只扫需要的字段。

## 14. 报 device time 之前先定口径

解析到 event，离“正确的 device time”还差一步：口径要先定。

| 指标  | 定义  | 适用场景 | 风险  |
|-----|-----|------|-----|
| 单个 event duration | 一次匹配 event 的 `duration_ns` | 单次 JIT module/kernel region | event 可能只是某个抽象层的窗口 |
| duration sum | 所有匹配 event duration 相加 | 串行、互不嵌套的工作量累计 | 并行和嵌套会重复计时 |
| interval union | 所有匹配时间区间的并集 | 至少一个匹配 event 活跃的 wall-clock | 必须保证时间基准一致 |
| step/module window | `XLA Modules` 中目标 step/module 的起止区间 | 单步端到端 device 时间 | module 名称和 step 选择必须明确 |
| cross-device aggregate | 各 device 的 max、union、平均或逐 device 报告 | 多 chip 分析 | 不能未经定义直接相加 |

比如只看 TPU:0 上某个 Pallas JIT module 的单次 device 窗口，就先定：

```text
plane = /device:TPU:0
line  = XLA Modules
event = 目标 jit module
```

如果看的是一组细粒度 `XLA Ops` 或 `XLA TraceMe` region，就先定好 step/module window，把跨窗口 event clip 掉，再算 interval union。

报任何 device-time 数字，至少把下面这些一起写出来：

* plane；
* line；
* event selector；
* step/window；
* duration sum 或 interval union；
* 单位；
* 是否跨 chip 聚合；
* 是否截断 samples 或输入事件。

## 15. 上层 XProf 又做了什么

能从 raw XSpace 里把 event 捞出来以后，XProf 这边就简单了：它不是重新定义 protobuf，而是在 XSpace 和 HLO 上做一层预处理和聚合，供 UI 和性能诊断使用：

```text
*.xplane.pb
  -> XProf SessionSnapshot
  -> preprocess XSpace
     -> step grouping
     -> derived timeline
     -> HLO / program association
  -> tool processors
     -> trace_viewer / trace_viewer@
     -> overview_page
     -> hlo_stats
     -> roofline_model
     -> memory_profile
     -> graph_viewer / memory_viewer
```

### 15.1 Trace Viewer 不是把 XEvent 原样画出来

XProf 的非 streaming `trace_viewer` 先调用：

```text
PreprocessSingleHostXSpace(step_grouping=true, derived_timeline=true)
```

再把处理后的 XSpace 转成 trace events。streaming `trace_viewer@` 走的是另一套 trace container 和索引，所以 UI 里的 line、group、derived event 和磁盘上的原始 XEvent 不保证一一对应。

### 15.2 `hlo_stats` self time 不等于单次 kernel wall-clock

XProf 把 XSpace 和嵌入或独立的 HLO proto 关联起来，生成 HLO/op 聚合统计。`self_time` 是某类 HLO op 去掉子事件后的累计成本，不是某次 kernel module 在 timeline 上的起止窗口。

类似的还有：

* `roofline_model` 结合 FLOP、访存量和设备峰值做模型分析；
* `memory_profile` 聚合 runtime allocator 数据；
* `memory_viewer` / `graph_viewer` 还依赖 HLO proto；
* `OpStats` 是由 XSpace 派生的聚合结果，不是 XPlane 中原样存在的 event 列表。

## 16. 从 kernel 到 XProf 的完整数据流

```text
Pallas kernel
  ├─ pallas_call(name=...)
  ├─ jax.named_scope(...)
  └─ JIT / HLO / Mosaic metadata
          │
          v
jax.profiler.trace()
  ├─ start_trace()
  │   ├─ 初始化 backend
  │   ├─ 注册 TPU/PJRT profiler
  │   └─ 启动 ProfilerSession + collectors
  │
  ├─ kernel execution + block_until_ready()
  │   ├─ host/Python events
  │   ├─ TPU device events
  │   └─ HLO metadata
  │
  └─ stop_trace()
      ├─ CollectData(XSpace)
      ├─ merge / normalize / annotate / sort
      └─ ExportToTensorBoard()
          ├─ *.xplane.pb
          └─ *.trace.json.gz
                  │
                  v
*.xplane.pb
  ├─ JAX ProfileData -> resolved event JSON
  ├─ generated protobuf -> raw schema JSON
  ├─ XProf -> trace/HLO/roofline/memory derived views
```

到这里，主线其实就一条：先顺着 `jax.profiler.trace()` 的生命周期搞清楚数据怎么来；把 `xplane.pb` 当成 collectors 实际交付的 canonical XSpace；报数字前先定义 plane、line、event 和时间窗口，最后再让 XProf 的派生视图解释结果。

## 17. XProf CLI 也能在终端解析部分数据

补充一句：这篇文章固定的 XProf（tag `xprof-v2.21.1`，commit `403109fc2d5b6e1f87cdd26379adefbcb94b89be`）自带 CLI，已经可以在终端里直接解析一部分 `.pb` 数据，快速看 plane / line / event 和统计值，不用为了一个小问题把整套 Web UI 起起来。

CLI 适合快速抽查；`trace_viewer`、`hlo_stats`、`roofline_model` 这些完整视图还是交给 XProf 工具链。解析旧文件时，最好把 CLI 版本和生成 `.pb` 的 jaxlib/XLA 版本一起记下来，对不上再排查。

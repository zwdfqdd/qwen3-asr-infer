# PERF-ARCH-002：TensorRT FP16 ForcedAligner

`PERF-ARCH-001` 的 vLLM token-classify 原始吞吐已验证为 +188.1%，但生产 oracle 严格门禁仅
6/11；块 mask 和反向跨窗诊断都未达到 11/11。继续逐层修复 vLLM 兼容性的投入产出不足，路线
停止。按项目负责人新决策，提前启动独立 TensorRT FP16 路线。

本目录当前只有**导出可行性探针**，尚未安装 ONNX/TensorRT、构建 engine、创建服务或修改生产
链路。探针通过不代表 TensorRT 已可用。

## 边界

- 当前稳定 vLLM 0.19.1 环境、`src/`、网关、启动脚本和生产模型目录均不修改。
- 模型只读取 ModelScope 固定 revision：`Qwen/Qwen3-ForcedAligner-0.6B`，revision
  `cf1c50164ea3ac48240d12bef5ead74aee0720cc`。
- CPU 保留官方 `encode_timestamp()`、processor、timestamp token mask、`parse_timestamp()`、
  单调性和边界校验；候选编译范围只覆盖 `model.thinker(...).logits`。
- 当前只评估 FP16。固定 shape 已通过，现推进动态 shape T1；BF16、INT8 W8A8、TensorRT
  profile 和独立服务仍须在动态导出边界明确后另行推进。
- 11 语种、逐项 1 ms、成功率 100% 和原始吞吐 +30% 门禁不放宽。
- 报告、音频、文本和导出图可能包含样本派生信息，不提交 Git。

## 为什么先做 `torch.export`

TensorRT 不是权重格式替换。官方 thinker 内部包含音频塔、音频 embedding 注入、文本主干和分类头，
并有数据相关 shape/索引操作。若固定 batch=1、固定 shape 都不能由 PyTorch 捕获，就不应先投入
ONNX parser、TensorRT profile 或服务封装。

探针分阶段回答三个问题：

1. 当前固定源码边界是否仍是 `audio_tower → model → lm_head`，且官方 `align()` 只把 thinker
   logits 作为 GPU 模型输出；
2. 目标 A10 上 FP16 thinker 能否被 `torch.export` 固定 shape 捕获，并在回放时保持 timestamp
   位置 argmax 完全一致；
3. batch 固定为 1 时，音频帧数与文本长度能否使用同一 strict 动态图覆盖 10 秒和 7 秒异 shape
   输入，并保持各自 timestamp argmax。

## 1. 本地静态探针

不导入 qwen-asr、不需要 GPU：

```bash
python experiments/perf_arch_002/probe.py static \
  --source-root .tmp-qwen3-asr-official \
  --output-json performance-results/PERF-ARCH-002/static-probe.json
```

检查 thinker 成员、forward 调用顺序和官方 `align()` 的 thinker 调用契约。任一项失败即停止，先
重新审计固定版本源码。2026-08-19 本地静态探针已通过；目标 A10 固定 shape 也已通过，当前
继续第 3 节动态 T1。

## 2. 目标 A10 固定 shape 探针

先停止网关、vLLM 和其他占卡进程，确认显存释放；不安装任何新依赖：

```bash
python experiments/perf_arch_002/probe.py target \
  --model models/qwen3-forced-aligner-0.6b/pt \
  --audio test_data/audio_16000_10s.wav \
  --text test_data/audio_16000_10s.txt \
  --language Chinese \
  --output-json performance-results/PERF-ARCH-002/target-export-fixed-b1.json
```

目标 A10 首轮已执行且严格导出失败。eager 前向成功；首个阻塞位于官方音频塔
`[self.n_window * 2] * chunk_num.sum()`：运行时张量值被用于 Python 列表重复次数，
`torch.export(strict=True)` 无法自动提取专门化整数。这不是 TensorRT 解析失败，也不能用
`strict=False` 绕过。

下一轮只增加固定 shape 音频长度专门化：把当前 batch=1 输入的 chunk 长度、CNN 后长度和
`cu_seqlens` 固化为 Python 常量，先对改写前后 eager logits 与 timestamp argmax 做比较，通过后
才执行同一个 strict export：

```bash
python experiments/perf_arch_002/probe.py target \
  --model models/qwen3-forced-aligner-0.6b/pt \
  --audio test_data/audio_16000_10s.wav \
  --text test_data/audio_16000_10s.txt \
  --language Chinese \
  --specialize-audio-shapes \
  --output-json performance-results/PERF-ARCH-002/target-export-fixed-b1-specialized-01.json
```

该候选已在目标 A10 执行：`feature_length=1000`，10 个 100 帧 chunk，CNN 后均为 13，
`cu_segments=[104,26]`。改写前后 logits 最大绝对差为 **0.0**，timestamp argmax 完全一致，
证明音频 shape 专门化语义等价。strict export 随后越过音频塔，新的首个阻塞位于文本主干
`create_causal_mask()`：Transformers 4.57.6 在 fake tensor 捕获时通过 vmap 索引二维布尔 mask，
触发不支持的隐式 `.item()`；`always_classified is unsupported` 是伴随告警，不是主因。

第二候选修正后已在目标 A10 通过。当前 SDPA、无 padding 的 eager `create_causal_mask()`
合法返回 `None`，探针固定绕过 mask vmap 并保持 attention 内核原生 causal 语义。固定 shape
完整结果：输入 `input_ids/attention_mask=[1,312]`、`feature_attention_mask=[1,1000]`、
`input_features=[1,128,1000]`；音频改写 eager、文本 mask 改写 eager、导出图回放相对原始
eager 的 logits 最大差均为 **0.0**，timestamp argmax 均完全一致；
`torch.export(strict=True)` 成功，`summary.fixed_shape_export=true`。

捕获仍出现 `_export_root` potential side-effect warning。该警告未阻止本轮 strict export 和回放，
但已作为动态 shape/TensorRT 风险保留，不能描述为无风险。两个固定开关只用于导出可行性实验，
不修改安装包源码或生产模型。

## 3. 目标 A10 动态 shape 与异 shape 回放

固定 shape 通过后进入 T1。当前只尝试 **batch=1 固定**、mel bins 固定为 128、文本长度
动态，以及音频 `audio_frames = 100 × audio_chunks` 的派生动态维；不是动态 batch，也暂不支持
非 100 帧整倍数尾块。`audio_chunks` 范围为 2～32，即音频帧数 200～3200；文本范围为 4～模型
`max_position_embeddings`。主样本使用 10 秒音频，同一导出图必须再回放 7 秒异 shape 输入：

```bash
python experiments/perf_arch_002/probe.py target \
  --model models/qwen3-forced-aligner-0.6b/pt \
  --audio test_data/audio_16000_10s.wav \
  --text test_data/audio_16000_10s.txt \
  --language Chinese \
  --dynamic-shapes \
  --specialize-text-mask \
  --replay-audio test_data/audio_16000_7s.wav \
  --replay-text test_data/audio_16000_7s.txt \
  --output-json performance-results/PERF-ARCH-002/target-export-dynamic-b1-10s-to-7s.json
```

动态音频改写只接受 batch=1、无 padding、SDPA 和完整 100 帧 chunk：输入按派生维直接 reshape
为动态 chunk batch 后统一卷积，不再生成动态 padding 或尾块截断；因当前 PyTorch SDPA 路径
实际不消费 `cu_seqlens` 窗口边界，候选使用单个全序列 segment。该语义必须分别通过 10 秒和
7 秒原始 eager 对照，不能假定正确。首轮任意帧范围已证明 shape solver 无法接受；整倍数候选
只回答当前两个样本和 2～32 个完整 chunk，不代表生产任意时长已覆盖。

目标 A10 重跑结果为**通过**：两组改写 eager、strict 动态导出、主输入回放和 7 秒异 shape
回放均通过探针门禁，`summary.dynamic_shape_export=true`。末尾 `swigvarlink` DeprecationWarning
不影响导出或回放；但该结果仍严格限定在完整 100 帧 chunk 域。探针要求：

1. 两组原始 eager 与动态音频改写 eager 均保持 shape、finite 和 timestamp argmax；
2. 两组文本 mask 改写继续通过同一门禁；
3. `torch.export(strict=True, dynamic_shapes=...)` 成功；
4. 同一导出图对 10 秒与 7 秒输入均回放成功，timestamp argmax 与各自原始 eager 一致；
5. 报告保留 logits 逐位一致性、最大差、动态约束、两组输入哈希和全部导出警告。

任一步失败即记录首个新阻塞；不得退回固定 shape、不得改为 `strict=False`，也不安装
ONNX/TensorRT 试错。固定条件仍为 qwen-asr 0.0.6、CUDA 12.x、FP16、batch=1、真实 ≤32 秒
16 kHz 单声道音频。报告不保存正文，但音频、文本、报告和导出图仍视为可能包含样本派生信息，
不提交 Git。

## 4. T1.1 固定尾帧余数 + 动态完整 chunk 数

受限整百帧图通过后，下一单变量是非整百帧尾块。为避免 shape solver 再次面对动态余数，单个
导出图固定 `audio_tail_frames`，仅令完整 chunk 数动态：

```text
audio_frames = 100 × audio_full_chunks + fixed_tail_frames
```

主输入与回放必须具有相同 mel 尾帧余数。首次使用 1.602188 秒样本时，两组 eager 门禁通过，
但 `audio_full_chunks=1` 被 PyTorch 专门化，strict export 报派生维被固化为 160。T1.1 因此将
`audio_full_chunks` 下限固定为 2，短音频另立阶段。11.334188 秒中文样本与追加 1 秒尾部静音
的目标 A10 重跑已通过：两组 eager、strict 动态导出和异 shape 回放均通过门禁。复现命令为：

```bash
rm -f /tmp/perf_arch_002_tail_replay.wav
python -c 'import numpy as np, soundfile as sf; p="test_data/bb7575d0c350726cc1e85d729ab261ac.wav"; x,sr=sf.read(p,dtype="float32"); sf.write("/tmp/perf_arch_002_tail_replay.wav", np.pad(x,(0,sr)), sr, subtype="FLOAT")'

python experiments/perf_arch_002/probe.py target \
  --model models/qwen3-forced-aligner-0.6b/pt \
  --audio test_data/bb7575d0c350726cc1e85d729ab261ac.wav \
  --text test_data/bb7575d0c350726cc1e85d729ab261ac.txt \
  --language Chinese \
  --dynamic-shapes \
  --specialize-text-mask \
  --replay-audio /tmp/perf_arch_002_tail_replay.wav \
  --replay-text test_data/bb7575d0c350726cc1e85d729ab261ac.txt \
  --output-json performance-results/PERF-ARCH-002/target-export-dynamic-tail-b1-11p3s-to-12p3s.json

rm -f /tmp/perf_arch_002_tail_replay.wav
```

探针会先检查两组实际 processor tensor 的 `audio_tail_frames` 完全相同，再执行两组 eager 等价、
strict 动态导出和异 shape 回放。固定尾帧模式通过仍不等于任意余数单图支持。

## 5. T1.2 CNN 尾块输出桶规范化

T1.1 若按 99 个非零尾帧余数分别建图，维护成本不可接受。三个 stride=2 的卷积把尾帧分成
13 个输出长度桶：1～8 帧输出 1，9～16 帧输出 2，依此类推，89～96 帧输出 12，97～99 帧
输出 13。T1.2 只在**同一个 CNN 输出桶内**把尾帧零填充到桶上界；例如 tail=33 与 tail=37
都输出 5，分别补到 canonical tail=40。原始 eager 仍使用未规范化输入，候选 eager/export 使用
规范化输入，且该模式把 logits 门禁收紧为逐位一致、最大差 0.0。

目标 A10 复现命令（2026-08-20 已通过）：

```bash
rm -f /tmp/perf_arch_002_tail_bucket_replay.wav
python -c 'import numpy as np, soundfile as sf; p="test_data/bb7575d0c350726cc1e85d729ab261ac.wav"; x,sr=sf.read(p,dtype="float32"); sf.write("/tmp/perf_arch_002_tail_bucket_replay.wav", np.pad(x,(0,round(sr*1.04))), sr, subtype="FLOAT")'

python experiments/perf_arch_002/probe.py target \
  --model models/qwen3-forced-aligner-0.6b/pt \
  --audio test_data/bb7575d0c350726cc1e85d729ab261ac.wav \
  --text test_data/bb7575d0c350726cc1e85d729ab261ac.txt \
  --language Chinese \
  --dynamic-shapes \
  --specialize-text-mask \
  --canonicalize-tail-bucket \
  --replay-audio /tmp/perf_arch_002_tail_bucket_replay.wav \
  --replay-text test_data/bb7575d0c350726cc1e85d729ab261ac.txt \
  --output-json performance-results/PERF-ARCH-002/target-export-dynamic-tail-bucket5-b1.json

rm -f /tmp/perf_arch_002_tail_bucket_replay.wav
```

目标 A10 实测原始主输入为 1133 帧、tail=33，补 7 帧到 1140；原始回放为 1237 帧、
tail=37，补 3 帧到 1240。两者 CNN `output_length=5`、canonical tail=40。主/回放音频改写、
文本 mask 和 strict 动态图回放共六处比较均 logits 逐位一致、最大差 0.0，36 个 timestamp
bucket argmax 全部一致，`summary.dynamic_shape_export=true`。候选音频 shape 仍为 1140/1240，
探针现显式拒绝规范化后主/回放 shape 完全相同的无效回放。

随后目标 A10 输出桶 13 也通过：原始主输入 1197 帧、tail=97，补 2 帧到 1199；原始回放
1298 帧、tail=98，补 1 帧到 1299。两者 canonical tail=99、CNN `output_length=13`；候选
`tail_padding=1`、`tail_output_drop=0`，覆盖尾块输出长度等于完整 chunk 的边界。主/回放六处
比较仍全部 logits 逐位一致、最大差 0.0，36 个 timestamp bucket argmax 一致，strict 动态导出
和 1199→1299 异 shape 回放通过。

目标 A10 输出桶 1 随后也通过：原始主输入 1201 帧、tail=1，补 7 帧到 1208；原始回放
1308 帧、tail=8，无需补帧。两者 canonical tail=8、CNN `output_length=1`；候选
`tail_padding=92`、`tail_output_drop=12`，覆盖最大输出裁剪边界。主/回放六处比较仍全部
logits 逐位一致、最大差 0.0，36 个 timestamp bucket argmax 一致，strict 动态导出和
1208→1308 异 shape 回放通过。三轮均保留 `_export_root` potential side-effect warning。

其余 10 桶随后按独立进程、独立 strict export 顺序执行，10/10 全部通过；脚本逐轮校验真实 tail、
canonical tail、输出长度、裁剪量、六处 logits 逐位一致、timestamp argmax 和 warning。最终矩阵：

| 输出桶 | 实测原始 tail | canonical tail | `tail_padding` | `tail_output_drop` |
|---:|---:|---:|---:|---:|
| 1 | 1/8 | 8 | 92 | 12 |
| 2 | 9/16 | 16 | 84 | 11 |
| 3 | 17/24 | 24 | 76 | 10 |
| 4 | 25/32 | 32 | 68 | 9 |
| 5 | 33/37 | 40 | 60 | 8 |
| 6 | 41/48 | 48 | 52 | 7 |
| 7 | 49/56 | 56 | 44 | 6 |
| 8 | 57/64 | 64 | 36 | 5 |
| 9 | 65/72 | 72 | 28 | 4 |
| 10 | 73/80 | 80 | 20 | 3 |
| 11 | 81/88 | 88 | 12 | 2 |
| 12 | 89/96 | 96 | 4 | 1 |
| 13 | 97/98 | 99 | 1 | 0 |

13/13 桶的主/回放音频改写、文本 mask 和 strict 动态图回放均 logits 逐位一致、最大差 0.0，
timestamp argmax 一致，且都保留 `_export_root` warning。T1.2 因此完成 1～99 非零尾帧余数的
CNN 输出桶覆盖；这仍不覆盖 `audio_full_chunks<2` 的短音频。

不能把所有尾块统一补到完整 100 帧：官方音频 attention 是非 causal 全局双向 attention，跨桶
补帧会保留额外 CNN token 并可能污染有效前缀。T1.2 的规范化边界因此严格限定在 CNN 输出长度
不变的桶内。

## 6. T1.3 短音频分域动态图

首个“完整 chunk 规范化”候选已在目标 A10 被拒绝。原始主输入 160 帧
（`audio_full_chunks=1`）规范化为 300 帧后，音频 eager logits 逐位一致、最大差 0.0；但原始
回放输入 60 帧（`audio_full_chunks=0`）规范化为 200 帧后，logits 最大差 1.4296875，timestamp
argmax 也不一致，因此在 export 前触发 `AudioShapeReplayMismatch`。不得放宽门禁继续该路线。

根因是官方单 `<100` 帧输入只有一个 chunk，`pad_sequence` 的实际宽度就是原始 `T`，三层
Conv2d 直接在该宽度上计算；而 160 帧输入会拆成 `[100, 60]`，60 帧尾块因同批存在 100 帧
chunk 才会先补到 100。补帧后再裁 CNN token 无法恢复 full0 的原生卷积右边界。旧开关
`--normalize-short-tail-to-full-chunks` 只保留为失败复现路径，不再作为可用候选。

T1.3 因此拆分 `audio_full_chunks=0` 和 `audio_full_chunks=1`。当前新增
`--direct-short-single-chunk-bucket`，先处理 full0：不补帧、不裁输出，直接复刻官方
`[128,T] → [1,1,128,T] → 3×Conv2d → conv_out → position → 单序列 SDPA`；同一 CNN 输出
桶内直接声明 `audio_frames` 动态。首轮验证输出桶 8 的 57→64 帧：

```bash
rm -f /tmp/perf_arch_002_short_0p57.wav /tmp/perf_arch_002_short_0p64.wav
python -c 'import soundfile as sf; p="test_data/bb7575d0c350726cc1e85d729ab261ac.wav"; x,sr=sf.read(p,dtype="float32"); sf.write("/tmp/perf_arch_002_short_0p57.wav",x[:round(sr*0.57)],sr,subtype="FLOAT"); sf.write("/tmp/perf_arch_002_short_0p64.wav",x[:round(sr*0.64)],sr,subtype="FLOAT")'

python experiments/perf_arch_002/probe.py target \
  --model models/qwen3-forced-aligner-0.6b/pt \
  --audio /tmp/perf_arch_002_short_0p57.wav \
  --text test_data/bb7575d0c350726cc1e85d729ab261ac.txt \
  --language Chinese \
  --dynamic-shapes \
  --specialize-text-mask \
  --direct-short-single-chunk-bucket \
  --replay-audio /tmp/perf_arch_002_short_0p64.wav \
  --replay-text test_data/bb7575d0c350726cc1e85d729ab261ac.txt \
  --output-json performance-results/PERF-ARCH-002/target-export-dynamic-short-full0-bucket8-b1.json

rm -f /tmp/perf_arch_002_short_0p57.wav /tmp/perf_arch_002_short_0p64.wav
```

目标 A10 实测两组 contract 均为 `audio_full_chunks=0`，帧数 57/64，CNN 输出长度均为 8；
动态图范围为 57～64。桶 8 已通过六处 logits 逐位一致、最大差 0.0、timestamp argmax 一致、
`torch.export(strict=True)` 及同图 57→64 回放门禁。终端末尾的 `swigvarlink`
DeprecationWarning 是非致命退出警告。该结果只证明 full0 桶 8；随后仍需覆盖其余 12 个 full0
桶，并单独处理 full1。剩余 full0 桶必须继续按独立进程、独立 strict export 顺序执行：

```bash
for spec in \
  "1 1 8" "2 9 16" "3 17 24" "4 25 32" "5 33 40" "6 41 48" \
  "7 49 56" "9 65 72" "10 73 80" "11 81 88" "12 89 96" "13 97 99"
do
  set -- $spec
  bucket="$1"; lo="$2"; hi="$3"
  main="/tmp/perf_arch_002_short_${lo}.wav"
  replay="/tmp/perf_arch_002_short_${hi}.wav"
  report="performance-results/PERF-ARCH-002/target-export-dynamic-short-full0-bucket${bucket}-b1.json"
  rm -f "$main" "$replay"
  python -c 'import soundfile as sf,sys; p="test_data/bb7575d0c350726cc1e85d729ab261ac.wav"; x,sr=sf.read(p,dtype="float32"); sf.write(sys.argv[3],x[:round(sr*int(sys.argv[1])/100)],sr,subtype="FLOAT"); sf.write(sys.argv[4],x[:round(sr*int(sys.argv[2])/100)],sr,subtype="FLOAT")' "$lo" "$hi" "$main" "$replay"
  python experiments/perf_arch_002/probe.py target \
    --model models/qwen3-forced-aligner-0.6b/pt \
    --audio "$main" \
    --text test_data/bb7575d0c350726cc1e85d729ab261ac.txt \
    --language Chinese \
    --dynamic-shapes \
    --specialize-text-mask \
    --direct-short-single-chunk-bucket \
    --replay-audio "$replay" \
    --replay-text test_data/bb7575d0c350726cc1e85d729ab261ac.txt \
    --output-json "$report" || { status=$?; rm -f "$main" "$replay"; exit "$status"; }
  python -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); d=r["dynamic_shape"]; p=d["direct_short_bucket_plan"]; print({"bucket":int(sys.argv[2]),"frames":[p["main_audio_frames"],p["shape_replay_audio_frames"]],"output_length":p["output_length"],"passed":r["summary"]["passed"],"export_root_warning":r["export"]["export_root_side_effect_warning"]})' "$report" "$bucket"
  rm -f "$main" "$replay"
done
```

## 判定

- **完整动态输入域通过**：13 个尾帧输出桶、完整 chunk 和 full0/full1 短音频均完成两组 eager
  改写与同一 strict 动态图回放 timestamp argmax 门禁；然后才建立独立精确固定的
  ONNX/TensorRT 环境。
- **动态不通过**：保留首个阻塞算子、shape 约束或数据相关控制流，不静默降级，不重复固定实验。
- 即使动态 shape 通过，也不能进入服务集成；后续仍需 11 语种 oracle、逐项 1 ms、batch
  1/8/16/32 原始性能、端到端 +20%、成功率 100% 和故障门禁。

## 当前状态

- ARCH-001：拒绝，性能 spike 已完成但 11 语种语义门禁失败，继续修复不划算。
- ARCH-002：测试中；固定 batch=1、固定 shape strict export 已通过且三处 logits 最大差均为
  0.0。动态首轮任意帧范围在 shape guard 阶段失败；收紧到 2～32 个完整 100 帧 chunk 后，
  10 秒与 7 秒的改写 eager、strict 动态导出和同一图双 shape 回放已全部通过。固定尾帧 T1.1
  也已通过一个余数 profile；T1.2 已在目标 A10 完成 13/13 个 CNN 输出桶，全部六处 logits
  比较逐位一致、最大差 0.0，timestamp argmax 一致。T1.3 首个完整 chunk 规范化候选已因
  full0 回放最大差 1.4296875 且 timestamp argmax 不一致被拒绝；full0 原生单 chunk 动态
  宽度探针的目标 A10 桶 8（57→64 帧）已通过六处逐位 logits、timestamp argmax、strict
  export 和同图回放门禁。该结果只覆盖桶 8；full0 其余 12 桶与 full1 分域通过前仍不进入
  ONNX/TensorRT。
- 生产链：保持 PyTorch ForcedAligner、batch=32 和现有可选 MPS，不做静默切换。

# Qwen3-ASR vLLM 推理服务

面向 Linux + NVIDIA CUDA 的单 GPU 高吞吐语音识别服务。GPU 数据面使用原生 vLLM
Speech-to-Text；独立 aiohttp CPU 网关提供长音频切分、Base64 业务兼容接口、动态热词
Prompt、错误契约与健康/指标代理。v2.1.0 对外只暴露 `8080`。

```text
客户端 → aiohttp 网关 :8080
           ├─ ≤32 秒：原字节直送
           ├─ >32 秒：最长 32 秒 PCM16 WAV 分片、受控并发、顺序合并
           ├─ /chinese_asr 协议转换与 Prompt 热词
           └─ ForcedAligner 有界跨请求动态微批 → 字/词级及句级时间戳
       → vLLM :8081（仅 127.0.0.1）
       → Qwen3-ASR-0.6B / continuous batching / CUDA Graph
```

## 当前版本能力（v2.1.0）

| 能力 | 状态 |
|---|---|
| `/chinese_asr` Base64 JSON | 已实现 |
| `/v1/audio/transcriptions` multipart 文本转写 | 已实现；仅有限兼容，不宣称完整 OpenAI API |
| 0～7500 秒音频、超过 32 秒固定切分 | 已实现；7500 秒场景的内存与尾延迟仍待实测 |
| 动态热词 | 已实现；Qwen3-ASR Prompt 软偏置，不保证强制命中 |
| 主模型标点 | 已实现；直接使用 Qwen3-ASR 输出，不使用 CT-Transformer |
| 主模型语种识别 | 已实现；30 种语言与 22 种中国方言/口音，逐物理分片填充 `slid` |
| 分片级时间边界 | 始终提供；关闭或跳过对齐时不是字/词级时间戳 |
| ForcedAligner 字/词级及句级时间戳 | 已实现；仅 `/chinese_asr`、仅 11 种语言，默认开启 |
| 结构化请求追踪 | 已实现；保留统计摘要，并记录受限 `input_json`/`result_json`；输入 Base64 只保留前 64 字节 |
| VAD、流式 HTTP 网关、说话人识别 | 未实现；说话人识别需独立模型 |

## 语种与方言支持

### Qwen3-ASR 主模型

主模型支持以下 **30 种语言**。`/chinese_asr` 的 `language` 可使用英文名称或括号中的 ISO
代码显式指定；不传或传空值时由模型逐物理分片自动检测：

- Chinese（中文，`zh`）、English（英语，`en`）、Cantonese（粤语，`yue`）、Arabic（阿拉伯语，`ar`）
- German（德语，`de`）、French（法语，`fr`）、Spanish（西班牙语，`es`）、Portuguese（葡萄牙语，`pt`）
- Indonesian（印度尼西亚语，`id`）、Italian（意大利语，`it`）、Korean（韩语，`ko`）、Russian（俄语，`ru`）
- Thai（泰语，`th`）、Vietnamese（越南语，`vi`）、Japanese（日语，`ja`）、Turkish（土耳其语，`tr`）
- Hindi（印地语，`hi`）、Malay（马来语，`ms`）、Dutch（荷兰语，`nl`）、Swedish（瑞典语，`sv`）
- Danish（丹麦语，`da`）、Finnish（芬兰语，`fi`）、Polish（波兰语，`pl`）、Czech（捷克语，`cs`）
- Filipino（菲律宾语，`fil`）、Persian（波斯语，`fa`）、Greek（希腊语，`el`）、Hungarian（匈牙利语，`hu`）
- Macedonian（马其顿语，`mk`）、Romanian（罗马尼亚语，`ro`）

主模型还声明支持以下 **22 种中国方言/口音的自动识别**；这些标签可能出现在响应 `slid`，
但不能作为当前接口的显式 `language` 参数：

- Anhui（安徽）、Dongbei（东北）、Fujian（福建）、Gansu（甘肃）、Guizhou（贵州）
- Hebei（河北）、Henan（河南）、Hubei（湖北）、Hunan（湖南）、Jiangxi（江西）
- Ningxia（宁夏）、Shandong（山东）、Shaanxi（陕西）、Shanxi（山西）、Sichuan（四川）
- Tianjin（天津）、Yunnan（云南）、Zhejiang（浙江）
- Cantonese (Hong Kong accent)（粤语香港口音）
- Cantonese (Guangdong accent)（粤语广东口音）
- Wu language（吴语）、Minnan language（闽南语）

### Qwen3 ForcedAligner

ForcedAligner 只支持以下 **11 种标准语言**及对应别名：

- Chinese（中文，`zh`）、English（英语，`en`）、Cantonese（粤语，`yue`）
- French（法语，`fr`）、German（德语，`de`）、Italian（意大利语，`it`）
- Japanese（日语，`ja`）、Korean（韩语，`ko`）、Portuguese（葡萄牙语，`pt`）
- Russian（俄语，`ru`）、Spanish（西班牙语，`es`）

Aligner 不直接接受上述 22 种方言/口音标签。分片 `slid` 为空或不是这 11 种标准标签时，服务
不会猜测、映射或回退中文，而是保留 ASR 文本与物理分片边界、返回 `words=[]`，并在顶层
`message` 中列出跳过的分片和语种。

## 快速开始

必须使用独立环境，不与 Paraformer、funasr 或 TensorRT 推理栈混装。默认开启 ForcedAligner，
因此推理和 Aligner 两套精确依赖都必须安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-infer.txt
python -m pip install -r requirements-aligner.txt
python -m pip check
bash run.sh
```

`run.sh` 会仅从 ModelScope 检查或下载主模型和 ForcedAligner 的固定 revision。若要关闭时间戳，
必须显式设置两个运行开关：

```bash
ENABLE_WORD_TIMESTAMP=false ENABLE_SENTENCE_TIMESTAMP=false bash run.sh
```

Dockerfile 默认 `INSTALL_ALIGNER=true`，并执行 `COPY models models`、`COPY tests tests`、
`COPY test_data test_data`。构建前应确保两个模型目录均包含固定 revision 权重及 manifest：

```bash
python scripts/download_model.py
python scripts/download_model.py \
  --model-id Qwen/Qwen3-ForcedAligner-0.6B \
  --dir models/qwen3-forced-aligner-0.6b/pt \
  --revision cf1c50164ea3ac48240d12bef5ead74aee0720cc

docker build --no-cache --build-arg INSTALL_ALIGNER=true \
  -t qwen3-asr-infer:0.19.1 .
docker run -d --gpus '"device=0"' --restart=always \
  -p 8080:8080 qwen3-asr-infer:0.19.1
```

主模型固定 revision 为 `4ce9cc728b473a5aedbe7b6e1ea45646316824dc`。下载器校验文件大小
与 SHA256，并在模型目录写入 `.modelscope-manifest.json`；镜像启动时仍会执行完整性校验。

`/chinese_asr` 不传 `language` 时由主模型逐物理分片自动检测并填入 `slid`。主模型官方范围
是 30 种语言与 22 种中国方言/口音；默认 ForcedAligner 仅支持其中 11 种语言。支持的分片
执行真实对齐；Malay、Arabic 等不受支持的分片仍随整条请求返回 HTTP 200，保留 ASR 文本、
`slid` 和物理分片边界，但 `words=[]`，顶层 `message` 明确列出跳过项。服务不会伪造时间戳
或回退中文。完整列表和部分对齐契约见 [API 文档](docs/API文档.md)。

启动后：

```bash
curl http://127.0.0.1:8080/health
python tests/test_chinese_asr_single.py --audio test_data/audio_16000_10s.wav
```

## 默认输入与性能基线

当前默认 `AUDIO_CHUNK_SECONDS=32`、`MAX_AUDIO_SECONDS=7500`、`MAX_UPLOAD_MB=229`、
`MAX_JSON_BODY_MB=308`。两个字节上限按 7500 秒 16 kHz 单声道 PCM16 WAV 推导：音频约
228.88 MiB，Base64 后约 305.18 MiB；multipart 仍由请求体和音频上限共同约束。时长与两层字节限制分别保护
解码后音频、HTTP 接收和进程内存。

当前默认 `CHUNK_CONCURRENCY=3`、`LONG_CHUNKS_IN_FLIGHT=96`。网关到 vLLM 的连接池上限
保持 100，空闲连接默认保留 4 秒；目标 GPU 六轮累计 12000/12000，已验证不会复用失效空闲
连接。固定依赖中的 Fast Tokenizer 由 `src/sitecustomize.py` 串行保护完整公共调用，消除并发
Chat/多模态预处理触发的 `RuntimeError: Already borrowed` 和上游 HTTP 400；针对性验证及正式
ASR-only 三轮累计 8000/8000。正式三轮中位数为 114.18 QPS、1141.75 audio_s/s、平均延迟
824.0 ms、P95 1055.5 ms、P99 1606.0 ms。默认时间戳六轮中位数为 11.005 QPS、110.025
audio_s/s，当前确定瓶颈是串行同卡 Aligner，而不是 ASR 数据面。`dev` 已实现 Aligner
有界跨请求动态微批；生产默认为 `ALIGNER_CONCURRENCY=1`、`ALIGNER_BATCH_SIZE=32`。目标
A10 的 batch=4/8/16/32 单轮性能诊断均达到数值晋级线，其中 batch=32 达到 220.03
audio_s/s、P95 4641.8 ms、P99 5585.0 ms，当前是最佳平衡候选。batch=48 虽升至 228.99
audio_s/s，但仅增加 4.1%、P95 恶化 4.9%、显存升至 21821 MiB，已越过工程晋级拐点。
固定 batch=32 后，worker=2 吞吐仅再增 3.7%，执行墙钟却增加 91.3%，同样被拒绝。
attention=auto 已确认实际使用 SDPA，当前固定镜像缺少 `flash_attn`，显式 FA2 阻塞。batch=32
阶段剖析达到 1000/1000、218.95 audio_s/s；每批 WAV 解码、官方 `model.align()`、结果构建均值
分别为 281.12/1097.04/7.53 ms，占三项合计约 20.3%/79.2%/0.5%。`dev` 因此新增持久化
有界并行解码池；生产默认 `ALIGNER_DECODE_WORKERS=1` 保持串行。decode-workers=4 首轮达到
221.89 audio_s/s；解码仅下降 9.52%，模型调用反增 0.81%，最终吞吐仅 +1.34%、P95 -2.31%，
未达到晋级线，已拒绝并恢复 1，不再测试其他线程数。`dev` 随后实现默认关闭的有界预解码；
目标 A10 中预解码命中率 100%、残余等待仅 0.01 ms、批内解码降至 0.03 ms，但同卡
`model.align()` 从 1097.04 增至 1351.02 ms，吞吐仅 +1.84%、P95 -1.58%，仍未晋级并恢复关闭。
官方 `align()` 已使用 `torch.inference_mode()`。后续 vLLM token-classify 路线虽取得 +188.1%
原始吞吐，但生产 oracle 仅 6/11，继续逐层修复不划算，`PERF-ARCH-001` 已拒绝且不再重复 spike。
当前 `PERF-ARCH-002` TensorRT FP16 已完成固定 batch=1、固定 shape 探针：音频改写、文本
mask 改写和 strict 导出图回放的 logits 最大差均为 0.0，timestamp argmax 完全一致；捕获时的
`_export_root` potential side-effect warning 已留证。动态 T1 首轮的 10 秒/7 秒 eager 门禁通过，
但任意 100～3200 帧声明在 strict shape guard 阶段失败；收紧为 2～32 个完整 100 帧 chunk
后，10 秒/7 秒改写 eager、strict 动态导出和同一图双 shape 回放均已通过。T1.1 已实现固定
尾帧余数、动态完整 chunk 数的尾块图；1.602188 秒首轮因 `audio_full_chunks=1` 被专门化而
失败；将下限固定为 2 后，11.334188 秒→追加 1 秒静音的 T1.1 strict 动态导出与异 shape
回放已通过。该结果只证明一个固定尾帧余数 profile。T1.2 随后在目标 A10 通过同 CNN 输出
长度桶规范化：原始主输入为 1133 帧（11 个完整 chunk + tail=33），候选补 7 帧到 1140；原始
回放为 1237 帧（12 个完整 chunk + tail=37），候选补 3 帧到 1240。两者 CNN 尾块输出长度均为
5，canonical tail 均为 40。主/回放的音频改写、文本 mask、strict 动态图回放共六处比较均
logits 逐位一致、最大差 0.0，36 个 timestamp bucket argmax 全部一致，
`summary.dynamic_shape_export=true`。输出桶 13 边界随后也通过：原始 1197/1298 帧、
tail=97/98 分别补 2/1 帧到候选 1199/1299，canonical tail=99、CNN 输出长度 13；
`tail_padding=1`、`tail_output_drop=0`。六处 logits 同样逐位一致、最大差 0.0，timestamp
argmax 完全一致。输出桶 1 最大裁剪边界也已通过：原始 1201/1308 帧、tail=1/8，候选
1208/1308 帧、canonical tail=8、CNN 输出长度 1；`tail_padding=92`、`tail_output_drop=12`。
六处 logits 仍逐位一致、最大差 0.0。其余 10 桶随后按独立 strict export 顺序执行并 10/10
通过；至此 13/13 个 CNN 输出桶均保持六处 logits 逐位一致、最大差 0.0、timestamp argmax 一致，
且都保留 `_export_root` potential side-effect warning。T1.2 已覆盖 `audio_full_chunks≥2` 下的
1～99 非零尾帧余数。短音频 T1.3 的首个完整 chunk 规范化候选已被目标 A10 拒绝：160 帧
full1 主输入最大差 0.0，但 60 帧 full0 回放最大差 1.4296875 且 timestamp argmax 不一致。
官方 `<100` 帧单 chunk 会直接按原始宽度执行卷积，补到 100 后再裁输出无法恢复右边界语义。
现有 full0 原生单 chunk 动态宽度探针不补帧、不裁剪；目标 A10 桶 8 的 57→64 帧先单独通过，
随后桶 2～13 按独立 strict export 顺序执行并 11/11 通过，full0 动态域覆盖 9～99 帧。桶 1
（1～8 帧）因 CNN 时间维恒为 1 命中 PyTorch 0/1 专门化而无法动态导出；`T=2…8` 固定
profile 已 7/7 通过，`T=1` 官方处理器无法构造。full0 可达域至此闭合。full1 同尾块输出桶动态
模式最终 13/13 通过，覆盖 101～199 帧；`retained_output_length` 连续覆盖 14～26，每轮六处
逐位一致并完成 strict 同图回放。总帧 100 的固定 profile 也已在目标 A10 通过：
`chunk_lengths=[100]`、CNN 后长度 13，音频改写、文本 mask 和导出回放三处 logits 均逐位一致、
最大差 0.0。实际可达的 2～3200 帧 `torch.export` 输入域已闭合；下一步只在独立精确固定环境
验证 ONNX/TensorRT，仍不修改生产路径。生产继续保持 PyTorch
ForcedAligner、batch=32 和可选 MPS。历史 A10 纯 90 秒测试中 `3/96` 吞吐最高；仍须在
0～7500 秒真实混合流量下复测。详细数据和方法见 [性能调优](docs/性能调优.md)。

## 文档

- [使用说明](docs/使用说明.md)：环境、下载、启动、全部参数、调用和排障
- [API 文档](docs/API文档.md)：字段、输入输出、错误码、响应头和能力边界
- [技术文档](docs/技术文档.md)：技术路线、处理链路、并发模型与设计取舍
- [性能调优](docs/性能调优.md)：架构瓶颈、策略池、A10 基线、测试矩阵与 v2.0→v3.0 优化路线
- [性能实验台账](docs/性能实验台账.md)：实验状态、复现证据、采纳/拒绝决策与里程碑 tag 索引
- [发布验收](docs/发布验收.md)：静态检查、Linux/CUDA 验收、上线与回滚
- [修改日志](docs/修改日志.md)：v2.1.0/v2.0.0 变更及 v1.0.0 历史记录

## 测试

```bash
python tests/test_chinese_asr_single.py --audio test_data/audio_16000_10s.wav
python tests/verify_qwen_single.py --api-mode chinese-asr \
  --input test_data --ref test_data --baseline-cer 0.05
python tests/test_service.py --api-mode chinese-asr \
  --audio test_data/audio_16000_30s.wav --concurrency 96 --total 2000 \
  --require-server-timing \
  --output-json performance-results/PERF-BASE-001-run-01.json
```

CER 门禁会拒绝缺失或空参考，不会以零计分样本误放行。生产发布还必须在目标 Linux/CUDA
环境完成 [发布验收](docs/发布验收.md)。

docker run -it --gpus '"device=0"' --restart=always -p30960:8080 zhxgharbor.istarshine.com/asr/qwen3-asr-infer:0.19.1
docker run -it --gpus '"device=1"' --restart=always -p30961:8080 zhxgharbor.istarshine.com/asr/qwen3-asr-infer:0.19.1
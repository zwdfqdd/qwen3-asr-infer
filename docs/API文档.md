# API 文档（v2.0.0）

对外基址默认 `http://<host>:8080`。所有时间单位为秒，JSON 使用 UTF-8。

## 请求追踪

进入 aiohttp 请求中间件且能够形成 HTTP 响应的请求都会返回 `X-Request-ID`。客户端可在请求头
提供同名 ID；仅接受 1～64 个字母、数字、点、下划线、冒号或连字符，不合法或缺失时由网关
重新生成。该 ID 会进入结构化日志，客户端不得放入用户身份、文件名、URL、令牌或其他秘密。
HTTP 解析阶段即被拒绝、客户端提前断开、请求被取消或进程终止时可能无法返回响应头；取消和
断连若已进入中间件，会在日志中按观测状态 `499` 记录。

网关访问日志保留字段名、数据大小、音频时长、分片数、语言、结果字符数、错误码和耗时等
统计摘要，同时记录 `/chinese_asr` 的 `input_json` 和 `result_json`。`input_json.base64` 仅保留
原 Base64 ASCII 文本前 64 字节，完整长度仍由 `base64_encoded_chars` 表示；`result_json`
记录成功或业务错误响应。两者仍受日志系统的字符串、集合和嵌套上限保护，超限内容会明确标记
截断。日志包含热词、`article_url` 和识别文本等业务内容，应按敏感数据限制访问和保留周期；
音频二进制、multipart 文件名和后端 Prompt 仍不落日志。

## 1. 业务兼容接口

### `POST /chinese_asr`

请求头：`Content-Type: application/json`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `base64` | string | 是 | 完整音频文件的标准 Base64；推荐 16 kHz 单声道 WAV |
| `article_url` | string/null | 否 | 来源标识，服务不访问该地址，仅成功时原样返回 |
| `hotwords` | string[]/null | 否 | 动态热词；去空、去重后用于 Prompt 软偏置 |
| `language` | string/null | 否 | 缺省或空值时由主模型逐物理分片自动检测；显式值按 30 种语言名称或 ISO 代码强制识别 |

Qwen3-ASR-0.6B 官方支持语言识别与 ASR 的范围为 **30 种语言 + 22 种中国方言/口音**。
可显式指定的 30 种语言如下：

`Chinese (zh)`、`English (en)`、`Cantonese (yue)`、`Arabic (ar)`、`German (de)`、
`French (fr)`、`Spanish (es)`、`Portuguese (pt)`、`Indonesian (id)`、`Italian (it)`、
`Korean (ko)`、`Russian (ru)`、`Thai (th)`、`Vietnamese (vi)`、`Japanese (ja)`、
`Turkish (tr)`、`Hindi (hi)`、`Malay (ms)`、`Dutch (nl)`、`Swedish (sv)`、
`Danish (da)`、`Finnish (fi)`、`Polish (pl)`、`Czech (cs)`、`Filipino (fil)`、
`Persian (fa)`、`Greek (el)`、`Hungarian (hu)`、`Macedonian (mk)`、`Romanian (ro)`。

官方列出的 22 种方言/口音为：Anhui、Dongbei、Fujian、Gansu、Guizhou、Hebei、Henan、
Hubei、Hunan、Jiangxi、Ningxia、Shandong、Shaanxi、Shanxi、Sichuan、Tianjin、Yunnan、
Zhejiang、Cantonese (Hong Kong accent)、Cantonese (Guangdong accent)、Wu language、
Minnan language。方言属于模型自动识别能力声明，但不是当前接口可显式强制的 `language`
枚举；`slid` 忠实返回模型标签，不承诺对全部相近方言做到客观准确区分。

限制：音频默认不超过 2000 秒；解码后的音频文件不超过 280 MiB；整个 JSON 请求体不超过
280 MiB；最多 100 个热词，单词最多 64 字符，总计最多 1000 字符。Base64 会膨胀约 1/3，
因此 280 MiB JSON 请求体最多承载约 210 MiB 原始音频；multipart 还会产生边界和字段开销。
`MAX_JSON_BODY_MB` 保护 HTTP 接收阶段，`MAX_UPLOAD_MB` 保护 Base64 解码后或 multipart
文件读取阶段，二者不能由解析后的时长限制替代。`ENABLE_HOTWORD=false` 时热词仍会校验，
但不进入 Prompt。

```bash
AUDIO_BASE64="$(base64 -w 0 test_data/audio_16000_10s.wav)"
curl http://127.0.0.1:8080/chinese_asr \
  -H 'Content-Type: application/json' \
  -d "{\"base64\":\"$AUDIO_BASE64\",\"article_url\":\"source-001\",\"hotwords\":[\"通义千问\",\"水滴筹\"]}"
```

成功响应：

```json
{
  "code": 0,
  "article_url": "source-001",
  "istar_asr": "识别后的完整文本。",
  "asr": [
    {
      "idx": 0,
      "slid": "Chinese",
      "text": "识别后的完整文本。",
      "speaker": "",
      "timestamp": [0.12, 9.84],
      "words": [
        {"text": "识", "timestamp": [0.12, 0.31]},
        {"text": "别", "timestamp": [0.31, 0.52]}
      ]
    }
  ],
  "message": ""
}
```

### 成功响应字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 成功固定为 `0` |
| `article_url` | string/null | 原样返回请求值 |
| `istar_asr` | string | 各物理分片文本按原音频顺序合并后的完整文本 |
| `asr` | array | 默认在全部非空分片均成功对齐后按主模型标点聚合；存在跳过分片或显式关闭句级时间戳时按物理分片返回 |
| `message` | string | 完整成功时为空；有分片因语种不受支持而跳过对齐时返回明确告警 |

`asr[]`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `idx` | integer | 从 0 开始连续递增 |
| `slid` | string | 每个物理分片的模型自动检测语言或显式请求语言；未知/静音为空；句级跨语种时按出现顺序以逗号合并 |
| `text` | string | 当前分片文本或句级文本 |
| `speaker` | string | 当前未实现说话人识别，固定为空字符串 |
| `timestamp` | `[number, number]` | 默认句级模式是首尾对齐单元边界；存在跳过分片或关闭句级模式时是物理分片帧边界 |
| `words` | array | 默认返回字/词级对齐单元；关闭字/词级开关或当前分片因语种不受支持而跳过对齐时为空数组 |

时间戳开关默认均为 `true`，组合语义如下：

- 两个开关均为 `false`：`asr[]` 按物理分片，`timestamp` 是分片边界，`words=[]`；
  `slid` 仍返回逐片模型检测值或显式请求值。
- 仅 `ENABLE_WORD_TIMESTAMP=true`：`asr[]` 仍按物理分片，`words[]` 返回 Aligner 的中文字符
  或其他受支持语言的词级时间戳。
- `ENABLE_SENTENCE_TIMESTAMP=true`：`asr[]` 按 Qwen3-ASR 自带的 `。！？!?；;` 标点聚合；
  同时启用字/词级开关时，每句附带 `words[]`，否则为空。句子覆盖多种语言时 `slid` 以逗号
  合并；空识别文本返回空 `asr[]`。

ForcedAligner 支持 11 种语种及别名：`Chinese/zh`、`English/en`、`Cantonese/yue`、
`French/fr`、`German/de`、`Italian/it`、`Japanese/ja`、`Korean/ko`、
`Portuguese/pt`、`Russian/ru`、`Spanish/es`。

启用任一时间戳开关后采用逐分片部分对齐：

- `slid` 属于上述 11 种且文本非空：调用 ForcedAligner，`words[]` 在开启字/词级开关时返回
  真实对齐单元；
- `slid` 为空或属于 Arabic、Malay 等其他语言：不调用 ForcedAligner，该分片保留 ASR 文本、
  `slid` 和物理分片 `timestamp`，`words=[]`；
- 至少一个分片被跳过时仍返回 HTTP 200、`code=0`，顶层 `message` 形如
  `部分分片语种不受 ForcedAligner 支持，已跳过真实对齐：分片 0=Malay`；
- 同一长音频中，其他受支持分片仍正常对齐，不因某个分片不受支持而丢弃结果；
- 开启句级时间戳但存在跳过分片时，不执行句级聚合，整个 `asr[]` 保持物理分片结构，
  `message` 追加“句级聚合已禁用，响应保留物理分片边界”。

上述 `words=[]` 明确表示该分片**没有字/词级对齐结果**，不得将其物理分片边界理解为真实
字级、词级或句级时间戳。服务不会回退到 Chinese，也不会根据 ASR 文本伪造时间戳。真实
ForcedAligner 推理失败、返回异常或句级结果无法可靠映射时，整条请求仍返回 1009。

自动检测为 `Malay` 且已启用时间戳时，部分对齐成功响应示例：

```json
{
  "code": 0,
  "article_url": null,
  "istar_asr": "模型识别出的文本。",
  "asr": [
    {
      "idx": 0,
      "slid": "Malay",
      "text": "模型识别出的文本。",
      "speaker": "",
      "timestamp": [0.0, 32.0],
      "words": []
    }
  ],
  "message": "部分分片语种不受 ForcedAligner 支持，已跳过真实对齐：分片 0=Malay"
}
```

该响应是 ASR 成功、对齐跳过，不是 Aligner 成功。若同时启用句级时间戳，`message` 还会追加
`句级聚合已禁用，响应保留物理分片边界`，且 `asr[]` 继续按物理分片返回。

成功响应头 `X-Audio-Chunks` 始终表示网关物理分片数。非句级模式下它与 `asr[]` 数量相同；
句级模式下句子数可能不同。固定分片无 VAD、重叠或跨片去重，不能当作字级对齐。

### 业务错误

```json
{
  "code": 1001,
  "article_url": null,
  "istar_asr": "",
  "asr": [],
  "error": "DECODE_FAILED",
  "message": "音频解码失败，请确认为有效音频格式"
}
```

| code | error | HTTP | 当前语义 |
|---:|---|---:|---|
| 1000 | `INPUT_PARAM_FAILED` | 400 | Content-Type、JSON、必填字段或字段类型错误 |
| 1001 | `DECODE_FAILED` | 400 | Base64 非法、容器不可解码、零帧或采样率无效 |
| 1002 | `VAD_SEGMENT_ERROR` | 500 | 保留码；v2.0.0 未实现 VAD |
| 1003 | `AUDIO_SEGMENT_ERROR` | 500 | 长音频重编码或切分结果异常 |
| 1004 | `ASR_INFER_FAILED` | 500 | 后端推理失败、超时或响应结构异常 |
| 1005 | `AUDIO_TOO_LONG` | 400 | 音频时长、音频文件或 JSON 请求体超过限制 |
| 1006 | `MODEL_LOAD_FAILED` | 500 | 无法连接 vLLM 后端 |
| 1007 | `SERVICE_BUSY` | 503 | vLLM 返回 429 或 503 |
| 1008 | `HOTWORD_VERSION_CONFLICT` | 409 | 保留码；v2.0.0 无版本化词表 |
| 1009 | `ALIGNER_INFER_FAILED` | 500 | Aligner 实际推理失败、返回结构异常或句级结果无法可靠映射；可预判的不支持语种使用 HTTP 200 部分对齐契约 |

错误时 `article_url` 固定为 `null`，不会回显请求中的来源标识。零字节、零帧和无效音频均为
失败，不返回 HTTP 200 空结果。

## 2. multipart 转写接口

### `POST /v1/audio/transcriptions`

该接口只保证本文件列出的有限字段和文本响应，不承诺完整 OpenAI API 兼容。

| multipart 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `file` | file | 是 | 非空音频文件 |
| `model` | string | 否 | 缺省使用 `SERVED_MODEL_NAME`；显式值会透传 vLLM |
| `response_format` | string | 否 | 网关强制改为 `json` |
| `language` | string | 否 | 原样透传后端 |
| `temperature` | string | 否 | 原样透传后端 |
| 其他文本字段 | string | 否 | 网关原样透传后端，是否支持由 vLLM 决定 |

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@test_data/audio_16000_10s.wav \
  -F model=qwen3-asr \
  -F response_format=json
```

成功响应为 `{"text":"识别文本"}`，响应头 `X-Audio-Chunks` 为正整数。任一分片失败时整条
请求失败，不返回部分文本。该接口不执行 ForcedAligner，也不返回字/词级或句级时间戳。

原生错误结构：

```json
{
  "error": {
    "message": "ASR 推理失败",
    "type": "gateway_error",
    "code": "ASR_INFER_FAILED"
  }
}
```

| HTTP | 场景 |
|---:|---|
| 400 | 缺少非空 `file`、音频解码失败或时长超过上限 |
| 413 | multipart 请求体超过网关限制 |
| 415 | 请求不是 multipart/form-data |
| 502 | vLLM 不可连接、返回非成功状态或响应结构异常 |
| 503 | vLLM 返回 429/503，服务繁忙 |
| 504 | 单个后端分片超过 `BACKEND_TIMEOUT` |

## 3. 管理端点

- `GET /health`：同时检查网关与后端，成功返回
  `{"status":"ok","version":"2.0.0","model":"qwen3-asr","timestamps":true,"aligner_device":"cuda:0"}`；
  时间戳关闭时 `timestamps=false`、`aligner_device=null`，后端不可用返回 503。
- `GET /metrics`：透明代理 vLLM Prometheus 文本。
- `GET /v1/models`：透明代理 vLLM 模型列表。

## 4. 能力边界

当前工作区不提供 VAD、说话人识别、默认热词表或流式 HTTP 网关。Qwen3-ASR 主模型直接
输出标点，不接入 CT-Transformer。默认 ForcedAligner 仅服务 `/chinese_asr`，必须安装
`requirements-aligner.txt` 并使用 ModelScope 固定 revision 的本地权重；它不提供说话人
识别。官方流式能力不等于本项目已实现流式会话。动态热词只改变 Prompt，属于概率性软偏置；
`article_url` 永远不会被服务下载。
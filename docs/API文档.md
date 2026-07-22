# API 文档（v1.0.0）

对外基址默认 `http://<host>:8080`。所有时间单位为秒，JSON 使用 UTF-8。

## 1. 业务兼容接口

### `POST /chinese_asr`

请求头：`Content-Type: application/json`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `base64` | string | 是 | 完整音频文件的标准 Base64；推荐 16 kHz 单声道 WAV |
| `article_url` | string/null | 否 | 来源标识，服务不访问该地址，仅成功时原样返回 |
| `hotwords` | string[]/null | 否 | 动态热词；去空、去重后用于 Prompt 软偏置 |

限制：音频默认不超过 300 秒和 64 MiB；JSON 请求体不超过 96 MiB；最多 100 个热词，
单词最多 64 字符，总计最多 1000 字符。`ENABLE_HOTWORD=false` 时热词仍会校验但不进入 Prompt。

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
      "slid": "",
      "text": "识别后的完整文本。",
      "speaker": "",
      "timestamp": [0.0, 10.0],
      "words": []
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
| `istar_asr` | string | 各分片文本按原音频顺序合并后的完整文本 |
| `asr` | array | 每项对应一个网关分片；短音频通常只有一项 |
| `message` | string | 成功固定为空字符串 |

`asr[]`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `idx` | integer | 从 0 开始连续递增 |
| `slid` | string | 当前未输出语种，固定为空字符串 |
| `text` | string | 当前分片识别文本 |
| `speaker` | string | 当前未实现说话人识别，固定为空字符串 |
| `timestamp` | `[number, number]` | 分片在原音频中的开始/结束秒数，由整数采样帧边界计算 |
| `words` | array | 当前未实现字级时间戳，固定为空数组 |

成功响应头 `X-Audio-Chunks` 为正整数，并与 `asr[]` 数量相同。固定分片边界不是语义句子
边界或字级对齐；硬切不做重叠和跨片去重。

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
| 1002 | `VAD_SEGMENT_ERROR` | 500 | 保留码；v1.0.0 未实现 VAD |
| 1003 | `AUDIO_SEGMENT_ERROR` | 500 | 长音频重编码或切分结果异常 |
| 1004 | `ASR_INFER_FAILED` | 500 | 后端推理失败、超时或响应结构异常 |
| 1005 | `AUDIO_TOO_LONG` | 400 | 音频时长、音频文件或 JSON 请求体超过限制 |
| 1006 | `MODEL_LOAD_FAILED` | 500 | 无法连接 vLLM 后端 |
| 1007 | `SERVICE_BUSY` | 503 | vLLM 返回 429 或 503 |
| 1008 | `HOTWORD_VERSION_CONFLICT` | 409 | 保留码；v1.0.0 无版本化词表 |

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
请求失败，不返回部分文本。

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
  `{"status":"ok","version":"1.0.0","model":"qwen3-asr"}`；后端不可用返回 503。
- `GET /metrics`：透明代理 vLLM Prometheus 文本。
- `GET /v1/models`：透明代理 vLLM 模型列表。

## 4. 能力边界

v1.0.0 不提供 VAD、字级时间戳、句子级时间戳、说话人识别、默认热词表或流式网关。动态
热词只改变 Prompt，属于概率性软偏置。`article_url` 永远不会被服务下载。
# 测试说明

## 单条业务接口

```bash
python tests/test_chinese_asr_single.py \
  --audio test_data/audio_16000_10s.wav \
  --article-url source-001 \
  --hotword 通义千问 --hotword 水滴筹
```

校验 HTTP/业务码、字段类型、分片序号、时间戳单调性、`words=[]`，以及
`X-Audio-Chunks` 与 `asr[]` 数量一致。

## 并发压测

```bash
python tests/test_service.py --api-mode chinese-asr \
  --audio test_data/audio_16000_30s.wav --concurrency 96 --total 2000
```

报告原始请求 QPS、后端分片 RPS、audio_s/s、P50/P90/P95/P99、分片一致性和 GPU/CPU。
`--chunk-seconds` 只用于计算预期分片，必须与服务端配置一致。

## CER 门禁

```bash
python tests/verify_qwen_single.py --api-mode chinese-asr \
  --input test_data --ref test_data --baseline-cer 0.05
```

设置阈值时，参考必须覆盖全部输入且归一化后非空，否则非零退出。CER 当前执行 NFKC、去除
空白后计算字符级 Levenshtein；标点和大小写仍计入错误。单音频直接文本使用 `--ref-text`。
# 产品概述

Qwen3-ASR-0.6B 单 GPU 高吞吐 HTTP 推理服务。核心为原生 vLLM 数据面，配套最小 aiohttp
CPU 网关；统一对外端口 8080。

已实现：
- `/chinese_asr` Base64 JSON 业务接口；
- 有限 multipart `/v1/audio/transcriptions`；
- 0～300 秒音频，超过 30 秒固定分片、受控并发和顺序合并；
- 动态热词 Prompt 软偏置；
- 分片级时间边界、健康检查、vLLM 指标和模型列表代理；
- ModelScope 固定 revision 下载与完整性校验。

未实现：VAD、CT 标点、ForcedAligner/字级时间戳、说话人识别、流式网关。`words` 固定
为空，不得把分片边界描述为字级时间戳。项目不是完整公网接入层，生产仍需在上层补鉴权、
TLS、租户限流、熔断、灰度和多实例负载均衡。
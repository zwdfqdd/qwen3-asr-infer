# 项目结构

```text
qwen3asr_infer/
├── src/
│   ├── config.py       网关系统配置、校验与 settings 单例
│   └── gateway.py      aiohttp 协议网关、音频切分和后端代理
├── scripts/
│   ├── download_model.py  ModelScope 下载、SHA256 与 manifest
│   └── run_verify.sh      发布 CER 验证入口
├── tests/              单条契约测试、并发压测、CER 工具
├── docs/               API、使用、技术、性能、发布验收与日志
├── test_data/          音频及同名参考文本
├── models/qwen3-asr-0.6b/vllm/  本地权重，不入库
├── run.sh              统一启动 vLLM 与网关
└── run_gateway.sh      仅独立调试网关
```

处理链路：客户端访问 `:8080` → aiohttp 检查/切分 → 本机 vLLM `:8081` → 分片顺序合并。
不再存在 `main.py`、`schemas.py`、`qwen_engine.py` 或三级自建模型流水线。涉及配置必须修改
`src/config.py` 和对应文档；涉及行为变化同步 README、相关 docs 与修改日志。
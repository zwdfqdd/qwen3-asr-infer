"""固定依赖版本的进程级兼容修复，不修改 ModelScope 模型文件。"""

from functools import wraps
from threading import RLock
import warnings

try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
except ImportError:
    pass
else:
    # 历史上 vLLM 0.16 的 tokenizer 路径未透传该参数；迁移 0.19.1 后先保留兼容行为，
    # 待固定 Transformers 5.x 并完成 CER/并发门禁后再评估移除。
    _original_from_pretrained = PreTrainedTokenizerBase.from_pretrained.__func__

    @classmethod
    def _from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """保留原加载参数，并在调用方未指定时开启 tokenizer regex 修复。"""
        kwargs.setdefault("fix_mistral_regex", True)
        return _original_from_pretrained(
            cls, pretrained_model_name_or_path, *inputs, **kwargs
        )

    PreTrainedTokenizerBase.from_pretrained = _from_pretrained

    # 历史 vLLM 0.16 的 Chat 与多模态预处理线程会并发调用同一个 Fast Tokenizer。
    # 迁移 0.19.1 后保留已验证的防护，必须重跑并发稳定性后才能按证据移除。
    # Transformers 会在每次调用时修改 Rust tokenizer 的截断/填充状态；未串行化时会
    # 偶发触发 ``RuntimeError: Already borrowed`` 并让 vLLM 返回 HTTP 400。
    # 锁覆盖完整公共调用而不是只保护 enable_truncation，确保借用生命周期不交叠。
    _fast_tokenizer_call_lock = RLock()
    _original_fast_tokenizer_call = PreTrainedTokenizerFast.__call__

    @wraps(_original_fast_tokenizer_call)
    def _thread_safe_fast_tokenizer_call(self, *args, **kwargs):
        with _fast_tokenizer_call_lock:
            return _original_fast_tokenizer_call(self, *args, **kwargs)

    PreTrainedTokenizerFast.__call__ = _thread_safe_fast_tokenizer_call

    # Transformers 4.57.6 通用 AutoProcessor 的弃用提示与纯音频处理无关。
    warnings.filterwarnings(
        "ignore",
        message=r"The image_processor_class argument is deprecated.*",
        category=FutureWarning,
        module=r"transformers\..*",
    )

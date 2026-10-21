"""固定依赖版本的进程级兼容修复，不修改 ModelScope 模型文件。"""

import warnings

try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
except ImportError:
    pass
else:
    # vLLM 0.16 的独立 tokenizer 路径未透传该参数；默认执行真实 regex 修复。
    _original_from_pretrained = PreTrainedTokenizerBase.from_pretrained.__func__

    @classmethod
    def _from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """保留原加载参数，并在调用方未指定时开启 tokenizer regex 修复。"""
        kwargs.setdefault("fix_mistral_regex", True)
        return _original_from_pretrained(
            cls, pretrained_model_name_or_path, *inputs, **kwargs
        )

    PreTrainedTokenizerBase.from_pretrained = _from_pretrained

    # Transformers 4.57.6 通用 AutoProcessor 的弃用提示与纯音频处理无关。
    warnings.filterwarnings(
        "ignore",
        message=r"The image_processor_class argument is deprecated.*",
        category=FutureWarning,
        module=r"transformers\..*",
    )

import os
from typing import Any

from openai import OpenAI


class DeepSeekCompatibleClient(OpenAI):
    """兼容 OpenAI SDK 调用方式的 DeepSeek 客户端。"""

    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("未配置 DEEPSEEK_API_KEY 环境变量。")

        super().__init__(api_key=api_key, base_url=base_url, **kwargs)

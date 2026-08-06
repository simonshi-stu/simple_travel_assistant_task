"""读取项目根目录的本地 .env 文件，不覆盖已有环境变量。"""

import os
from pathlib import Path


def load_local_env() -> None:
    """加载简单的 KEY=VALUE 配置，避免每次启动都手动设置 PowerShell 变量。"""

    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            # 如果用户已经在 PowerShell 中设置了变量，优先使用已有值。
            os.environ.setdefault(key, value)

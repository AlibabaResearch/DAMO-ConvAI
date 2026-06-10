# set RAY_DEDUP_LOGS=0 before importing ray
import os
import logging

os.environ["RAY_DEDUP_LOGS"] = os.getenv("RAY_DEDUP_LOGS", "1")

# 全局日志级别配置 - 在项目初始化时就设置
os.environ.setdefault('LOG_LEVEL', 'WARNING')
os.environ.setdefault('PYTHONLOGLEVEL', 'WARNING')

# 全面抑制第三方库的DEBUG日志
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpcore.http11').setLevel(logging.WARNING)
logging.getLogger('httpcore.connection').setLevel(logging.WARNING)
logging.getLogger('httpcore.dispatch').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('mcp.client.sse').setLevel(logging.WARNING)
logging.getLogger('mcp').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('ray').setLevel(logging.WARNING)
logging.getLogger('ray.util').setLevel(logging.WARNING)

# Enable deterministic mode if DETERMINISTIC_MODE environment variable is set
if os.getenv("DETERMINISTIC_MODE", "0") == "1":
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    print("Deterministic mode enabled")


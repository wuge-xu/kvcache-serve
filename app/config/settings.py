import os

MODEL_NAME = os.getenv("KVCACHE_MODEL", "sshleifer/tiny-gpt2")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "64"))

import os

# torch + lightgbm in the same process can segfault on macOS due to duplicate
# OpenMP runtimes; this env var avoids it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from mymodule.utils.env import ensure_env_loaded  # noqa: E402
from mymodule.utils.log import ensure_logging_configured  # noqa: E402

ensure_env_loaded()
ensure_logging_configured()

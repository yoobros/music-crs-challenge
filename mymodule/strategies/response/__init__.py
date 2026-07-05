"""Response generator registry.

Two orthogonal axes:
  - Logic: chosen via `--response-gen {noop|pas|auto}` (env `MYMODULE_RESPONSE_GEN`)
  - LM provider: chosen via `--response-provider {ollama|openai}`
    (env `MYMODULE_CHAT_PROVIDER`). Ignored for `noop`.

To add a new variant (e.g. `another/generator.py`):
  1. Create `mymodule/strategies/response/<variant>/generator.py` with a
     class subclassing `DspyResponseGenerator`.
  2. Import and register here:
        from mymodule.strategies.response.another.generator import AnotherResponseGenerator
        RESPONSE_REGISTRY["another"] = AnotherResponseGenerator

Neither the logic name nor the provider is encoded in TID — they're a
format-only concern that doesn't require re-running retrieval.
"""

from mymodule.strategies.response.auto.generator import AutoResponseGenerator
from mymodule.strategies.response.base import BaseResponseGenerator
from mymodule.strategies.response.noop import NoopResponseGenerator
from mymodule.strategies.response.pas.generator import PasResponseGenerator

RESPONSE_REGISTRY: dict[str, type[BaseResponseGenerator]] = {
    "noop": NoopResponseGenerator,
    "pas": PasResponseGenerator,
    "auto": AutoResponseGenerator,
}


def get_response_generator(name: str, **kwargs) -> BaseResponseGenerator:
    """Instantiate a response generator by logic name.

    Kwargs are forwarded to the class constructor — most notably `provider`
    for DSPy-backed variants.
    """
    if name not in RESPONSE_REGISTRY:
        available = ", ".join(sorted(RESPONSE_REGISTRY.keys()))
        raise KeyError(f"No response generator '{name}'. Available: [{available}]")
    return RESPONSE_REGISTRY[name](**kwargs)

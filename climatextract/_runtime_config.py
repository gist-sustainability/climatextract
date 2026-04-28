"""Process-wide config state read by adapters at handler construction time.

``main.py`` populates this after parsing ``climatextract.toml`` so that
adapters can resolve TOML values (model name, concurrency caps) without
taking a hard import dependency on ``main.py``. Both sides depend only
on this small module.
"""

from typing import Optional

from climatextract.params import ExperimentParams

_current: Optional[ExperimentParams] = None


def set_current(params: ExperimentParams) -> None:
    global _current
    _current = params


def get_current() -> Optional[ExperimentParams]:
    """Return the current ExperimentParams, lazy-loading the default TOML
    on first access if nothing has been set yet.

    Lazy load handles the case where a user constructs a handler
    *before* ``extract()`` runs (e.g., in a script that builds the
    handler first and passes it via ``llm=``). Without this, the handler
    would see an empty config and silently fall back to its class
    ``MODEL`` instead of honoring TOML.
    """
    global _current
    if _current is None:
        try:
            # Lazy import — _runtime_config can't depend on main.py at module load.
            from climatextract.main import _load_config
            _, params, *_ = _load_config()
            _current = params
        except Exception:
            pass  # No TOML / unreadable — handlers fall back to class defaults.
    return _current


def clear() -> None:
    global _current
    _current = None

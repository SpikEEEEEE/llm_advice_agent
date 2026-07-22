from __future__ import annotations

import math
from decimal import Decimal
from numbers import Integral, Real
from typing import Any


def sanitize_json_value(value: Any) -> Any:
    """Recursively convert values to strict-JSON-safe primitives.

    LLM/provider payloads can contain NaN or Infinity. Python's JSON decoder and
    encoder accept those extensions by default, while Starlette intentionally
    rejects them in HTTP responses. Converting them to null keeps the raw shape
    auditable without allowing a malformed model value to break the API.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value) if value.is_finite() else None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {
            str(key): sanitize_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json_value(item) for item in value]
    return value

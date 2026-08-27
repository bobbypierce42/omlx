# SPDX-License-Identifier: Apache-2.0
"""Shared redaction for short messages sent through public transports."""

from __future__ import annotations

import re
from typing import Any

_BEARER_SECRET_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_NAMED_SECRET_RE = re.compile(
    r"(?i)\b(api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|authorization|password|secret|credential|token|key)"
    r"\s*[:=]\s*(?:bearer\s+)?[^\s,;}\]]+"
)
_JSON_SECRET_RE = re.compile(
    r'(?i)(?P<prefix>"(?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|authorization|password|secret|credential|token|key)"\s*:\s*)'
    r'"(?:\\.|[^"\\])*"'
)
_TOKEN_SECRET_RE = re.compile(r"(?i)\b(?:sk|rk|pk)-[A-Za-z0-9_-]{8,}\b")


def safe_message(value: Any, fallback: str = "realtime request failed") -> str:
    """Return a bounded message with common credential forms redacted."""
    message = str(value).replace("\r", " ").replace("\n", " ").strip()
    message = _BEARER_SECRET_RE.sub("Bearer <redacted>", message)
    message = _NAMED_SECRET_RE.sub(
        lambda match: f"{match.group(1)}=<redacted>", message
    )
    message = _JSON_SECRET_RE.sub(r'\g<prefix>"<redacted>"', message)
    message = _TOKEN_SECRET_RE.sub("<redacted>", message)
    return (message[:500] or fallback)[:500]

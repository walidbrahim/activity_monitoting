"""
apps.common.gui_config
=======================
GUI-only configuration: theme colours and display constants.

This module is intentionally separate from the engine config layer
so that theme changes never touch radar_engine/ code.

Usage::

    from apps.common.gui_config import GuiThemeConfig, load_gui_theme

    theme = load_gui_theme()          # from AppConfig or base.yaml
    bg    = theme.fig_bg              # "#0F172A"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass   # keep imports minimal


@dataclass(frozen=True)
class GuiThemeConfig:
    """Colour palette and display constants for the PyQt6 / PyQtGraph GUI.

    All colour values are CSS hex strings (e.g. ``"#0F172A"``).
    """
    # Backgrounds
    fig_bg:    str = "#0F172A"
    panel_bg:  str = "#1E1B4B"
    card_bg:   str = "#1E293B"

    # Card state colours
    card_ok:    str = "#059669"
    card_warn:  str = "#D97706"
    card_alert: str = "#DC2626"

    # Text
    text:    str = "#E5E7EB"
    subtext: str = "#9CA3AF"
    grid:    str = "#334155"

    # Zone colours
    bed:      str = "#6366F1"
    chair:    str = "#14B8A6"
    monitor:  str = "#6B8E7A"
    ignore:   str = "#DC2626"

    # Overlay elements
    occupant:  str = "#22D3EE"
    room_edge: str = "#CBD5E1"
    fov:       str = "#EF4444"
    radar:     str = "#F43F5E"
    origin:    str = "#FACC15"

    @classmethod
    def from_dict(cls, d: dict) -> "GuiThemeConfig":
        """Build from a plain dict (e.g. from app_cfg.gui_theme.model_dump())."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)


def load_gui_theme(app_cfg=None) -> GuiThemeConfig:
    """Load GUI theme from AppConfig or fall back to defaults.

    Args:
        app_cfg: Optional loaded AppConfig instance. If None, defaults are used.

    Returns:
        A frozen GuiThemeConfig.
    """
    if app_cfg is None:
        return GuiThemeConfig()
    try:
        return GuiThemeConfig.from_dict(app_cfg.gui_theme.model_dump())
    except Exception:
        return GuiThemeConfig()


# ── Notification helper (secrets from env vars) ────────────────────────────────

class PushoverCredentials:
    """Reads Pushover credentials from environment variables only.

    Never stored in YAML, never in git.

    Environment variables expected:
        PUSHOVER_USER_KEY   — your Pushover user key
        PUSHOVER_API_TOKEN  — your Pushover application API token

    Usage::

        creds = PushoverCredentials.load()
        if creds:
            send_pushover(creds.user_key, creds.api_token, message)
    """
    __slots__ = ("user_key", "api_token")

    def __init__(self, user_key: str, api_token: str):
        self.user_key  = user_key
        self.api_token = api_token

    @classmethod
    def load(cls) -> "PushoverCredentials | None":
        """Return credentials if both env vars are set, else None."""
        import os
        user_key  = os.getenv("PUSHOVER_USER_KEY")
        api_token = os.getenv("PUSHOVER_API_TOKEN")
        if user_key and api_token:
            return cls(user_key, api_token)
        return None

    def __bool__(self) -> bool:
        return bool(self.user_key and self.api_token)

"""
ui_helpers.py — Root forwarding proxy to src/ui_helpers.py
Ensures seamless imports whether sys.path includes src or root.
"""

import os
import sys
import importlib.util

_SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "ui_helpers.py")

_spec = importlib.util.spec_from_file_location("_src_ui_helpers_mod", _SRC_PATH)
_src_ui = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_src_ui)

LIGHT_THEME_CSS = getattr(_src_ui, "LIGHT_THEME_CSS", "")
apply_custom_theme = getattr(_src_ui, "apply_custom_theme", lambda: None)
get_light_plotly_layout = getattr(_src_ui, "get_light_plotly_layout", lambda: {})
get_dark_plotly_layout = getattr(_src_ui, "get_dark_plotly_layout", lambda: {})
render_metric_panel = getattr(_src_ui, "render_metric_panel", lambda *a, **k: None)
render_disclaimer = getattr(_src_ui, "render_disclaimer", lambda *a, **k: None)
render_architecture_diagram = getattr(_src_ui, "render_architecture_diagram", lambda: None)

__all__ = [
    "LIGHT_THEME_CSS",
    "apply_custom_theme",
    "get_light_plotly_layout",
    "get_dark_plotly_layout",
    "render_metric_panel",
    "render_disclaimer",
    "render_architecture_diagram"
]


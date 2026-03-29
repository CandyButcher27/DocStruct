"""Utilities for loading and accessing the DocStruct pipeline YAML configuration."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "defaults.yaml"

# ---------------------------------------------------------------------------
# Internal loader (cached after first call)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return the parsed dict. Cached per path."""
    try:
        import yaml  # PyYAML
    except ImportError as exc:
        raise ImportError("PyYAML is required for config loading. Run: pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Return the merged pipeline configuration dictionary.

    Args:
        path: Optional path to a custom YAML file. If *None*, the bundled
              ``config/defaults.yaml`` is used.

    Returns:
        A plain dict mirroring the YAML structure.
    """
    resolved = path if path is not None else str(_DEFAULT_CONFIG_PATH)
    return _load_yaml(resolved)


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_classification_thresholds(cfg: Dict[str, Any]) -> Dict[str, float]:
    """Return the per-class confidence gate dict, e.g. ``{"text": 0.35, ...}``."""
    defaults = {
        "text": 0.35,
        "header": 0.50,
        "table": 0.55,
        "figure": 0.45,
        "caption": 0.45,
    }
    raw = cfg.get("classification_thresholds", {}) or {}
    defaults.update({k: float(v) for k, v in raw.items()})
    return defaults


def get_ensemble_weights(cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return ``(model_weight, rule_weight, geo_weight)`` for hybrid mode."""
    ens = cfg.get("ensemble", {}) or {}
    return (
        float(ens.get("model_weight", 0.60)),
        float(ens.get("rule_weight", 0.25)),
        float(ens.get("geo_weight", 0.15)),
    )


def get_detector_thresholds(cfg: Dict[str, Any]):
    """Return a populated ``ClasswiseThresholdConfig`` from the config dict."""
    # Import here to avoid circular imports
    from models.detector import ClasswiseThresholdConfig

    dt = cfg.get("detector_thresholds", {}) or {}
    return ClasswiseThresholdConfig(
        table=float(dt.get("table", 0.70)),
        figure=float(dt.get("figure", 0.70)),
        text=float(dt.get("text", 0.50)),
        header=float(dt.get("header", 0.60)),
        caption=float(dt.get("caption", 0.60)),
    )


def get_doclaynet_confidence(cfg: Dict[str, Any]) -> float:
    """Return the doclaynet model confidence threshold."""
    dt = cfg.get("detector_thresholds", {}) or {}
    return float(dt.get("doclaynet_confidence", 0.50))


def get_table_candidate_params(cfg: Dict[str, Any]) -> Dict[str, float]:
    """Return table-candidate tuning params as a flat dict."""
    tc = cfg.get("table_candidates", {}) or {}
    return {
        "geometry_threshold": float(tc.get("geometry_threshold", 0.45)),
        "overlap_threshold": float(tc.get("overlap_threshold", 0.35)),
        "fusion_acceptance": float(tc.get("fusion_acceptance", 0.45)),
        "table_match_threshold": float(tc.get("table_match_threshold", 0.40)),
    }

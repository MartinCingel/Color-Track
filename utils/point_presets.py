"""Persistent point-tracker preset storage.

The point tracker has two practical modes, normal markers and tiny features.
This module stores the last-used parameters for each mode in a project-local
JSON file so repeated videos with similar character do not require manual
retuning after every application restart.  The defaults below are also the
first-run settings shipped with Color Track.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


PRESET_VERSION = 8

POINT_PRESET_FIELDS = (
    "sample_size",
    "expected_diameter",
    "normal_init_threshold",
    "normal_init_min_probability_margin",
    "normal_init_max_area_ratio",
    "normal_init_growth_radius",
    "normal_init_close_iterations",
    "tiny_init_peak_threshold",
    "kalman_enabled",
    "kalman_model",
    "measurement_noise",
    "process_noise",
    "kalman_gate_sigma",
    "min_search_radius",
    "max_search_radius",
    "normal_search_margin",
    "miss_expansion",
    "max_search_margin",
    "max_prediction_error",
    "miss_motion_allowance",
    "normal_recenter_on_edge",
    "normal_recenter_edge_px",
    "normal_recenter_extra_margin",
    "normal_recenter_min_area_ratio",
    "measurement_center_mode",
    "adaptive_prefilter_enabled",
    "lab_l_weight",
    "lab_a_weight",
    "lab_b_weight",
    "luminance_polarity_enabled",
    "luminance_polarity_min_delta",
    "luminance_polarity_tolerance",
    "colour_diagnostics_enabled",
    "shape_validation",
    "tiny_peak_threshold",
    "tiny_contrast_threshold",
    "tiny_init_contrast_threshold",
    "tiny_direction_min_cosine",
    "tiny_update_confidence_threshold",
    "tiny_mosse_mode",
    "normal_mosse_mode",
    "tiny_mosse_window_scale",
    "tiny_mosse_learning_rate",
    "tiny_mosse_psr_threshold",
    "tiny_mosse_peak_margin_threshold",
    "debug_uncertain_center_policy",
)

DEFAULT_POINT_PRESETS: Dict[str, Dict[str, Any]] = {
    "normal": {
        "sample_size": 5,
        "expected_diameter": 20.0,
        "normal_init_threshold": 0.72,
        "normal_init_min_probability_margin": 0.08,
        "normal_init_max_area_ratio": 6.0,
        "normal_init_growth_radius": 0.0,
        "normal_init_close_iterations": 1,
        "tiny_init_peak_threshold": 0.40,
        "kalman_enabled": False,
        "kalman_model": "constant_velocity",
        "measurement_noise": 0.7,
        "process_noise": 0.5,
        "kalman_gate_sigma": 4.0,
        "min_search_radius": 12,
        "max_search_radius": 80,
        "normal_search_margin": 15,
        "miss_expansion": 16,
        "max_search_margin": 150,
        "max_prediction_error": 22.0,
        "miss_motion_allowance": 14.0,
        "normal_recenter_on_edge": True,
        "normal_recenter_edge_px": 3,
        "normal_recenter_extra_margin": 20,
        "normal_recenter_min_area_ratio": 0.25,
        "measurement_center_mode": "full_blob",
        "adaptive_prefilter_enabled": False,
        "lab_l_weight": 1.0,
        "lab_a_weight": 1.0,
        "lab_b_weight": 1.0,
        "luminance_polarity_enabled": True,
        "luminance_polarity_min_delta": 8.0,
        "luminance_polarity_tolerance": 35.0,
        "colour_diagnostics_enabled": False,
        "shape_validation": "soft",
        "tiny_peak_threshold": 0.55,
        "tiny_contrast_threshold": 0.04,
        "tiny_init_contrast_threshold": 8.0,
        "tiny_direction_min_cosine": 0.20,
        "tiny_update_confidence_threshold": 0.65,
        "tiny_mosse_mode": "off",
        "normal_mosse_mode": "off",
        "tiny_mosse_window_scale": 4.0,
        "tiny_mosse_learning_rate": 0.05,
        "tiny_mosse_psr_threshold": 6.0,
        "tiny_mosse_peak_margin_threshold": 0.15,
        "debug_uncertain_center_policy": "none",
    },
    "tiny": {
        "sample_size": 3,
        "expected_diameter": 3.0,
        "normal_init_threshold": 0.72,
        "normal_init_min_probability_margin": 0.08,
        "normal_init_max_area_ratio": 6.0,
        "normal_init_growth_radius": 0.0,
        "normal_init_close_iterations": 1,
        "tiny_init_peak_threshold": 0.50,
        "kalman_enabled": True,
        "kalman_model": "constant_acceleration",
        "measurement_noise": 1.0,
        "process_noise": 2.0,
        "kalman_gate_sigma": 4.0,
        "min_search_radius": 12,
        "max_search_radius": 80,
        "normal_search_margin": 15,
        "miss_expansion": 16,
        "max_search_margin": 150,
        "max_prediction_error": 22.0,
        "miss_motion_allowance": 14.0,
        "normal_recenter_on_edge": False,
        "normal_recenter_edge_px": 3,
        "normal_recenter_extra_margin": 20,
        "normal_recenter_min_area_ratio": 0.25,
        "measurement_center_mode": "full_blob",
        "adaptive_prefilter_enabled": False,
        "lab_l_weight": 2.0,
        "lab_a_weight": 1.0,
        "lab_b_weight": 1.0,
        "luminance_polarity_enabled": True,
        "luminance_polarity_min_delta": 8.0,
        "luminance_polarity_tolerance": 35.0,
        "colour_diagnostics_enabled": False,
        "shape_validation": "off",
        "tiny_peak_threshold": 0.43,
        "tiny_contrast_threshold": 0.01,
        "tiny_init_contrast_threshold": 3.0,
        "tiny_direction_min_cosine": 0.20,
        "tiny_update_confidence_threshold": 0.65,
        "tiny_mosse_mode": "primary",
        "normal_mosse_mode": "off",
        "tiny_mosse_window_scale": 4.0,
        "tiny_mosse_learning_rate": 0.05,
        "tiny_mosse_psr_threshold": 6.0,
        "tiny_mosse_peak_margin_threshold": 0.15,
        "debug_uncertain_center_policy": "none",
    },
}


def preset_path() -> Path:
    from utils.app_paths import app_data_path
    return app_data_path("settings", "point_tracker_presets.json")


def _clean_mode(mode: str) -> str:
    return "tiny" if str(mode) == "tiny" else "normal"


def default_point_preset(mode: str) -> Dict[str, Any]:
    return dict(DEFAULT_POINT_PRESETS[_clean_mode(mode)])


def load_point_presets() -> Dict[str, Dict[str, Any]]:
    presets = {
        "normal": default_point_preset("normal"),
        "tiny": default_point_preset("tiny"),
    }
    path = preset_path()
    if not path.exists():
        return presets
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        version = int(data.get("version", 0)) if isinstance(data, dict) else 0
        stored = data.get("presets", data)
        for mode in ("normal", "tiny"):
            if isinstance(stored.get(mode), dict):
                for key in POINT_PRESET_FIELDS:
                    if key not in stored[mode]:
                        continue
                    # Version 2 restores the original reliable normal-marker
                    # default: Kalman is OFF unless the user enables it again.
                    # Older autosaved presets could silently carry a Kalman-on
                    # normal mode into supposedly default tests.
                    if version < 2 and mode == "normal" and key == "kalman_enabled":
                        continue
                    presets[mode][key] = stored[mode][key]
    except Exception:
        return presets
    return presets


def save_point_presets(presets: Dict[str, Dict[str, Any]]) -> None:
    path = preset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned: Dict[str, Dict[str, Any]] = {}
    for mode in ("normal", "tiny"):
        merged = default_point_preset(mode)
        merged.update({k: presets.get(mode, {}).get(k, merged[k]) for k in POINT_PRESET_FIELDS})
        cleaned[mode] = {k: merged[k] for k in POINT_PRESET_FIELDS}
    payload = {"version": PRESET_VERSION, "presets": cleaned}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_point_preset(mode: str, values: Dict[str, Any]) -> None:
    mode = _clean_mode(mode)
    presets = load_point_presets()
    preset = default_point_preset(mode)
    preset.update({k: values[k] for k in POINT_PRESET_FIELDS if k in values})
    presets[mode] = preset
    save_point_presets(presets)


def restore_default_point_preset(mode: str) -> Dict[str, Any]:
    mode = _clean_mode(mode)
    presets = load_point_presets()
    presets[mode] = default_point_preset(mode)
    save_point_presets(presets)
    return dict(presets[mode])


def learned_preset_path() -> Path:
    return preset_path().with_name("learned_tracker_presets.json")


def export_calibration_path() -> Path:
    return preset_path().with_name("export_calibration.json")


def load_export_calibration() -> Dict[str, Any]:
    path = export_calibration_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_export_calibration(values: Dict[str, Any]) -> None:
    path = export_calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(values), indent=2), encoding="utf-8")


def named_settings_preset_path() -> Path:
    return preset_path().with_name("named_tracker_settings.json")


def load_named_settings_presets() -> Dict[str, Dict[str, Any]]:
    path = named_settings_preset_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        presets = data.get("presets", {}) if isinstance(data, dict) else {}
        return {str(name): dict(value) for name, value in presets.items() if isinstance(value, dict)}
    except Exception:
        return {}


def save_named_settings_preset(name: str, payload: Dict[str, Any]) -> None:
    name = str(name).strip()
    if not name:
        raise ValueError("A settings preset needs a name.")
    presets = load_named_settings_presets()
    presets[name] = dict(payload)
    path = named_settings_preset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "presets": presets}, indent=2), encoding="utf-8")


def load_learned_presets() -> Dict[str, Dict[str, Any]]:
    path = learned_preset_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        presets = data.get("presets", {}) if isinstance(data, dict) else {}
        return {str(name): dict(value) for name, value in presets.items() if isinstance(value, dict)}
    except Exception:
        return {}


def save_learned_preset(name: str, payload: Dict[str, Any]) -> None:
    name = str(name).strip()
    if not name:
        raise ValueError("A learned preset needs a name.")
    presets = load_learned_presets()
    presets[name] = dict(payload)
    path = learned_preset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "presets": presets}, indent=2), encoding="utf-8")

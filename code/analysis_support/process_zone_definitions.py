#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/process_zone_definitions.py
# Renamed package path: code/analysis_support/process_zone_definitions.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import numpy as np
import pandas as pd


BOUNDARY_DISTANCE_COL = "zou_boundary_distance_km"
ABRUPT_DISTANCE_COL = "tlrts_distance_km"
DOMAIN_COL = "domain"

# Legacy manuscript thresholds retained for older sensitivity calls. New Figure 03/04
# process groups use the target-specific piecewise values below unless explicit
# overrides are passed to zone_masks/process_zone_summary.
BOUNDARY_FRONTAL_KM = 5.0
ABRUPT_CORE_KM = 2.0
ABRUPT_BACKGROUND_KM = 6.0

TARGETS = ("d_u", "grad_mag_km")

BOUNDARY_TRANSITION_KM = {
    "d_u": {"npf": 3.25, "pf": 3.25},
    "grad_mag_km": {"npf": 4.25, "pf": 1.75},
}

ABRUPT_IMPACT_KM = {
    "d_u": {"pf_all": 3.25, "pf_transition": 4.25, "pf_interior": 2.75},
    "grad_mag_km": {"pf_all": 8.75, "pf_transition": 5.25, "pf_interior": 9.25},
}

EXTREME_THRESHOLDS = {
    ("pf", "d_u"): -16.5,
    ("npf", "d_u"): -7.4,
    ("pf", "grad_mag_km"): 30.5,
    ("npf", "grad_mag_km"): 17.0,
}

ZONE_LABELS = {
    "npf_interior": "NPF interior",
    "npf_transition": "NPF transition",
    "pf_transition_impacted": "PF transition + thermokarst-feature impacted",
    "pf_transition_background": "PF transition background",
    "pf_interior_impacted": "PF interior + thermokarst-feature impacted",
    "pf_interior_background": "PF interior background",
}

EXPORT_ZONE_KEYS = {
    "npf_interior": "npf_interior",
    "npf_transition": "npf_transition",
    "pf_transition_impacted": "pf_transition_thermokarst_feature_impacted",
    "pf_transition_background": "pf_transition_background",
    "pf_interior_impacted": "pf_interior_thermokarst_feature_impacted",
    "pf_interior_background": "pf_interior_background",
}

EXPORT_COLUMN_NAMES = {
    "npf_boundary_transition_km": "npf_dB_transition_km",
    "pf_boundary_transition_km": "pf_dB_transition_km",
    "pf_transition_abrupt_impact_km": "pf_transition_dA_impact_km",
    "pf_interior_abrupt_impact_km": "pf_interior_dA_impact_km",
    "row_boundary_transition_km": "row_dB_transition_km",
    "row_abrupt_impact_km": "row_dA_impact_km",
}

ZONE_DOMAINS = {
    "npf_interior": "npf",
    "npf_transition": "npf",
    "pf_transition_impacted": "pf",
    "pf_transition_background": "pf",
    "pf_interior_impacted": "pf",
    "pf_interior_background": "pf",
}

NPF_ZONE_ORDER = ["npf_transition", "npf_interior"]
PF_ZONE_ORDER = [
    "pf_transition_impacted",
    "pf_transition_background",
    "pf_interior_impacted",
    "pf_interior_background",
]
ZONE_ORDER = [*NPF_ZONE_ORDER, *PF_ZONE_ORDER]

PF_MATRIX_ZONES = [
    ["pf_transition_impacted", "pf_transition_background"],
    ["pf_interior_impacted", "pf_interior_background"],
]

PF_MATRIX_ROW_KEYS = ["pf_transition", "pf_interior"]
PF_MATRIX_COL_KEYS = ["impacted", "background"]
PF_MATRIX_ROW_LABELS = ["Transition", "Interior"]
PF_MATRIX_COL_LABELS = ["Thermokarst-feature impacted", "Background"]


def normalize_target(target: str) -> str:
    if target not in TARGETS:
        raise ValueError(f"Unsupported target: {target!r}")
    return target


def numeric_array(df: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def domain_array(df: pd.DataFrame) -> np.ndarray:
    return df[DOMAIN_COL].astype(str).str.lower().to_numpy()


def boundary_transition_km(target: str, domain: str) -> float:
    target = normalize_target(target)
    key = str(domain).lower()
    return float(BOUNDARY_TRANSITION_KM[target][key])


def abrupt_impact_km(target: str, pf_boundary_class: str) -> float:
    target = normalize_target(target)
    key = str(pf_boundary_class).lower()
    return float(ABRUPT_IMPACT_KM[target][key])


def threshold_values(
    target: str,
    *,
    boundary_frontal_km: float | None = None,
    abrupt_core_km: float | None = None,
) -> dict[str, float]:
    target = normalize_target(target)
    if boundary_frontal_km is None:
        npf_boundary = boundary_transition_km(target, "npf")
        pf_boundary = boundary_transition_km(target, "pf")
    else:
        npf_boundary = float(boundary_frontal_km)
        pf_boundary = float(boundary_frontal_km)
    if abrupt_core_km is None:
        pf_transition_abrupt = abrupt_impact_km(target, "pf_transition")
        pf_interior_abrupt = abrupt_impact_km(target, "pf_interior")
    else:
        pf_transition_abrupt = float(abrupt_core_km)
        pf_interior_abrupt = float(abrupt_core_km)
    return {
        "npf_boundary_transition_km": float(npf_boundary),
        "pf_boundary_transition_km": float(pf_boundary),
        "pf_transition_abrupt_impact_km": float(pf_transition_abrupt),
        "pf_interior_abrupt_impact_km": float(pf_interior_abrupt),
    }


def row_thresholds_for_zone(target: str, zone_key: str) -> tuple[float, float]:
    thresholds = threshold_values(target)
    domain = ZONE_DOMAINS[zone_key]
    if domain == "npf":
        return thresholds["npf_boundary_transition_km"], np.nan
    if zone_key.startswith("pf_transition"):
        return thresholds["pf_boundary_transition_km"], thresholds["pf_transition_abrupt_impact_km"]
    return thresholds["pf_boundary_transition_km"], thresholds["pf_interior_abrupt_impact_km"]


def zone_masks(
    df: pd.DataFrame,
    *,
    target: str = "d_u",
    boundary_frontal_km: float | None = None,
    abrupt_core_km: float | None = None,
    abrupt_background_km: float | None = None,
) -> dict[str, np.ndarray]:
    _ = abrupt_background_km
    thresholds = threshold_values(
        target,
        boundary_frontal_km=boundary_frontal_km,
        abrupt_core_km=abrupt_core_km,
    )
    domain = domain_array(df)
    d_b = numeric_array(df, BOUNDARY_DISTANCE_COL)
    d_a = numeric_array(df, ABRUPT_DISTANCE_COL) if ABRUPT_DISTANCE_COL in df else np.full(len(df), np.nan)

    is_pf = domain == "pf"
    is_npf = domain == "npf"
    boundary_ok = np.isfinite(d_b) & (d_b >= 0.0)
    abrupt_ok = np.isfinite(d_a) & (d_a >= 0.0)

    npf_transition = is_npf & boundary_ok & (d_b < thresholds["npf_boundary_transition_km"])
    npf_interior = is_npf & boundary_ok & (d_b >= thresholds["npf_boundary_transition_km"])
    pf_transition = is_pf & boundary_ok & (d_b < thresholds["pf_boundary_transition_km"])
    pf_interior = is_pf & boundary_ok & (d_b >= thresholds["pf_boundary_transition_km"])

    transition_impacted = abrupt_ok & (d_a < thresholds["pf_transition_abrupt_impact_km"])
    transition_background = abrupt_ok & (d_a >= thresholds["pf_transition_abrupt_impact_km"])
    interior_impacted = abrupt_ok & (d_a < thresholds["pf_interior_abrupt_impact_km"])
    interior_background = abrupt_ok & (d_a >= thresholds["pf_interior_abrupt_impact_km"])

    return {
        "npf_transition": npf_transition,
        "npf_interior": npf_interior,
        "pf_transition_impacted": pf_transition & transition_impacted,
        "pf_transition_background": pf_transition & transition_background,
        "pf_interior_impacted": pf_interior & interior_impacted,
        "pf_interior_background": pf_interior & interior_background,
    }


def assign_process_zone(
    df: pd.DataFrame,
    *,
    target: str = "d_u",
    boundary_frontal_km: float | None = None,
    abrupt_core_km: float | None = None,
    abrupt_background_km: float | None = None,
) -> pd.Series:
    masks = zone_masks(
        df,
        target=target,
        boundary_frontal_km=boundary_frontal_km,
        abrupt_core_km=abrupt_core_km,
        abrupt_background_km=abrupt_background_km,
    )
    zone = np.full(len(df), "other", dtype=object)
    for key in ZONE_ORDER:
        zone[masks[key]] = key
    return pd.Series(zone, index=df.index, name="process_zone")


def extreme_mask(df: pd.DataFrame, *, target: str) -> np.ndarray:
    target = normalize_target(target)
    domain = domain_array(df)
    values = numeric_array(df, target)
    out = np.zeros(len(df), dtype=bool)
    for zone_domain in ("pf", "npf"):
        threshold = EXTREME_THRESHOLDS[(zone_domain, target)]
        domain_mask = domain == zone_domain
        if target == "d_u":
            out |= domain_mask & np.isfinite(values) & (values <= threshold)
        else:
            out |= domain_mask & np.isfinite(values) & (values >= threshold)
    return out


def process_zone_summary(
    df: pd.DataFrame,
    *,
    target: str,
    boundary_frontal_km: float | None = None,
    abrupt_core_km: float | None = None,
    abrupt_background_km: float | None = None,
) -> pd.DataFrame:
    target = normalize_target(target)
    thresholds = threshold_values(
        target,
        boundary_frontal_km=boundary_frontal_km,
        abrupt_core_km=abrupt_core_km,
    )
    masks = zone_masks(
        df,
        target=target,
        boundary_frontal_km=boundary_frontal_km,
        abrupt_core_km=abrupt_core_km,
        abrupt_background_km=abrupt_background_km,
    )
    extreme = extreme_mask(df, target=target)
    domain = domain_array(df)
    total_extreme_by_domain = {
        zone_domain: int(np.count_nonzero(extreme & (domain == zone_domain)))
        for zone_domain in ("npf", "pf")
    }

    rows: list[dict[str, object]] = []
    for zone_key in ZONE_ORDER:
        mask = masks[zone_key]
        zone_domain = ZONE_DOMAINS[zone_key]
        n_pixels = int(np.count_nonzero(mask))
        extreme_n = int(np.count_nonzero(mask & extreme))
        domain_n = int(np.count_nonzero(domain == zone_domain))
        total_extreme = total_extreme_by_domain[zone_domain]
        row_boundary, row_abrupt = row_thresholds_for_zone(target, zone_key)
        if boundary_frontal_km is not None:
            row_boundary = float(boundary_frontal_km)
        if abrupt_core_km is not None and zone_domain == "pf":
            row_abrupt = float(abrupt_core_km)
        rows.append(
            {
                "target": target,
                "zone_key": zone_key,
                "display_label": ZONE_LABELS[zone_key],
                "domain": zone_domain,
                "n_pixels": n_pixels,
                "domain_n_pixels": domain_n,
                "extreme_n": extreme_n,
                "domain_extreme_n": total_extreme,
                "extreme_density": extreme_n / n_pixels if n_pixels > 0 else np.nan,
                "domain_extreme_share": extreme_n / total_extreme if total_extreme > 0 else np.nan,
                "sample_share": n_pixels / domain_n if domain_n > 0 else np.nan,
                "npf_boundary_transition_km": thresholds["npf_boundary_transition_km"],
                "pf_boundary_transition_km": thresholds["pf_boundary_transition_km"],
                "pf_transition_abrupt_impact_km": thresholds["pf_transition_abrupt_impact_km"],
                "pf_interior_abrupt_impact_km": thresholds["pf_interior_abrupt_impact_km"],
                "row_boundary_transition_km": row_boundary,
                "row_abrupt_impact_km": row_abrupt,
                "threshold_source": "piecewise_profile_fit" if boundary_frontal_km is None and abrupt_core_km is None else "explicit_override",
            }
        )
    return pd.DataFrame(rows)


def export_process_zone_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return a submission-facing copy with terminology-safe keys and columns."""
    out = summary.copy()
    out["zone_key"] = out["zone_key"].map(EXPORT_ZONE_KEYS).fillna(out["zone_key"])
    return out.rename(columns=EXPORT_COLUMN_NAMES)

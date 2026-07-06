"""Experiment configurations for hybrid steering ablation.

C1-C4: steering model combinations.
D1-D4: crush threshold variations (all use C4 steering).
"""

CONFIGS: dict[str, dict[str, bool]] = {
    "C1": {"sfm": True, "ttc": False, "orca": False, "crush": False},
    "C2": {"sfm": True, "ttc": True, "orca": False, "crush": False},
    "C3": {"sfm": True, "ttc": False, "orca": True, "crush": False},
    "C4": {"sfm": True, "ttc": True, "orca": True, "crush": True},
}

CRUSH_CONFIGS: dict[str, float | None] = {
    "D1": None,
    "D2": 5.0,
    "D3": 5.5,
    "D4": 7.0,
}

BOTTLENECK_WIDTHS: list[float] = [0.8, 1.0, 1.2, 1.8, 2.4, 3.6]


def get_config(name: str) -> dict[str, bool]:
    """Look up a steering configuration by name.

    Handles both C-configs (C1-C4) and D-configs (D1-D4).
    D-configs return the C4 steering config.

    Args:
        name: Configuration name (C1-C4 or D1-D4).

    Returns:
        Dict with boolean keys sfm, ttc, orca, crush.
    """
    if name in CONFIGS:
        return CONFIGS[name]
    if name in CRUSH_CONFIGS:
        # D-configs use C4 base (full hybrid) with variable crush threshold.
        # D1 disables crush entirely; D2-D4 enable crush with different rho_crit.
        crush_on = CRUSH_CONFIGS[name] is not None
        return {"sfm": True, "ttc": True, "orca": True, "crush": crush_on}
    raise KeyError(f"Unknown config: {name}")


def get_param_overrides(name: str) -> dict:
    """Get parameter overrides for D-configs.

    Args:
        name: Configuration name.

    Returns:
        Dict of parameter overrides (empty for C-configs).
    """
    if name in CRUSH_CONFIGS and CRUSH_CONFIGS[name] is not None:
        return {"rho_crit": CRUSH_CONFIGS[name]}
    return {}

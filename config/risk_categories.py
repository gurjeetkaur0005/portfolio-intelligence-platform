RISK_CATEGORIES = {
    "ultra_conservative": {
        "target": [0.15, 0.00, 0.60, 0.00, 0.10, 0.15],
        "drift_band": 0.02,
        "weight": 0.10,
    },
    "conservative": {
        "target": [0.25, 0.05, 0.45, 0.05, 0.12, 0.08],
        "drift_band": 0.025,
        "weight": 0.24,
    },
    "balanced": {
        "target": [0.40, 0.10, 0.30, 0.05, 0.10, 0.05],
        "drift_band": 0.03,
        "weight": 0.36,
    },
    "aggressive": {
        "target": [0.55, 0.15, 0.15, 0.05, 0.07, 0.03],
        "drift_band": 0.04,
        "weight": 0.20,
    },
    "ultra_aggressive": {
        "target": [0.65, 0.20, 0.05, 0.03, 0.05, 0.02],
        "drift_band": 0.05,
        "weight": 0.10,
    },
}
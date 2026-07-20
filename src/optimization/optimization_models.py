from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class OptimizationResult:
    status: str
    trade_weights: Optional[np.ndarray]
    post_trade_weights: Optional[np.ndarray]
    tracking_error_before: Optional[float]
    tracking_error_after: Optional[float]
    turnover: Optional[float]
    objective_value: Optional[float]
    message: str
    
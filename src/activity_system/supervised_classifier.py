"""Stage C lightweight multinomial logistic classifier on standardized temporal gait features."""

from typing import Dict, Tuple
import numpy as np

from activity_system.schemas import ActivityLabel


class SupervisedActivityClassifier:
    """Calibrated linear softmax model over engineered temporal features (Stage C comparison model)."""

    FEATURE_NAMES = (
        "speed_bh_per_sec",
        "ankle_std_bh",
        "knee_std_bh",
        "stride_amp_bh",
        "cadence_hz",
        "lower_body_visible_ratio",
    )

    CLASSES = (
        ActivityLabel.STANDING,
        ActivityLabel.WALKING,
        ActivityLabel.RUNNING,
    )

    def __init__(self) -> None:
        # Calibrated weights for [speed_bh_per_sec, ankle_std_bh, knee_std_bh, stride_amp_bh, cadence_hz, lower_vis]
        self._weights = np.array(
            [
                [-4.5, -18.0, -10.0, -16.0, -1.5, 1.2],   # STANDING
                [1.8, 12.0, 8.0, 14.0, 1.2, 1.5],         # WALKING
                [4.2, 22.0, 15.0, 20.0, 2.2, 1.5],        # RUNNING
            ],
            dtype=np.float64,
        )
        self._bias = np.array([2.4, -1.4, -6.8], dtype=np.float64)

    def predict_proba(self, features: Dict[str, float]) -> Tuple[ActivityLabel, float, Dict[str, float]]:
        """Predict activity class and calibrated softmax probability when visibility requirements are met."""
        if features.get("lower_body_visible_ratio", 0.0) < 0.40:
            return ActivityLabel.UNKNOWN, 0.0, {c.value: 0.0 for c in self.CLASSES}

        x = np.array(
            [float(features.get(name, 0.0)) for name in self.FEATURE_NAMES],
            dtype=np.float64,
        )
        logits = (self._weights @ x) + self._bias
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        prob_map = {self.CLASSES[i].value: round(float(probs[i]), 4) for i in range(len(self.CLASSES))}
        if best_prob < 0.55:
            return ActivityLabel.UNKNOWN, best_prob, prob_map
        return self.CLASSES[best_idx], round(best_prob, 4), prob_map

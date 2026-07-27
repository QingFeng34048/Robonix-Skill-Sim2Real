from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyLimits:
    joint_min_rad: np.ndarray
    joint_max_rad: np.ndarray
    max_delta_rad: np.ndarray
    gripper_min: float = 0.0
    gripper_max: float = 1.0


class SafetyFilter:
    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits

    def apply(
        self,
        *,
        current_joints: list[float],
        action: list[float],
        task_max_delta: float,
    ) -> tuple[list[float], float]:
        raw = np.asarray(action, dtype=np.float64)

        if raw.shape != (7,):
            raise ValueError(f"expected action shape (7,), got {raw.shape}")

        current = np.asarray(current_joints, dtype=np.float64)

        allowed_delta = np.minimum(
            self.limits.max_delta_rad,
            np.full(6, task_max_delta),
        )

        delta = np.clip(raw[:6], -allowed_delta, allowed_delta)
        target = current + delta
        target = np.clip(
            target,
            self.limits.joint_min_rad,
            self.limits.joint_max_rad,
        )

        gripper = float(
            np.clip(
                raw[6],
                self.limits.gripper_min,
                self.limits.gripper_max,
            )
        )

        if not np.all(np.isfinite(target)) or not np.isfinite(gripper):
            raise ValueError("action contains NaN or Inf")

        return target.tolist(), gripper


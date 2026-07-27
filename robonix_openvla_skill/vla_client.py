from __future__ import annotations

import json
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class VLAResult:
    actions: list[list[float]]
    done: bool = False
    success: bool | None = None
    detail: str = ""


class VLAClient:
    def __init__(self, server_url: str, timeout_s: float) -> None:
        self.server_url = server_url
        self.timeout_s = timeout_s

    def healthcheck(self) -> None:
        # 推荐在 server_oft.py 增加 /health。
        response = requests.get(
            self.server_url.rsplit("/act", 1)[0] + "/health",
            timeout=min(self.timeout_s, 3.0),
        )
        response.raise_for_status()

    def predict(
        self,
        *,
        task_id: str,
        instruction: str,
        image_jpeg: bytes,
        state: list[float],
    ) -> VLAResult:
        response = requests.post(
            self.server_url,
            files={"image": ("observation.jpg", image_jpeg, "image/jpeg")},
            data={
                "task_id": task_id,
                "instruction": instruction,
                "state": json.dumps(state),
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()

        payload = response.json()
        actions = payload.get("action_chunk")

        if actions is None:
            action = payload.get("action")
            if action is None:
                raise RuntimeError("VLA response has neither action nor action_chunk")
            actions = action if action and isinstance(action[0], list) else [action]

        if not actions:
            raise RuntimeError("VLA response contains an empty action chunk")

        return VLAResult(
            actions=actions,
            done=bool(payload.get("done", False)),
            success=payload.get("success"),
            detail=str(payload.get("detail", "")),
        )


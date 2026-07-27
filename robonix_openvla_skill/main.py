from __future__ import annotations

import os

from robonix_api import Deferred, Err, Ok, Skill

from openvla_oft_mcp import (
    CancelTask_Request,
    CancelTask_Response,
    ExecuteTask_Request,
    ExecuteTask_Response,
    GetTaskStatus_Request,
    GetTaskStatus_Response,
)

from .runtime_config import RuntimeConfig
from .task_manager import TaskManager


skill = Skill(
    id=os.environ.get("RBNX_INSTANCE_NAME", "openvla_oft"),
    namespace="robonix/skill/openvla_oft",
)

settings: RuntimeConfig | None = None
manager: TaskManager | None = None


@skill.on_init
def init(cfg: dict):
    global settings, manager

    try:
        settings = RuntimeConfig.from_dict(cfg)
        settings.validate()
        manager = TaskManager(skill=skill, settings=settings)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        return Err(f"invalid configuration: {exc}")

    return Ok()


@skill.on_activate
def activate():
    if manager is None:
        return Err("skill has not been initialized")

    try:
        manager.connect_dependencies()
    except ValueError as exc:
        # Atlas 暂时找不到 Camera 或 Arm Primitive。
        return Deferred(str(exc))
    except Exception as exc:
        return Err(f"activation failed: {exc}")

    return Ok()


@skill.on_deactivate
def deactivate():
    if manager is not None:
        manager.cancel_all(reason="skill deactivated")
        manager.close()
    return Ok()


@skill.on_shutdown
def shutdown():
    return deactivate()


@skill.mcp(
    "robonix/skill/openvla_oft/execute",
    description="Execute a configured language-conditioned manipulation task.",
)
def execute(req: ExecuteTask_Request) -> ExecuteTask_Response:
    if manager is None:
        raise RuntimeError("skill is not initialized")

    result = manager.start(
        task_id=req.task_id,
        instruction=req.instruction,
        timeout_s=req.timeout_s,
    )

    return ExecuteTask_Response(
        accepted=result.accepted,
        run_id=result.run_id,
        detail=result.detail,
    )


@skill.mcp("robonix/skill/openvla_oft/execute/status")
def status(req: GetTaskStatus_Request) -> GetTaskStatus_Response:
    if manager is None:
        raise RuntimeError("skill is not initialized")

    current = manager.status(req.run_id)

    return GetTaskStatus_Response(
        state=current.state,
        detail=current.detail,
        progress=current.progress,
        current_step=current.current_step,
        max_steps=current.max_steps,
    )


@skill.mcp("robonix/skill/openvla_oft/execute/cancel")
def cancel(req: CancelTask_Request) -> CancelTask_Response:
    if manager is None:
        raise RuntimeError("skill is not initialized")

    accepted, detail = manager.cancel(req.run_id)

    return CancelTask_Response(
        accepted=accepted,
        detail=detail,
    )


if __name__ == "__main__":
    skill.run()


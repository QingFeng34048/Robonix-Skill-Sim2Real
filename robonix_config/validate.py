from robonix_config.schema import ExperimentConfig, TaskConfig


def get_task_map(cfg: ExperimentConfig) -> dict[str, TaskConfig]:
    return {
        task.task_id: task
        for task in cfg.tasks
        if task.enabled
    }


def validate_config(cfg: ExperimentConfig) -> None:
    task_ids = [task.task_id for task in cfg.tasks]

    if not task_ids:
        raise ValueError("No tasks configured.")

    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate task_id found.")

    task_map = get_task_map(cfg)

    if cfg.collect.default_task_id not in task_map:
        raise ValueError(
            f"Unknown collect.default_task_id: "
            f"{cfg.collect.default_task_id}"
        )

    if cfg.client.default_task_id not in task_map:
        raise ValueError(
            f"Unknown client.default_task_id: "
            f"{cfg.client.default_task_id}"
        )

    for task in task_map.values():
        if len(task.init_pose) != 7:
            raise ValueError(
                f"{task.task_id}.init_pose must contain 7 values."
            )

        if not task.instruction.strip():
            raise ValueError(
                f"{task.task_id} has empty instruction."
            )

        if task.execute_chunk_steps < 1:
            raise ValueError(
                f"{task.task_id}.execute_chunk_steps must be >= 1."
            )

    if cfg.camera.model_res != [224, 224]:
        raise ValueError(
            "Current RLDS feature shape requires model_res=[224, 224]."
        )

    if cfg.train.use_l1_regression and cfg.train.use_diffusion:
        raise ValueError(
            "L1 regression and diffusion cannot both be enabled."
        )

    if not cfg.train.use_l1_regression and not cfg.train.use_diffusion:
        raise ValueError(
            "Enable one action prediction method."
        )

    if not 0.0 <= cfg.dataset.validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio must be in [0, 1)."
        )


from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TaskConfig:
    task_id: str
    instruction: str
    init_pose: list[float]

    max_steps: int = 120
    max_delta: float = 0.05
    execute_chunk_steps: int = 2
    enabled: bool = True


@dataclass
class RobotConfig:
    can_port: str = "can0"
    rad_to_sdk_int: float = 57295.7795
    piper_raw_to_degree: float = 0.001

    gripper_open_sdk: int = 80000
    gripper_close_sdk: int = 0
    gripper_threshold_raw: int = 3000


@dataclass
class CameraConfig:
    camera_id: int = 1
    capture_res: list[int] = field(default_factory=lambda: [640, 480])
    model_res: list[int] = field(default_factory=lambda: [224, 224])
    flip_code: int = -1


@dataclass
class DatasetConfig:
    name: str = "piper_multitask"
    hdf5_root: Path = Path("outputs/dataset_hdf5")
    rlds_root: Path = Path("outputs/data_rlds")
    version: str = "1.0.0"

    exclude_fail: bool = True
    min_action_norm: float = 0.0
    validation_ratio: float = 0.1


@dataclass
class CollectConfig:
    fps: int = 10
    default_task_id: str = "pick_banana"
    return_to_init_after_episode: bool = True


@dataclass
class ConvertConfig:
    merge_tasks: bool = True
    preserve_gripper_changes: bool = True
    random_seed: int = 42


@dataclass
class TrainConfig:
    vla_path: str = "openvla/openvla-7b"
    run_root_dir: Path = Path("outputs/checkpoints")

    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    use_proprio: bool = True
    num_images_in_input: int = 1

    batch_size: int = 4
    learning_rate: float = 5e-5
    grad_accumulation_steps: int = 1
    max_steps: int = 30000
    save_freq: int = 1000
    image_aug: bool = True

    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = False

    shuffle_buffer_size: int = 100000
    use_val_set: bool = True
    val_freq: int = 2000
    val_time_limit: int = 180

    wandb_entity: str = ""
    wandb_project: str = "openvla-oft"
    wandb_log_freq: int = 10


@dataclass
class ServerConfig:
    checkpoint_path: Path = Path("outputs/checkpoints/latest")
    host: str = "0.0.0.0"
    port: int = 8001

    save_images: bool = False
    save_dir: Path = Path("outputs/inference_images")

    center_crop: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    return_first_action_for_legacy: bool = True


@dataclass
class ClientConfig:
    server_url: str = "http://localhost:8001/act"
    default_task_id: str = "pick_banana"
    control_freq: int = 10
    request_timeout: float = 10.0
    use_action_chunk: bool = True


@dataclass
class ExperimentConfig:
    project_name: str = "piper_multitask"

    tasks: list[TaskConfig] = field(default_factory=list)

    robot: RobotConfig = field(default_factory=RobotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    convert: ConvertConfig = field(default_factory=ConvertConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    client: ClientConfig = field(default_factory=ClientConfig)


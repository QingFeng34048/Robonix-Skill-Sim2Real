import cgi
import io
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
from PIL import Image

# Run this server inside the openvla-oft repo / conda env.
# These imports are from OpenVLA-OFT.
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM


# ====== Edit these for your experiment ======
# IMPORTANT:
# This must be the OFT checkpoint ROOT directory, for example:
#   .../openvla-7b+pick_up_the_banana2+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--1000_chkpt
# It should contain:
#   dataset_statistics.json
#   action_head--1000_checkpoint.pt
#   config.json / model files if merge_lora_during_training=True
#   lora_adapter/
#
# Do NOT point this to the old OpenVLA base model path.
# Do NOT point this to checkpoint_root/lora_adapter.
OFT_CHECKPOINT_PATH = YOUR_OFT_CHECKPOINT_PATH_HERE

UNNORM_KEY = "pick_up_the_banana"

HOST = "0.0.0.0"
PORT = 8001
SAVE_DIR = YOUR_SAVE_DIR_HERE  # Set to None to disable saving incoming images.
# Must match your OFT finetune config.
USE_L1_REGRESSION = True
USE_DIFFUSION = False
USE_FILM = False
NUM_IMAGES_IN_INPUT = 1

# Set this to True only if you trained OFT with use_proprio=True
# and your client sends the exact same proprio/state vector order used in the RLDS dataset.
USE_PROPRIO = False

# For best distribution match:
# - True is commonly used in official OFT LIBERO evaluation.
# - For a real camera, test both. If the banana/object is near the image edge, False may work better.
CENTER_CROP = False

# Quantization saves VRAM but can slightly change behavior. For best quality, keep both False.
LOAD_IN_8BIT = False
LOAD_IN_4BIT = False

# Keep old-client compatibility: return first action in "action".
# Also return full OFT chunk in "action_chunk".
RETURN_FIRST_ACTION_ONLY_FOR_LEGACY = True


def _build_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        base_model=YOUR_BASE_MODEL_NAME_OR_PATH_HERE,  # e.g. "decapoda-research/llama-7b-hf"
        pretrained_checkpoint=OFT_CHECKPOINT_PATH,
        
        use_l1_regression=USE_L1_REGRESSION,
        use_diffusion=USE_DIFFUSION,
        use_film=USE_FILM,
        num_images_in_input=NUM_IMAGES_IN_INPUT,
        use_proprio=USE_PROPRIO,
        load_in_8bit=LOAD_IN_8BIT,
        load_in_4bit=LOAD_IN_4BIT,
        center_crop=CENTER_CROP,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=UNNORM_KEY,

        # Required by OpenVLA-OFT helper code in some branches/config paths.
        lora_rank=32,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50,
    )


def _validate_oft_checkpoint(path: str) -> None:
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"OFT checkpoint path does not exist: {checkpoint}")
    if not (checkpoint / "dataset_statistics.json").exists():
        raise FileNotFoundError(
            f"Missing dataset_statistics.json in {checkpoint}. "
            "Use the OFT checkpoint root directory, not lora_adapter."
        )
    action_heads = list(checkpoint.glob("action_head--*checkpoint.pt"))
    if len(action_heads) != 1:
        raise FileNotFoundError(
            f"Expected exactly one action_head--*checkpoint.pt in {checkpoint}, found {len(action_heads)}. "
            "Use a single OFT checkpoint directory such as ...--1000_chkpt."
        )


def _load_policy_bundle():
    _validate_oft_checkpoint(OFT_CHECKPOINT_PATH)

    cfg = _build_cfg()

    # get_vla() loads the merged OFT VLA checkpoint and dataset statistics.
    # get_action_head() loads action_head--*_checkpoint.pt.
    model = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=model.llm_dim)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, llm_dim=model.llm_dim, proprio_dim=PROPRIO_DIM)

    return cfg, model, processor, action_head, proprio_projector


def _image_from_bytes(image_bytes: bytes, form: cgi.FieldStorage) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        width = form.getvalue("width")
        height = form.getvalue("height")
        if width is None or height is None:
            raise
        width = int(width)
        height = int(height)
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        if arr.size != width * height * 3:
            raise ValueError("raw image size mismatch")
        arr = arr.reshape((height, width, 3))
        return Image.fromarray(arr, mode="RGB")


def _to_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class VLARequestHandler(BaseHTTPRequestHandler):
    cfg = None
    model = None
    processor = None
    action_head = None
    proprio_projector = None

    step = 0
    cond = threading.Condition()
    busy = False

    def _send_json(self, status_code: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        with VLARequestHandler.cond:
            while VLARequestHandler.busy:
                VLARequestHandler.cond.wait()
            VLARequestHandler.busy = True

        try:
            if self.path != "/act":
                self._send_json(404, {"error": "not found"})
                return

            content_type = self.headers.get("Content-Type")
            if not content_type:
                self._send_json(400, {"error": "missing Content-Type"})
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
                keep_blank_values=True,
            )

            instruction = form.getvalue("instruction")
            if instruction is None:
                self._send_json(400, {"error": "missing instruction"})
                return

            image_field = form["image"] if "image" in form else None
            if image_field is None:
                self._send_json(400, {"error": "missing image"})
                return

            image_bytes = image_field.file.read() if hasattr(image_field, "file") else image_field.value
            try:
                image = _image_from_bytes(image_bytes, form)
            except Exception:
                self._send_json(400, {"error": "invalid image"})
                return

            os.makedirs(SAVE_DIR, exist_ok=True)
            VLARequestHandler.step += 1
            np_image = np.array(image, dtype=np.uint8)
            imageio.imwrite(os.path.join(SAVE_DIR, f"{VLARequestHandler.step:06d}.png"), np_image)

            obs = {
                "full_image": np_image,
                "task_description": str(instruction),
            }

            state_raw = form.getvalue("state")
            state_list = None
            if state_raw is not None:
                try:
                    state_list = json.loads(state_raw)
                except Exception:
                    state_list = state_raw

            if self.cfg.use_proprio:
                if state_list is None:
                    self._send_json(400, {"error": "missing state; server USE_PROPRIO=True"})
                    return
                obs["state"] = np.asarray(state_list, dtype=np.float32)

            try:
                actions = get_vla_action(
                    self.cfg,
                    self.model,
                    self.processor,
                    obs,
                    str(instruction),
                    action_head=self.action_head,
                    proprio_projector=self.proprio_projector,
                    noisy_action_projector=None,
                    use_film=self.cfg.use_film,
                )

                action_chunk = [_to_list(action) for action in actions]
                first_action = action_chunk[0]

                print(
                    f"step={VLARequestHandler.step}, "
                    f"instruction={instruction}, "
                    f"state={state_list}, "
                    f"first_action={first_action}, "
                    f"chunk_len={len(action_chunk)}"
                )

                payload = {
                    "action": first_action if RETURN_FIRST_ACTION_ONLY_FOR_LEGACY else action_chunk,
                    "action_chunk": action_chunk,
                    "chunk_len": len(action_chunk),
                }
                self._send_json(200, payload)

            except Exception as exc:
                print(traceback.format_exc())
                self._send_json(500, {"error": str(exc)})

        finally:
            with VLARequestHandler.cond:
                VLARequestHandler.busy = False
                VLARequestHandler.cond.notify()


def main() -> None:
    cfg, model, processor, action_head, proprio_projector = _load_policy_bundle()

    VLARequestHandler.cfg = cfg
    VLARequestHandler.model = model
    VLARequestHandler.processor = processor
    VLARequestHandler.action_head = action_head
    VLARequestHandler.proprio_projector = proprio_projector

    print(f"Loaded OFT checkpoint: {OFT_CHECKPOINT_PATH}")
    print(f"Serving on http://{HOST}:{PORT}/act")
    server = ThreadingHTTPServer((HOST, PORT), VLARequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()

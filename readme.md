<p align="center">
  <img src="images/robonix-logo.svg" alt="Robonix" width="420" />
</p>

# Robonix-Skill-Toolkit

This project provides practical guidance for data collection, fine-tuning, deployment, and optimization on physical robotic arms based on VLA models.

We provide the following example to demonstrate the performance of Robonix-Skill-Toolkit:

https://github.com/user-attachments/assets/daf14475-6878-4553-8284-d4ef6c2db285

# News

* 🔥 [2026-06] We released Robonix-Skill-Toolkit for [Robonix](https://github.com/syswonder/robonix), built on the [OpenVLA-OFT](https://openvla-oft.github.io) framework and the [AgileX Piper Robotic Arm](https://github.com/agilexrobotics/Agilex-College)!

# Repository Structure

The robotic arm side code runs on a Dell Linux machine, while the VLA side code runs on the IFLab server.

- `client/`: Client-side code for the Piper robotic arm, including camera checking, data collection, and deployment scripts on the Dell Linux machine.
- `openvla-oft/`: Server-side VLA code on the IFLab server, including fine-tuning, inference, and model serving scripts.
- `images/`: Image assets used in this README, such as logos or example figures.
- `data/`: Dataset directory for collected HDF5 files and converted RLDS-format data.
- `outputs/`: Model output directory for fine-tuned checkpoints, logs, and related artifacts.

# Step1: Getting Started & Environment Setup

```
# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

After manually configuring all file paths, run `openvla-oft/vla-scripts/finetune.sh`. The core fine-tuning script is located at `openvla-oft/vla-scripts/finetune.py`.

# Step2: Data Collection

`client/check_cam.py` is used to verify that the camera displays images correctly. Test camera IDs 0, 1, 2, and 3, then record the valid ID.

`client/collect_data.py` is used for data collection. Run the script, enter a prompt (e.g., "pick up the banana") as instructed, and press Enter to start the collection process. Press `S` to begin recording and `Y` to stop recording.

`client/hdf5_to_rlds.py` converts the saved HDF5 files generated during data collection into datasets in RLDS format. A helper script, `client/run_rlds.sh`, is provided for this conversion.

# Step3: Data Example

Here is an example of the `feature.json` file for a dataset in RLDS format.
![image](https://github.com/QingFeng34048/image-and-video/blob/main/feature.png)

Here is an example of the `dataset_info.json` file for a dataset in RLDS format.
![image](https://github.com/QingFeng34048/image-and-video/blob/main/info.png)

# Step4: Fine-Tuning

The Piper robotic arm acts as the client, while OpenVLA serves as the server. The two communicate over a local area network via HTTP.
On the client side, the system accesses the camera to capture frames, receives text prompts, packages images, prompts, and robot states, and sends an HTTP POST request to the server.
On the server side, the system performs inference to compute robot actions and sends the results back to the client.
In our setup, a Dell laptop running Ubuntu controls the robotic arm. Connect the two USB cables from the robotic arm and camera to the laptop.
The server code is located in `openvla-oft/server_oft.py` and is launched via `openvla-oft/run.sh`. Once started, the server remains idle until it receives data and commands from the client.
Next, navigate to the client folder on the client machine and run `client/run.sh` to operate the robotic arm and observe its movements.

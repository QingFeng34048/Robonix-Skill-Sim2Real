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
=======
## Data Example

Here is an example of `dataset_statistics.json` of datasets in RLDS format.
```JSON
{
    "action": 
    {"mean": [-4.684889063355513e-05, 0.0018815468065440655, -0.005818227306008339,
     0.0005891860346309841, 0.0059196725487709045, -0.00021760746312793344, 0.5048092603683472],
 "std": [0.01849505864083767, 0.04295733943581581, 0.02245117537677288,
  0.010312630794942379, 0.021517977118492126, 0.007021446246653795, 0.49997571110725403],
  "max": [0.09862855076789856, 0.17505651712417603, 0.13838714361190796, 
  0.16306611895561218, 0.169960156083107, 0.05651378631591797, 1.0], 
  "min": [-0.13224360346794128, -0.169907808303833, -0.16931438446044922, 
  -0.0757472887635231, -0.10855948179960251, -0.14308208227157593, 0.0], 
  "q01": [-0.06464594677090645, -0.11778496503829956, -0.10294377714395524, 
  -0.03143058657646179, -0.04263914346694946, -0.023333258628845215, 0.0],
   "q99": [0.06227089822292339, 0.1293100583553317, 0.038070203065872284, 
   0.044037799313664576, 0.09472043052315718, 0.016853941679000922, 1.0]}, 
   "proprio": 
   {"mean": [0.19157855212688446, 0.8735357522964478, -0.7225562334060669, 
   0.02684261091053486, 0.41170039772987366, -0.8020003437995911, 0.5114427804946899],
    "std": [0.2147376984357834, 0.7654772400856018, 0.42819467186927795, 
    0.07230901718139648, 0.37241050601005554, 0.061794690787792206, 0.49987301230430603], 
    "max": [0.5764299035072327, 2.0375845432281494, 0.0077667152509093285, 
    0.2751336991786957, 1.3230293989181519, -0.5129871964454651, 1.0], 
    "min": [-0.1765400469303131, -0.043563418090343475, -1.3302026987075806, 
    -0.24099506437778473, -0.3140370845794678, -1.0806206464767456, 0.0], 
    "q01": [-0.14411846935749054, -0.0038222710136324167, -1.2606513500213623, 
    -0.20694369077682495, -0.10068978682160377, -0.9871463668346405, 0.0], 
    "q99": [0.5081002712249756, 2.036607265472412, 0.00753982225432992, 
    0.22860042661428662, 1.3032548427581787, -0.6741683483123779, 1.0]}, 
    "num_transitions": 3015, "num_trajectories": 20
}
```

Here is an example of `dataset_info.json` of datasets in RLDS format.
```JSON
{
  "fileFormat": "tfrecord",
  "moduleName": "abc",
  "name": "pick_up_the_banana",
  "splits": [
    {
      "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
      "name": "train",
      "numBytes": "631431835",
      "shardLengths": [
        "2",
        "3",
        "3",
        "2",
        "2",
        "3",
        "3",
        "2"
      ]
    }
  ],
  "version": "1.0.0"
}
```


# Step4: Fine-Tuning

The Piper robotic arm acts as the client, while OpenVLA serves as the server. The two communicate over a local area network via HTTP.
On the client side, the system accesses the camera to capture frames, receives text prompts, packages images, prompts, and robot states, and sends an HTTP POST request to the server.
On the server side, the system performs inference to compute robot actions and sends the results back to the client.
In our setup, a Dell laptop running Ubuntu controls the robotic arm. Connect the two USB cables from the robotic arm and camera to the laptop.
The server code is located in `openvla-oft/server_oft.py` and is launched via `openvla-oft/run.sh`. Once started, the server remains idle until it receives data and commands from the client.
Next, navigate to the client folder on the client machine and run `client/run.sh` to operate the robotic arm and observe its movements.

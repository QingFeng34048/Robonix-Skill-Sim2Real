#!/usr/bin/env bash
set -euo pipefail

PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"

if [[ ! -f "$ROS_SETUP" ]]; then
    echo "ROS setup not found: $ROS_SETUP" >&2
    exit 1
fi

source "$ROS_SETUP"

python3 -c 'import grpc_tools' >/dev/null
python3 -c 'import numpy, requests, yaml' >/dev/null

FLAGS=(--mcp --ros2)
[[ "${RBNX_BUILD_CLEAN:-}" == "1" ]] && FLAGS+=(--clean)

rbnx codegen -p "$PKG" "${FLAGS[@]}"

cd "$PKG/rbnx-build/codegen/ros2_idl"
colcon build

echo "[build] OpenVLA-OFT skill built successfully"


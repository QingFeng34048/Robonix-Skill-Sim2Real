#!/usr/bin/env bash
set -euo pipefail

PKG_ROOT="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ROS_DISTRO="${ROS_DISTRO:-humble}"

source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "$PKG_ROOT/rbnx-build/codegen/ros2_idl/install/setup.bash"

cd "$PKG_ROOT"

export PYTHONPATH="$(
    rbnx path robonix-api
):$PKG_ROOT:${PYTHONPATH:-}"

exec python3 -m robonix_openvla_skill.main


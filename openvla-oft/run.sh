et -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="openvla_tsc"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT"

echo "Starting OpenVLA inference server..."
echo "REPO_ROOT=$REPO_ROOT"
echo "CONDA_ENV=$CONDA_ENV"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
conda run --no-capture-output -n "$CONDA_ENV" python -u "/robotic_arm202606/openvla-oft/server_oft.py"
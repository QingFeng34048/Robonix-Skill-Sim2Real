cd /robotic_arm202606/openvla-oft
unset PYTHONPATH
#export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH="/robotic_arm202606/openvla-oft$PYTHONPATH"
cd vla-scripts
torchrun --standalone --nnodes 1 --nproc-per-node 1 finetune.py
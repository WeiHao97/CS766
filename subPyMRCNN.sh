#!/usr/bin/env bash

#SBATCH -p slurm_sbel_cmg
#SBATCH --account=skunkworks --qos=skunkworks_owner

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
#SBATCH --gres=gpu:1
#SBATCH -t 13-2:00 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH -o cuda_Training-%j.log

source activate maskrcnn1
## Load CUDA into your environment
module load cuda/9.0
module load gcc/5.1.0

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm
pip install -U scikit-image
pip install -U cython
pip install opencv-python

#export INSTALL_DIR=$PWD

# install pycocotools
#cd $INSTALL_DIR
#git clone https://github.com/cocodataset/cocoapi.git
#cd cocoapi/PythonAPI
#python setup.py build_ext install

# install PyTorch Detection
#cd $INSTALL_DIR
#git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
#cd maskrcnn-benchmark


# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0

python setup.py build develop
#unset INSTALL_DIR
# (1)
# you need to create maskrcnn virtual environment first
# (2)
python tools/train_net.py --config-file "configs/defect_detection.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0005 SOLVER.MAX_ITER 60000 SOLVER.STEPS "(30000, 40000)" TEST.IMS_PER_BATCH 1

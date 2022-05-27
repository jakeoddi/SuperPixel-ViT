#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=superpixel


module purge


singularity exec --nv --overlay /scratch/jeo9933/pytorch-example/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "source /ext3/miniconda3/bin/activate;
python -m train"


exit


#!/bin/bash
#SBATCH -J SqueezeNet
#SBATCH --chdir==/shared/home/u124298
#SBATCH -o /shared/home/u124298/slurm.%N.%J.%u.out # STDOUT
#SBATCH -e /shared/home/u124298/slurm.%N.%J.%u.err # STDERR
#SBATCH --exclusive="user"

source /shared/profiles.d/easybuild.sh

ml Cython/0.29.10-foss-2019b-Python-3.6.6
ml torchvision/0.2.1-foss-2019b-Python-3.6.6-PyTorch-1.1.0
ml numpy/1.18.1-foss-2019b-Python-3.6.6
ml OpenCV/3.4.7-foss-2019b-Python-3.6.6
ml matplotlib/3.0.3-foss-2019b-Python-3.6.6
ml PyTorch/1.1.0-foss-2019b-Python-3.6.6-CUDA-9.0.176
ml NLTK/3.5-foss-2019b-Python-3.6.6

python /shared/home/u124298/SqueezeNetLSTM.py

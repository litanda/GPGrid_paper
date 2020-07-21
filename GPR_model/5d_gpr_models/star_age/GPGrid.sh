#!/bin/bash
#SBATCH --ntasks 20
#SBATCH --time 4-23:0:0
#SBATCH --qos daviesgr
#SBATCH --account=daviesgr-cartography
#SBATCH --job-name=gp5d
#SBATCH --mail-type ALL

set -e

module purge; module load bluebear
module load bear-apps/2019b
module load matplotlib
module load SciPy-bundle/2019.10-foss-2019b-Python-3.7.4
pip install --user gpy

python -u Gpy_paper_model_augmentation_5d_mr_process_star_age.py

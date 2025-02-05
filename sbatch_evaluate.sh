#!/bin/bash
#SBATCH --account=rrg-punithak
#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --mail-user=skannan3@ualberta.ca
#SBATCH --mail-type=ALL

#module load gcc/9.3.0 arrow/8 python/3.10.2
module load gcc python arrow
module load opencv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install pip==21.2.4
pip install -r requirements.txt

sh /home/shreya/scratch/Regional/polygon-transformer/run_scripts/evaluation/evaluate_regional.sh
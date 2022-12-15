#!/bin/bash

#SBATCH --job-name=diff_group     # Submit a job named "example"
#SBATCH --partition=a100        # 계산노드 종류 선택: a6000 or a100
#SBATCH --gres=gpu:1             # Use 1 GPU
#SBATCH --time=1-04:30:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=20000              # cpu memory size
#SBATCH --cpus-per-task=8        # cpu 개수
#SBATCH --output=log_mnist_group.txt         # 스크립트 실행 결과 std output을 저장할 파일 이름

ml purge
ml load cuda/11.3            # 필요한 쿠다 버전 로드
eval "$(conda shell.bash hook)"  # Initialize Conda Environment
conda activate test          # Activate your conda environment

# srun jupyter notebook --no-browser --port=9759
srun python train_mnist_group.py
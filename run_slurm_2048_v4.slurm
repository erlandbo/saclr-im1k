#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=5-00:30:00

#SBATCH --nodes=1

#SBATCH --gres=gpu:4  
#SBATCH --mem=192G      
#SBATCH --gpus-per-node=4      
#SBATCH --cpus-per-task=10      
#SBATCH --tasks-per-node=4      

#SBATCH --job-name="2048100ep"

#SBATCH --constraint="h100|a100"
#SBATCH --constraint=gpu80g

module purge
module load Python/3.11.5-GCCcore-13.2.0

source venv/bin/activate
# pip install -r requirements.txt
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121


python main_saclr_multigpu.py --method saclr-all-batchmix-cosine --rho 0.999 --alpha 0.05 --temp 0.15 --s_init_t 0.0 --lr_scale linear --base_lr 0.5 --data_path ~/Datasets/imagenet/ --epochs 1000 --batch_size 128 --savedir test --num_workers 10 --bmix_rate 0.7
#python main_saclr_multigpu.py --method saclr-all-cosine --rho 0.999 --alpha 0.05 --temp 0.15 --s_init_t 0.0 --lr_scale linear --base_lr 0.5 --data_path /cluster/work/erlandbo/imagenet/ --epochs 100 --batch_size 2048 --savedir saclrall_cosine-simclrtuned_2048_ep100 --num_workers 10

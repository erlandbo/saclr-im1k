#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=6-00:30:00

#SBATCH --nodes=1

#SBATCH --gres=gpu:1  
#SBATCH -c10                  # Number of cores (can vary)
#SBATCH --mem=64G      
#SBATCH --job-name="100ep512"

#SBATCH --constraint="h100|a100"
#SBATCH --constraint=gpu80g

module purge
module load Python/3.11.5-GCCcore-13.2.0

source venv/bin/activate
# pip install -r requirements.txt
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121


#python main_saclr.py --method saclr-all-cosine --rho 0.999 --alpha 0.05 --temp 0.15 --s_init_t 0.0 --data_path /cluster/work/erlandbo/imagenet/ --epochs 100 --batch_size 512 --savedir saclrall_cosine-simclrtuned_512_ep100 --num_workers 10
python main_saclr.py --method saclr-all-cosine --rho 0.999 --alpha 0.05 --temp 0.15 --s_init_t 0.0 --lr_scale linear --base_lr 0.5 --data_path /cluster/work/erlandbo/imagenet/ --epochs 100 --batch_size 512 --savedir saclrall_cosine-simclrtuned_512_ep100_lr05 --num_workers 10

#python main_saclr.py --method saclr-1-cosine --temp 0.2 --data_path /cluster/work/erlandbo/imagenet/ --epochs 100 --batch_size 512 --savedir saclr-1-cosine --num_workers 10 --s_init_t -1.0 --rho 0.999 --alpha 0.05 --single_s

#python main_saclr.py --method saclr-all --temp 0.5 --data_path /cluster/work/erlandbo/imagenet/ --epochs 100 --batch_size 512 --savedir saclr-all-ep200rho09999-alpha001 --num_workers 10 --rho 0.9999 --alpha 0.01

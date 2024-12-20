python main_sacl.py --lr_scale linear --arch resnet18 --first_conv --drop_maxpool --optimizer sgd --base_lr 0.03 --weight_decay 5e-4 --single_s --alpha 0.125 --rho 0.99 --s_init_t 2 --temp 0.5 --data_path ./data/ --dataset cifar10 --epochs 1000 --random_state 44
python main_sacl.py --lr_scale linear --arch resnet18 --first_conv --drop_maxpool --optimizer sgd --base_lr 0.03 --weight_decay 5e-4 --no-single_s --alpha 0.125 --rho 0.9 --s_init_t 2 --temp 0.5 --data_path ./data/ --dataset cifar10 --epochs 1000 --random_state 44
python main_sacl.py --lr_scale linear --arch resnet18 --first_conv --drop_maxpool --optimizer sgd --base_lr 0.03 --weight_decay 5e-4 --single_s --alpha 0.125 --rho 0.99 --s_init_t 2 --temp 0.5 --data_path ./data/ --dataset cifar100 --epochs 1000 --random_state 44
python main_sacl.py --lr_scale linear --arch resnet18 --first_conv --drop_maxpool --optimizer sgd --base_lr 0.03 --weight_decay 5e-4 --no-single_s --alpha 0.125 --rho 0.9 --s_init_t 2 --temp 0.5 --data_path ./data/ --dataset cifar100 --epochs 1000 --random_state 44
python main_sacl.py --single_s --alpha 0.125 --rho 0.99 --s_init_t 2 --temp 0.5 --data_path ./data/ --dataset imagenette --epochs 800 --random_state 44 --arch resnet18
python main_sacl.py --no-single_s --alpha 0.125 --rho 0.9 --s_init_t 2 --temp 0.5 --data_path ./data/ --dataset imagenette --epochs 800 --random_state 44 --arch resnet18
\
python main_sacl.py --single_s --alpha 0.125 --rho 0.99 --s_init_t 2 --temp 0.5 --data_path ~/Datasets/imagenet/ --dataset imagenet --epochs 400 --random_state 44
python main_sacl.py --no-single_s --alpha 0.125 --rho 0.9 --s_init_t 2 --temp 0.5 --data_path ~/Datasets/imagenet/ --dataset imagenet --epochs 400 --random_state 44
python main_sacl.py --single_s --alpha 0.125 --rho 0.99 --s_init_t 2 --temp 0.5 --data_path ~/Datasets/imagenet100/ --dataset imagenet100 --epochs 400 --random_state 44
python main_sacl.py --no-single_s --alpha 0.125 --rho 0.9 --s_init_t 2 --temp 0.5 --data_path ~/Datasets/imagenet100/ --dataset imagenet100 --epochs 400 --random_state 44

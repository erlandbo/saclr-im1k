# SACLR

### Prepare imagenet dataset:
    # download ImageNet ILSVRC2012 from ImageNet-website
    # Prepare ImageNet folders, for example with the help of extract_ILSVRC.sh

### Prepare imagenet100 dataset:
    - python make_imagenet100.py full/imagenet/path desired/imagenet100/path 



## Usage SACLR-1 (M=1 negative sample)
```
./run_experiments.sh
```
###
#### ImageNet1k
##### Matrix-method
```
python main_sacl.py --single_s --alpha 0.125 --rho 0.99 --s_init_t 2 --temp 0.5 --data_path ~/Datasets/imagenet/ --dataset imagenet --epochs 400 --random_state 44
```
##### Row-method
```
python main_sacl.py --no-single_s --alpha 0.125 --rho 0.9 --s_init_t 2 --temp 0.5 --data_path ~/Datasets/imagenet/ --dataset imagenet --epochs 400 --random_state 44
```
#### ImageNet100
##### Matrix-method
```
python main_sacl.py --single_s --alpha 0.125 --rho 0.99 --s_init_t 2 --temp 0.5 --data_path ~/Datasets/imagenet100/ --dataset imagenet100 --epochs 400 --random_state 44
```
##### Row-method
```
python main_sacl.py --no-single_s --alpha 0.125 --rho 0.9 --s_init_t 2 --temp 0.5 --data_path ~/Datasets/imagenet100/ --dataset imagenet100 --epochs 400 --random_state 44
```

#### Imagenette
##### Matrix-method
```
python main_sacl.py --single_s --alpha 0.125 --rho 0.99 --s_init_t 2 --temp 0.5 --data_path ./data/ --dataset imagenette --epochs 800 --random_state 44 --arch resnet18
```
##### Row-method
```
python main_sacl.py --no-single_s --alpha 0.125 --rho 0.9 --s_init_t 2 --temp 0.5 --data_path ./data/ --dataset imagenette --epochs 800 --random_state 44 --arch resnet18
```

#### CIFAR
##### Matrix-method
```
python main_sacl.py --lr_scale linear --arch resnet18 --first_conv --drop_maxpool --optimizer sgd --base_lr 0.03 --weight_decay 5e-4 --single_s --alpha 0.125 --rho 0.99 --s_init_t 2 --temp 0.5 --data_path ./data/ --dataset cifar100 --epochs 1000 --random_state 44
```
##### Row-method
```
python main_sacl.py --lr_scale linear --arch resnet18 --first_conv --drop_maxpool --optimizer sgd --base_lr 0.03 --weight_decay 5e-4 --no-single_s --alpha 0.125 --rho 0.9 --s_init_t 2 --temp 0.5 --data_path ./data/ --dataset cifar100 --epochs 1000 --random_state 44
```


## Usage full-batchmode version SACLR-all (more than one negative sample)
##### Matrix-method
include argument --method fullbatchmode
```
python main_sacl.py --single_s --alpha 0.125 --rho 0.99 --s_init_t 2 --temp 0.5 --data_path ~/Datasets/imagenet/ --dataset imagenet --epochs 400 --random_state 44 --method fullbatchmode
```
##### Row-method
```
python main_sacl.py --no-single_s --alpha 0.125 --rho 0.9 --s_init_t 2 --temp 0.5 --data_path ~/Datasets/imagenet/ --dataset imagenet --epochs 400 --random_state 44 --method fullbatchmode
```

## Usage evaluation

#### Linear classifier evaluation ImageNet
```
python eval_linear.py --data_path ~/Datasets/imagenet/ --dataset imagenet --model_checkpoint_path logs/your-folder-name/checkpoint_last.pth --random_state 44
```
#### Linear classifier evaluation Cifar
```
python eval_linear.py --optimizer sgd --lr 30.0 --weight_decay 0.0 --epochs 90 --dataset cifar10 --model_checkpoint_path logs/your-folder-name/checkpoint_last.pth --random_state 44
```
#### kNN classifier evaluation Imagenette
```
python eval_sklearn.py --data_path ./data/ --dataset imagenette --model_checkpoint_path logs/your-folder-name/checkpoint_last.pth --random_state 44
```
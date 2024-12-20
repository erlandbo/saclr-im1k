# SACLR

### Prepare imagenet dataset:
    # download ImageNet ILSVRC2012 from ImageNet-website
    # Prepare ImageNet folders, for example with the help of extract_ILSVRC.sh

### Prepare imagenet100 dataset:
    - python make_imagenet100.py full/imagenet/path desired/imagenet100/path 


## Usage SACLR
###
#### ImageNet1k
##### SACLR
```
python main_saclr.py --method saclr-all --no-single_s --alpha 0.125 --rho 0.99 --s_init_t 2 --temp 0.5 --data_path ~/Datasets/imagenet/ --epochs 100 --batch_size 512 --savedir saclrall
```
##### SimCLR
```
python main_saclr.py --method simclr --temp 0.5 --data_path ~/Datasets/imagenet/ --epochs 100 --batch_size 512 --savedir simclr
```

## Usage evaluation

#### Linear classifier evaluation ImageNet
```
python evaluate_linear.py --data_path ~/Datasets/imagenet/ --backbone_path logs/resnet50_last.pth --savedir linear
```

#### finetune evaluation ImageNet
```
python evaluate_finetune.py --data_path ~/Datasets/imagenet/ --backbone_path logs/resnet50_last.pth --train-percent 10 --savedir linear
```
#### kNN evaluation ImageNet
```
python evaluate_nn.py --data_path ~/Datasets/imagenet/ --backbone_path logs/resnet50_last.pth --savedir linear
```

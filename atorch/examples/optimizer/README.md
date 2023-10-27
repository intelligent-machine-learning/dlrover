# A demo to using AGD and WSAM Optimizers

## Usage
```
python main.py [--use-gpu] [--dataset DataSet] [--model Model] [--batch-size BS] [--epochs Epochs] [--scheduler Scheduler] [--base_optimizer Base] [--lr LR] [--weight_decay WD] [--optimizer Optimizer] [--mode Mode] [--rho Rho] [...]
```

- Supported dataset: cifar10 & cifar100
- Supported model: Resnet18, Resnet34, Resnet50
- more parameters can be found in [main.py](./main.py)

## Example
Before running experiments, please set the environment variable:
```
export CUDA_VISIBLE_DEVICES=0
``` 
Train Resnet18 on Cifar10 using AGD optimizer:
```
python main.py --use-gpu --dataset cifar10 --model resnet18 --batch-size 128 --epochs 200 --scheduler cosine --base_optimizer agd --lr 0.001 --eps 1e-8 --weight-decay 5e-4
```

Train Resnet18 on Cifar10 using WSAM optimizer with sgd as the base optimizer:
```
python main.py --use-gpu --dataset cifar10 --model resnet18 --batch-size 128 --epochs 200 --scheduler cosine --base_optimizer sgd --lr 0.1 --weight-decay 5e-4 --optimizer wsam --mode decouple --rho 0.2 --gamma 0.9
```

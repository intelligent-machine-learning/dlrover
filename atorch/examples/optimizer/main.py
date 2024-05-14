from __future__ import print_function

import argparse
import logging
import random
import ssl
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from model import ResNet18, ResNet34, ResNet50
from torch import nn, optim
from utils import getData, test, test_cpu  # type: ignore[attr-defined]

from atorch.optimizers import AGD, WeightedSAM

ssl._create_default_https_context = ssl._create_unverified_context


# Training settings
parser = argparse.ArgumentParser(description="PyTorch Example")
parser.add_argument("--dataset", type=str, default="cifar10", help="choose dataset (options: cifar10, cifar100)")
parser.add_argument(
    "--label-smoothing", type=float, default=0.0, dest="label_smoothing", help="label smoothing rate (default: 0.0)"
)
parser.add_argument(
    "--batch-size", type=int, default=256, metavar="B", help="input batch size for training (default: 256)"
)
parser.add_argument(
    "--test-batch-size", type=int, default=256, metavar="TB", help="input batch size for testing (default: 256)"
)
parser.add_argument("--epochs", type=int, default=200, metavar="E", help="number of epochs to train (default: 200)")
parser.add_argument("--lr", type=float, default=0.15, metavar="LR", help="learning rate (default: 0.15)")
parser.add_argument("--lr-min", type=float, default=0.0, dest="lr_min", help="minimum learning rate (default: 0.0)")
parser.add_argument("--lr-decay", type=float, default=0.1, help="learning rate ratio")
parser.add_argument(
    "--lr-decay-epoch", type=int, nargs="+", default=[80, 120], help="decrease learning rate at these epochs."
)
parser.add_argument("--scheduler", type=str, default="cosine", help="choose scheduler")
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
parser.add_argument(
    "--weight-decay", "--wd", default=5e-4, type=float, metavar="W", help="weight decay (default: 5e-4)"
)
parser.add_argument("--weight-decouple", action="store_true", help="weight decouple")
parser.add_argument("--model", type=str, default="resnet18", help="choose model")
parser.add_argument("--depth", type=int, default=20, help="choose the depth of resnet")
parser.add_argument("--optimizer", type=str, default="vanilla", help="choose optim")
parser.add_argument("--base_optimizer", type=str, default="sgd", help="choose base optim")
parser.add_argument("--eps", type=float, default=1e-8, help="choose epsilon")
parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive WSAM.")
parser.add_argument("--rho", type=float, default=0.05, help="rho parameter for SAM.")
parser.add_argument("--gamma", type=float, default=0.5, help="gamma parameter for WSAM.")
parser.add_argument("--mode", type=str, default="decouple", help="choose mode for wsam, (couple, decouple)")
parser.add_argument("--log_file", type=str, default=None, help="log file")
parser.add_argument("--use-gpu", action="store_true", help="whether to use gpu")
parser.add_argument("--pin-memory", type=bool, default=True, help="whether to use pin_memory")
parser.add_argument("--save-ckpt", action="store_true", help="whether to save ckpt")
args = parser.parse_args()

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=args.log_file, level=logging.INFO, format=LOG_FORMAT)
logging.captureWarnings(True)

# set random seed to reproduce the work
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False


def main():
    for arg in vars(args):
        logging.info("{}: {}".format(arg, getattr(args, arg)))
    # if not os.path.isdir('checkpoint/'):
    #     os.makedirs('checkpoint/')
    # get dataset
    train_loader, test_loader = getData(
        name=args.dataset, train_bs=args.batch_size, test_bs=args.test_batch_size, pin_memory=args.pin_memory
    )

    # make sure to use cudnn.benchmark for second backprop
    cudnn.benchmark = True

    # get model and optimizer
    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    elif args.dataset == "oxford-iiit-pet":
        num_classes = 37
    else:
        raise Exception(f"dataset {args.dataset} not supported yet.")

    if args.model == "resnet18":
        model = ResNet18(num_classes=num_classes)
    elif args.model == "resnet34":
        model = ResNet34(num_classes=num_classes)
    elif args.model == "resnet50":
        model = ResNet50(num_classes=num_classes)
    else:
        raise Exception(f"model {args.model} not supported yet.")

    if args.use_gpu:
        model = model.cuda()
    logging.info(model)
    logging.info("    Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.base_optimizer == "sgd":
        base_optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, nesterov=False, weight_decay=args.weight_decay
        )
    elif args.base_optimizer == "adam":
        base_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.base_optimizer == "adamw":
        logging.info(
            "For AdamW, we automatically correct the weight decay term for you!"
            "If this is not what you want, please modify the code!"
        )
        args.weight_decay = args.weight_decay / args.lr
        base_optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.base_optimizer == "agd":
        logging.info(
            "For AGD, we automatically correct the weight decay term for you!"
            "If this is not what you want, please modify the code!"
        )
        args.weight_decay = args.weight_decay / args.lr
        base_optimizer = AGD(
            model.parameters(),
            lr=args.lr,
            delta=args.eps,
            weight_decay=args.weight_decay,
        )
    else:
        raise Exception("We do not support this optimizer yet!!")

    # learning rate schedule
    if args.scheduler == "multistep":
        scheduler = lr_scheduler.MultiStepLR(base_optimizer, args.lr_decay_epoch, gamma=args.lr_decay, last_epoch=-1)
    elif args.scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            base_optimizer,
            T_max=args.epochs * len(train_loader),
            eta_min=args.lr_min,
            last_epoch=-1,
        )
    else:
        raise Exception("We do not support this scheduler yet!!")

    if args.optimizer == "vanilla":
        logging.info("Using optimizer 'base_optimizer'")
        optimizer = base_optimizer
    elif args.optimizer == "wsam":
        logging.info("Using optimizer WSAM")
        decouple = True if args.mode == "decouple" else False
        optimizer = WeightedSAM(
            model, base_optimizer, rho=args.rho, gamma=args.gamma, adaptive=args.adaptive, decouple=decouple
        )
    else:
        raise Exception("We do not support this optimizer yet!!")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        logging.info("Current Epoch: %d", epoch)

        # train for one epoch
        if args.optimizer == "vanilla":
            train_loss = train(train_loader, model, criterion, optimizer, scheduler, args)
        elif args.optimizer == "wsam":
            train_loss = train_wsam(train_loader, model, criterion, optimizer, scheduler, args)
        else:
            raise Exception(f"invalid optimizer name {args.optimizer}")

        logging.info("Training Loss of Epoch {}: {}".format(epoch, train_loss))

        if args.use_gpu:
            acc = test(model, test_loader)
        else:
            acc = test_cpu(model, test_loader)
        logging.info("Testing of Epoch {}: {} \n".format(epoch, acc))

        if acc > best_acc:
            best_acc = acc
            if args.save_ckpt:
                ckpt_path = "./ckpt.pkl"
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": base_optimizer.state_dict(),
                        "best_accuracy": best_acc,
                    },
                    ckpt_path,
                )

    logging.info("Best Acc: {}\n".format(best_acc))
    logging.shutdown()


def train(train_loader, model, criterion, optimizer, scheduler, args):
    starttime = time.time()
    train_loss = 0.0
    total_num = 0
    correct = 0

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.use_gpu:
            data, target = data.cuda(non_blocking=args.pin_memory), target.cuda(non_blocking=args.pin_memory)
        output = model(data)
        loss = criterion(output, target)
        if args.base_optimizer == "adahessian":
            loss.backward(create_graph=True)
        else:
            loss.backward()
        train_loss += loss.item() * target.size()[0]
        total_num += target.size()[0]
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        optimizer.step()
        optimizer.zero_grad()
        if args.scheduler == "cosine":
            scheduler.step()

    if args.scheduler == "multistep":
        scheduler.step()

    endtime = time.time()
    logging.info("cost: {}".format(endtime - starttime))
    train_loss /= total_num

    return train_loss


def train_wsam(train_loader, model, criterion, optimizer, scheduler, args):
    starttime = time.time()
    train_loss = 0.0
    total_num = 0

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.use_gpu:
            data, target = data.cuda(non_blocking=args.pin_memory), target.cuda(non_blocking=args.pin_memory)

        def closure():
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        optimizer.zero_grad()

        train_loss += loss.item() * target.size()[0]
        total_num += target.size()[0]

        if args.scheduler == "cosine":
            scheduler.step()

    if args.scheduler == "multistep":
        scheduler.step()

    endtime = time.time()
    logging.info("cost: {}".format(endtime - starttime))
    train_loss /= total_num

    return train_loss


if __name__ == "__main__":
    main()

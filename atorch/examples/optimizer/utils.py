import torch
from torchvision import datasets, transforms


def getData(name="cifar10", train_bs=128, test_bs=128, pin_memory=True):

    if name == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "./data",
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            ),
            batch_size=train_bs,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "./data",
                train=False,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            ),
            batch_size=test_bs,
            shuffle=False,
        )

    if name == "cifar10":
        transform_train_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
        ]

        transform_train = transforms.Compose(transform_train_list)

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
            ]
        )

        trainset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform_train,
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=pin_memory
        )

        testset = datasets.CIFAR10(
            root="./data",
            train=False,
            download=False,
            transform=transform_test,
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=pin_memory
        )

    if name == "cifar100":
        transform_train_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
        ]

        transform_train = transforms.Compose(transform_train_list)

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
            ]
        )

        trainset = datasets.CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=transform_train,
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=pin_memory
        )

        testset = datasets.CIFAR100(
            root="./data",
            train=False,
            download=False,
            transform=transform_test,
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=pin_memory
        )

    if name == "svhn":
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                "./data", split="train", download=True, transform=transforms.Compose([transforms.ToTensor()])
            ),
            batch_size=train_bs,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN("./data", split="test", download=True, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=test_bs,
            shuffle=False,
        )

    if name == "oxford-iiit-pet":
        transform_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        trainset = datasets.OxfordIIITPet(root="data", transform=transform_train, download=True, split="trainval")
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=train_bs, shuffle=True, num_workers=8, pin_memory=pin_memory
        )
        testset = datasets.OxfordIIITPet(root="data", transform=transform_test, download=False, split="test")
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_bs, shuffle=False, num_workers=8, pin_memory=pin_memory
        )

    return train_loader, test_loader


def test(model, test_loader):
    # print('Testing')
    model.eval()
    correct = 0
    total_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_num += len(data)
    # print('testing_correct: ', correct / total_num, '\n')
    return correct / total_num


def test_cpu(model, test_loader):
    # print('Testing')
    model.eval()
    correct = 0
    total_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data, target = data.cuda(), target.cuda()
            data, target = data, target
            output = model(data)
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_num += len(data)
    # print('testing_correct: ', correct / total_num, '\n')
    return correct / total_num

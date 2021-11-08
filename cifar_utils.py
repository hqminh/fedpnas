from utils import *


def load_cifar_transfer(deg=0):
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=(deg, deg)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    trainset = torchvision.datasets.CIFAR10(root=f'./data/', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=f'./data/', train=False, download=True, transform=transform)
    n_test = testset.data.shape[0]
    Xs, Ys = get_data_slice(trainset, 0, 1000)
    Xt, Yt = get_data_slice(testset, 0, n_test)
    return Xs, torch.LongTensor(Ys), Xt, torch.LongTensor(Yt), classes


def load_cifar_meta(n_client, normalize=True):
    degrees = [-30, -15, 0, 15, 30]
    spec_xform = [
        transforms.RandomRotation(degrees=(d, d)) for d in degrees
    ]
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    Xs, Ys, Xt, Yt = [], [], [], []
    for i in range(n_client):
        if normalize:
            transform = transforms.Compose([
                spec_xform[i],
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transforms.Compose([
                spec_xform[i],
                transforms.ToTensor()
            ])
        trainset = torchvision.datasets.CIFAR10(root=f'./data/client-{i+1}/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=f'./data/client-{i+1}/', train=False, download=True, transform=transform)
        print(f'Data transformed for Client {i} ({degrees[i]} degree rotation)')
        n_data = trainset.data.shape[0]
        n_test = testset.data.shape[0]
        Xi, Yi = get_data_slice(trainset, i * (n_data // n_client), (i + 1) * (n_data // n_client))
        Xti, Yti = get_data_slice(testset, i * (n_test // n_client), (i + 1) * (n_test // n_client))
        Xs.append(Xi)
        Ys.append(Yi)
        Xt.append(Xti)
        Yt.append(Yti)
    Xs, Ys = torch.stack(Xs), torch.LongTensor(Ys)
    Xt, Yt = torch.stack(Xt), torch.LongTensor(Yt)
    return Xs, Ys, Xt, Yt, classes

def load_cifar_std(n_client, normalize=True):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    Xs, Ys = [], []
    n_data = trainset.data.shape[0]
    n_test = testset.data.shape[0]
    for i in range(n_client):
        Xi, Yi = get_data_slice(trainset, i * (n_data // n_client), (i + 1) * (n_data // n_client))
        Xs.append(Xi)
        Ys.append(Yi)

    Xs, Ys = torch.stack(Xs), torch.LongTensor(Ys)
    Xt, Yt = get_data_slice(testset, 0, n_test)
    Yt = torch.LongTensor(Yt)
    return Xs, Ys, Xt, Yt, classes


def load_cifar(n_client, meta_task=False):
    if meta_task:
        return load_cifar_meta(n_client)
    else:
        return load_cifar_std(n_client)
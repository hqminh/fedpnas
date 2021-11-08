from utils import *


def load_mnist_transfer(deg=0):
    transform = transforms.Compose([
        transforms.Pad((2, 2, 2, 2)),
        transforms.RandomRotation(degrees=(deg, deg)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    trainset = torchvision.datasets.MNIST(root=f'./data/', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=f'./data/', train=False, download=True, transform=transform)
    n_test = testset.data.shape[0]
    Xs, Ys = get_data_slice(trainset, 0, 1000)
    Xt, Yt = get_data_slice(testset, 0, n_test)
    return Xs, torch.LongTensor(Ys), Xt, torch.LongTensor(Yt), classes


def load_mnist_meta(n_client, mode='degree'):
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    if mode == 'degree':
        degrees = [-30, -15, 0, 15, 30]
        spec_xform = [
            transforms.RandomRotation(degrees=(d, d)) for d in degrees
        ]
        desc = [f'RandomRotation Deg={d}' for d in degrees]
    else:
        spec_xform = [
            torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=(-0.5, -0.5)),
            torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=(0.5, 0.5)),
            transforms.RandomRotation(degrees=(0, 0)),
            transforms.RandomRotation(degrees=(-90, -90)),
            transforms.RandomRotation(degrees=(90, 90)),
        ]
        desc = [
            'ColorJitter Hue=-0.5',
            'ColorJitter Hue=0.5',
            'RandomRotation Deg=0',
            'RandomRotation Deg=-90',
            'RandomRotation Deg=90',
        ]

    Xs, Ys, Xt, Yt = [], [], [], []
    for i in range(n_client):
        transform = transforms.Compose([
            transforms.Pad((2, 2, 2, 2)),
            spec_xform[i],
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,))
        ])
        trainset = torchvision.datasets.MNIST(root=f'./data/client-{i+1}/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=f'./data/client-{i+1}/', train=False, download=True, transform=transform)
        print(f'Data transformed for Client {i} ({desc[i]})')
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


def load_mnist_std(n_client):
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    transform = transforms.Compose([
        transforms.Pad((2, 2, 2, 2)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))]
    )
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
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


def load_mnist(n_client, meta_task=False, mode='mixed'):
    if meta_task:
        return load_mnist_meta(n_client, mode=mode)
    else:
        return load_mnist_std(n_client)
import torchvision

DATA_ROOT = r'C:\Users\user\Desktop\f'
fashion_mnist = torchvision.datasets.FashionMNIST(download=True, train=True, root=DATA_ROOT).train_data.float()


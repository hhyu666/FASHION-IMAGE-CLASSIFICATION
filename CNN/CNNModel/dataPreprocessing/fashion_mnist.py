import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATA_ROOT = r'../../Dataset'
class Fashion_MNIST:
    def __init__(self, batch_size, threads):
        fashion_mnist = torchvision.datasets.FashionMNIST(download=False, train=True, root=DATA_ROOT).train_data.float()
        train_set = torchvision.datasets.FashionMNIST(
            root=DATA_ROOT,
            download=True,
            train=True,
            transform=transforms.Compose([
                transforms.Resize(size=(32, 32)),
                transforms.Pad(padding=4),
                transforms.RandomAdjustSharpness(1),
                transforms.RandomAdjustSharpness(2),
                transforms.RandomHorizontalFlip(p=.75),
                transforms.RandomRotation(degrees=13),
                # 测试也要如下预处理（怎么训练怎么考试）
                transforms.ToTensor(),
                transforms.Normalize(
                    (fashion_mnist.mean() / 255,),
                    (fashion_mnist.std() / 255,)
                )
            ]))

        test_set = torchvision.datasets.FashionMNIST(
            root=DATA_ROOT,
            download=True,
            train=False,
            transform=transforms.Compose([
                transforms.Resize(size=(32, 32)),
                transforms.Pad(padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (fashion_mnist.mean() / 255,),
                    (fashion_mnist.std() / 255,)
                )
            ]))

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)
        self.classes = ("T - shirt / top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                        "Ankle boot")
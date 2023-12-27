from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np


ds = datasets.MNIST(
    root='data',
    train=True,
    download=True,

    transform=lambda img: np.array(np.asarray(img).flatten())/256,
    target_transform=lambda x: np.array(
        [1 if i == x else 0 for i in range(10)])
)
ds1 = Subset(ds, range(0, 1000))
dl1 = DataLoader(ds1, shuffle=True, batch_size=None)

ds2 = Subset(ds, range(1000, 1000+100))
dl2 = DataLoader(ds2, shuffle=True, batch_size=None)
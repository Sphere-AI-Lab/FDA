import sys
import inspect
import random
import torch
import copy

from torch.utils.data.dataset import random_split

from src.datasets.cars import Cars
from src.datasets.cifar10 import CIFAR10
from src.datasets.cifar100 import CIFAR100
from src.datasets.dtd import DTD
from src.datasets.eurosat import EuroSAT, EuroSATVal
from src.datasets.gtsrb import GTSRB
from src.datasets.imagenet import ImageNet
from src.datasets.mnist import MNIST
from src.datasets.resisc45 import RESISC45
from src.datasets.stl10 import STL10
from src.datasets.svhn import SVHN
from src.datasets.sun397 import SUN397

registry = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}


# {'CIFAR10': <class 'src.datasets.cifar10.CIFAR10'>, 'CIFAR100': <class 'src.datasets.cifar100.CIFAR100'>, 'Cars': <class 'src.datasets.cars.Cars'>, 'DTD': <class 'src.datasets.dtd.DTD'>, 'EuroSAT': <class 'src.datasets.eurosat.EuroSAT'>, 'EuroSATVal': <class 'src.datasets.eurosat.EuroSATVal'>, 'GTSRB': <class 'src.datasets.gtsrb.GTSRB'>, 'ImageNet': <class 'src.datasets.imagenet.ImageNet'>, 'MNIST': <class 'src.datasets.mnist.MNIST'>, 'RESISC45': <class 'src.datasets.resisc45.RESISC45'>, 'STL10': <class 'src.datasets.stl10.STL10'>, 'SUN397': <class 'src.datasets.sun397.SUN397'>, 'SVHN': <class 'src.datasets.svhn.SVHN'>}


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def split_train_into_train_val(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, val_shot, seed=0):
    total_size = len(dataset.train_dataset)
    if val_fraction:
        assert val_shot is None
        val_size = int(total_size * val_fraction)
    elif val_shot:
        assert val_fraction is None
        val_size = min(val_shot, total_size)
    else:
        val_size = 0

    train_size = total_size - val_size
    lengths = [train_size, val_size]
    # 使用pytorch内置函数，进行随机划分验证集
    trainset, valset = random_split(
        dataset.train_dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed)
    )

    new_dataset_class = type(new_dataset_class_name, (GenericDataset,), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.val_dataset = valset
    new_dataset.val_loader = torch.utils.data.DataLoader(
        new_dataset.val_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.test_dataset = dataset.test_dataset
    new_dataset.test_loader = dataset.test_loader

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def split_test_into_val_test(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, val_shot, seed=0):
    total_size = len(dataset.test_dataset)
    if val_fraction:
        assert val_shot is None
        val_size = int(total_size * val_fraction)
    elif val_shot:
        assert val_fraction is None
        val_size = min(val_shot, total_size)
    else:
        val_size = 0

    test_size = total_size - val_size
    lengths = [val_size, test_size]

    valset, testset = random_split(
        dataset.test_dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed)
    )

    new_dataset_class = type(new_dataset_class_name, (GenericDataset,), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = dataset.train_dataset
    new_dataset.train_loader = dataset.train_loader

    new_dataset.test_dataset = testset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.val_dataset = valset
    new_dataset.val_loader = torch.utils.data.DataLoader(
        new_dataset.val_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def get_dataset(dataset_name, preprocess, location, batch_size=128, num_workers=16, val_fraction=None,
                val_shot=None, seed=0):
    if dataset_name.endswith('ValfromTrain'):
        base_dataset_name = dataset_name.split('ValfromTrain')[0]
        base_dataset = get_dataset(base_dataset_name, preprocess, location, batch_size, num_workers)
        dataset = split_train_into_train_val(
            base_dataset, dataset_name, batch_size, num_workers, val_fraction, val_shot, seed)
        return dataset
    elif dataset_name.endswith('ValfromTest'):
        base_dataset_name = dataset_name.split('ValfromTest')[0]
        base_dataset = get_dataset(base_dataset_name, preprocess, location, batch_size, num_workers)
        dataset = split_test_into_val_test(
            base_dataset, dataset_name, batch_size, num_workers, val_fraction, val_shot, seed)
        return dataset
    else:
        assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
        dataset_class = registry[dataset_name]
        dataset = dataset_class(
            preprocess, location=location, batch_size=batch_size, num_workers=num_workers
        )
        return dataset
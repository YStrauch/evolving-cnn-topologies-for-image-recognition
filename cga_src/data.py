'''Data Loaders'''

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torchvision.utils import make_grid
import torchvision.transforms as transforms


class TorchDataset(ABC):
    '''Abstract Base dataset'''

    def __init__(self,
                 batch_size_train=100,
                 batch_size_test=500,
                 shuffle_train=True,
                 transf=None
                 ):
        self._batch_size_train = batch_size_train
        self._batch_size_test = batch_size_test
        self._shuffle_train = shuffle_train
        self._transforms = transf or transforms.ToTensor()

        self._load()

    @property
    def train(self):
        '''The train data set'''
        return self._train

    @property
    def test(self):
        '''The test data set'''
        return self._test

    @property
    def example(self):
        '''An example of the data'''
        return self._example

    @property
    @abstractmethod
    def num_classes(self):
        '''The output shape of labels'''

    @property
    @abstractmethod
    def _pytorch_dataset(self):
        '''To be implemented by the concrete class'''

    def _load(self):
        '''
            Downloads and loads the data set into a pytorch tensor
        '''
        train_dataset = self._pytorch_dataset(root='./cga_src/download',
                                              train=True,
                                              transform=self._transforms,
                                              download=True
                                              )

        test_dataset = self._pytorch_dataset(root='./cga_src/download',
                                             train=False,
                                             transform=self._transforms,
                                             download=True
                                             )

        # Data loader
        self._train = torch.utils.data.DataLoader(dataset=train_dataset,
                                                  batch_size=self._batch_size_train,
                                                  shuffle=self._shuffle_train)

        self._test = torch.utils.data.DataLoader(dataset=test_dataset,
                                                 batch_size=self._batch_size_test,
                                                 shuffle=False)

        self._example = train_dataset[0][0]
        assert isinstance(self._example, torch.Tensor)
        return self

    def show_samples(self, N=50):
        '''Visualises the first N training data'''
        dataiter = iter(self.train)
        images, _ = dataiter.next()

        grid = make_grid(
            images[:N], nrow=int(np.ceil(np.sqrt(N))),
            padding=0
        )

        figure, axis = plt.subplots(figsize=(15, 15))
        axis.imshow(grid.numpy().transpose((1, 2, 0)))
        axis.axis('off')
        figure.tight_layout()
        plt.show()

    @staticmethod
    def imshow(img):
        npimg = img.numpy().transpose((1, 2, 0))
        if npimg.shape[2] == 1:
            # monochrome image, remove color channel
            npimg = npimg.reshape(npimg.shape[0], npimg.shape[1])
        plt.axis('off')
        plt.imshow(npimg)
        plt.show()


class MNIST(TorchDataset):
    '''MNIST data set'''

    def __init__(self,
                 batch_size_train=100,
                 batch_size_test=500,
                 shuffle_train=True,
                 transf=None
                 ):

        transf = transf or transforms.Compose([
            transforms.RandomRotation(10, expand=True),
            transforms.RandomCrop(
                size=(32, 32),
                pad_if_needed=True
            ),
            transforms.ToTensor(),
        ])

        super(MNIST, self).__init__(
            batch_size_train=batch_size_train,
            batch_size_test=batch_size_test,
            shuffle_train=shuffle_train,
            transf=transf,
        )

    @property
    def _pytorch_dataset(self):
        return torchvision.datasets.MNIST

    @property
    def num_classes(self):
        return 10


class CIFAR10(TorchDataset):
    '''CIFAR10 data set'''

    def __init__(self,
                 batch_size_train=50,
                 batch_size_test=250,
                 shuffle_train=True,
                 transf=None
                 ):

        transf = transf or transforms.Compose([
            # transforms.Pad(4),
            transforms.RandomCrop(
                size=(32, 32),
                padding=4
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        super(CIFAR10, self).__init__(
            batch_size_train=batch_size_train,
            batch_size_test=batch_size_test,
            shuffle_train=shuffle_train,
            transf=transf,
        )

    @property
    def _pytorch_dataset(self):
        return torchvision.datasets.CIFAR10

    @property
    def num_classes(self):
        return 10


if __name__ == "__main__":

    data = MNIST()
    print(data.example.shape)

    # TorchDataset.imshow(data.example)
    data.show_samples(100)

'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import os
import gzip
import numpy as np
import torch as th, torch.utils.data
import pickle
from PIL import Image
from torch.utils.data import Dataset
import torchvision as V
try:
    import utils
except ModuleNotFoundError as e:
    from . import utils

TRANSFORM_TRAIN = V.transforms.Compose([
    V.transforms.RandomCrop(32, padding=4),
    V.transforms.RandomHorizontalFlip(),
    V.transforms.ToTensor(),
    V.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

TRANSFORM_TEST = V.transforms.Compose([
    V.transforms.ToTensor(),
    V.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

# https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


class Cifar10Dataset(Dataset):
    '''
    the cifar 10 dataset
    '''
    def __init__(self, path, kind='train', transform=None):
        self.path = path
        self.transform = transform
        #
        files_train = [f'data_batch_{x}' for x in range(1,5+1)]
        files_train = [os.path.join(path, x) for x in files_train]
        file_test = os.path.join(path, 'test_batch')
        file_meta = os.path.join(path, 'batches.meta')
        #
        images, labels = [], []
        self.meta = unpickle(file_meta)
        if kind == 'train':
            for i in files_train:
                data = unpickle(i)
                images.append(data['data'])
                labels.extend(data['labels'])
            images = np.vstack(images).reshape(-1, 3, 32, 32)
            labels = np.array(labels)
        elif kind == 'test':
            data = unpickle(file_test)
            images = np.array(data['data']).reshape(-1, 3, 32, 32)
            labels = np.array(data['labels'])
        else:
            raise ValueError('unknown kind')
        self.images = images.transpose((0,2,3,1))
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label



def get_dataset(path: str, kind='train'):
    """
    Load cifar10 data from `path`
    """
    files_train = [f'data_batch_{x}' for x in range(1,5+1)]
    files_train = [os.path.join(path, x) for x in files_train]
    file_test = os.path.join(path, 'test_batch')
    file_meta = os.path.join(path, 'batches.meta')
    images, labels = [], []
    if kind == 'train':
        for i in files_train:
            data = unpickle(i)
            images.append(data['data'])
            labels.extend(data['labels'])
        images = np.vstack(images).reshape(-1, 3, 32, 32)
        labels = np.array(labels)
    else:
        data = unpickle(file_test)
        images = np.array(data['data']).reshape(-1, 3, 32, 32)
        labels = np.array(data['labels'])
    return images, labels


def get_loader(path: str, batchsize: int, kind='train', ddp:bool = False):
    """
    Load cifar10 data and turn them into dataloaders
    """
    if kind == 'train':
        #x_train, y_train = get_dataset(path, kind='train')
        #x_train = utils.renorm(th.from_numpy(x_train).float() / 255.)
        #y_train = th.from_numpy(y_train).long().view(-1, 1)
        #data_train = th.utils.data.TensorDataset(x_train, y_train)
        arc_data_split = os.getenv('ARC_DATA_SPLIT', 'test')
        assert arc_data_split in ('train', 'test')
        if arc_data_split == 'train':
            transform = TRANSFORM_TEST
            shuffle = False
        else:
            transform = TRANSFORM_TRAIN
            shuffle = True
        data_train = Cifar10Dataset(path, kind='train', transform=transform)
        if not ddp:
            loader_train = th.utils.data.DataLoader(data_train,
                batch_size=batchsize, shuffle=shuffle, pin_memory=True, num_workers=4)
        else:
            sampler = th.utils.data.distributed.DistributedSampler(data_train)
            loader_train = th.utils.data.DataLoader(data_train,
                batch_size=batchsize, shuffle=shuffle, pin_memory=True, num_workers=4,
                sampler=sampler)
        return loader_train
    else:
        #x_test, y_test = get_dataset(path, kind='test')
        #x_test = utils.renorm(th.from_numpy(x_test).float() / 255.)
        #y_test = th.from_numpy(y_test).long().view(-1, 1)
        #data_test = th.utils.data.TensorDataset(x_test, y_test)
        transform = TRANSFORM_TEST
        data_test = Cifar10Dataset(path, kind='test', transform=transform)
        loader_test = th.utils.data.DataLoader(data_test,
                batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)
        return loader_test


'''
run some tests
'''
if __name__ == '__main__':
    import pylab as lab
    from tqdm import tqdm
    x_train, y_train = get_dataset(os.path.join(os.getenv('HOME'), '.torch/cifar-10-batches-py'), kind='train')
    print('Train set shape', x_train.shape, y_train.shape)
    x_test, y_test = get_dataset(os.path.join(os.getenv('HOME'), '.torch/cifar-10-batches-py'), kind='test')
    print('Test  set shape', x_test.shape,  y_test.shape)
    print(x_test[0])

    meta = unpickle(os.path.join(os.getenv('HOME'), '.torch/cifar-10-batches-py/batches.meta'))
    print(meta)
    label_names = meta['label_names']

    print('testing loaders')
    train_loader = get_loader(os.path.join(os.getenv('HOME'), '.torch/cifar-10-batches-py/'), 128, kind='train')
    for (images, labels) in tqdm(train_loader):
        nums = images.shape[0]
        assert(images.shape == th.Size([nums, 3, 32, 32]))
        # print(images[0]) normed

    while True:
        idx = np.random.randint(0, x_train.shape[0])
        lab.figure()
        lab.imshow(utils.chw2hwc(x_train[idx]))
        title = 'idx(' + str(idx) + ') class ' + str(y_train[idx]) + ' name ' + label_names[y_train[idx]]
        lab.title(title)
        lab.show()

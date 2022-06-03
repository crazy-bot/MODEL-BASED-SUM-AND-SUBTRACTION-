import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
import numpy as np

def get_weights_MNIST(fpath):
    sum_weights = np.load(fpath)
    return sum_weights


def get_MNIST(data_dir, batch_size, num_workers):
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

    train_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    m = len(train_dataset)

    train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader

class MNISTPair:

    def __init__(self, root, is_train=True):
        np.random.seed(100)
        root = root + '/mnist_train.csv' if is_train else root+'/mnist_test.csv'
        xy = np.loadtxt(root, delimiter=',', dtype=np.float32)
        np.random.shuffle(xy)

        self.len = xy.shape[0]
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, 0]

        labels = np.arange(19)
        
        self.labelmap = {k:v for v,k in enumerate(labels)}

        if is_train:
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])


    def __getitem__(self, index):
        d1 = self.x_data[index]
        d2 = self.x_data[index+1]

        l1 = self.y_data[index]
        l2 = self.y_data[index+1]

        label = torch.LongTensor([self.labelmap[l1+l2]])[0]

        #import pdb; pdb.set_trace()

        d1 = self.transform(d1.reshape(28,28,1))
        d2 = self.transform(d2.reshape(28,28,1))

        return (d1,d2,label)

    def __len__(self):
        return self.len-1


class MNISTPairv2:

    def __init__(self, root):
        np.random.seed(100)
        root = root + '/mnist_train.csv'
        xy = np.loadtxt(root, delimiter=',', dtype=np.float32)
        np.random.shuffle(xy)

        self.len = xy.shape[0]
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, 0]
    
        labels = np.arange(-9, 19)

        self.labelmap = {k:v for v,k in enumerate(labels)}
        self.idxtolabel = {v:k for k,v in self.labelmap.items()}

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        

    def __getitem__(self, index):
        d1 = self.x_data[index]
        d2 = self.x_data[index+1]

        l1 = self.y_data[index]
        l2 = self.y_data[index+1]

        p = np.random.random()

        if p > .5:
            label = torch.LongTensor([self.labelmap[l1+l2]])[0]
            flag = torch.FloatTensor([1])
            self.p = 0
        else:
            label = torch.LongTensor([self.labelmap[l1-l2]])[0]
            flag = torch.FloatTensor([0])
            self.p = 1

        #import pdb; pdb.set_trace()

        d1 = self.transform(d1.reshape(28,28,1))
        d2 = self.transform(d2.reshape(28,28,1))

        return (d1, d2, flag, label, l1, l2)

    def __len__(self):
        return self.len-1


class MNISTPairv2Test:

    def __init__(self, root, issum):
        np.random.seed(100)
        root = root +'/mnist_test.csv'
        xy = np.loadtxt(root, delimiter=',', dtype=np.float32)
        np.random.shuffle(xy)

        self.len = xy.shape[0]
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, 0]
    
        labels = np.arange(-9, 19)

        self.labelmap = {k:v for v,k in enumerate(labels)}
        self.idxtolabel = {v:k for k,v in self.labelmap.items()}

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.issum = issum

    def __getitem__(self, index):
        d1 = self.x_data[index]
        d2 = self.x_data[index+1]

        l1 = self.y_data[index]
        l2 = self.y_data[index+1]

        if self.issum:
            label = torch.LongTensor([self.labelmap[l1+l2]])[0]
            flag = torch.FloatTensor([1]) 
        else:
            label = torch.LongTensor([self.labelmap[l1-l2]])[0]
            flag = torch.FloatTensor([0])

        #import pdb; pdb.set_trace()

        d1 = self.transform(d1.reshape(28,28,1))
        d2 = self.transform(d2.reshape(28,28,1))

        return (d1, d2, flag, label, l1, l2)

    def __len__(self):
        return self.len-1
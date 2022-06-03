
'''
This file is the starting point of training a classifier 
of gender or hair color classificazion individually
'''

import sys
import os
from time import time, ctime
import torch
from torch.utils.data import DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import recall_score, precision_score, classification_report

import warnings

from utils.utils import *
from data.data import *
from net.net import TwoTowerModel, TwoTowerModelExtended

warnings.filterwarnings("ignore")

class Trainer:
    '''
    class to handle training
    '''
    def __init__(self, exp_name, epochs, batch_size, dataroot, lr, num_classes, isresume) -> None:
        '''
        initializes the object
        Parameters:
        -----------
        epochs: no of iterations to train 
        batch_size: no of images in a batch
        dataroot: root dir of data
        lr: learning rate
        num_classes: no of classes
        isresume: if to resume training
        '''
        
        #--------- set hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.writer = SummaryWriter(log_dir='runs/'+ exp_name)
        self.lr = lr
        
        #--------- get data loaders
        dset = MNISTPairv2(root=dataroot, is_train=True)
        length = len(dset)
        tl = int(0.8*length)
        train_dset, val_dset = random_split(dset, [tl, length-tl])
        self.train_loader = DataLoader(train_dset, batch_size=batch_size, num_workers=0, shuffle=True)
        self.val_loader = DataLoader(val_dset, batch_size=batch_size, num_workers=4, shuffle=False)

        print('train loader size', len(self.train_loader))
        print('val loader size', len(self.val_loader))

        #---------Initialize model, optimizer, loss function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clf = TwoTowerModelExtended(in_channel=1, out_channel=num_classes).to( self.device)
       
        if isresume and os.path.exists(ckpt_dir+'/best_epoch.pt'):
            self.clf.load_state_dict(torch.load(ckpt_dir+'/best_epoch.pt'))

        cls_weights = get_weights_MNIST()
        self.cls_weights = torch.from_numpy(cls_weights)
        
        self.optimizer = torch.optim.Adam(self.clf.parameters(), lr=lr, weight_decay=0.01)

        

    def __training_loop(self, epoch, io):
        '''
        This is the main training loop called at every epoch
        
        Parameters:
        -----------
        epoch: epoch no
        io: the log file handler
        '''
        time_str = 'Train start:' + ctime(time())
        io.cprint(time_str)

        self.clf.train()
        train_loss = 0
        predlist = []
        labellist = []
        for i , (data1, data2, sumtensor, label) in enumerate(self.train_loader):

            data1, data2, sumtensor, label = data1.to(self.device), data2.to(self.device), sumtensor.to(self.device), label.to(self.device)
            #import pdb; pdb.set_trace()
            pred = self.clf(data1, data2, sumtensor)
            
            loss = F.cross_entropy(pred, label, weight = self.cls_weights)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            
            pred = torch.argmax(pred, dim=1)
            predlist.extend(pred.cpu().numpy())
            labellist.extend(label.cpu().numpy())

            
            if i%100 == 0:
                log = 'Training epoch:{} iteration: {} loss:{}'.format(epoch, i, loss)
                io.cprint(log)

        time_str = 'Train end:' + ctime(time())
        io.cprint(time_str)

        train_loss /= len(self.train_loader)
        report = classification_report(labellist, predlist)
        prec = precision_score(labellist, predlist, average = 'weighted')
        recall = recall_score(labellist, predlist, average = 'weighted')
        pack = {
            'prec': prec,
            'recall': recall,
            'report': report,
            'loss': train_loss
        }
        return pack

    def __validation_loop(self, epoch, io):
        time_str = 'Validation start:' + ctime(time())
        io.cprint(time_str)

        self.clf.eval()
        val_loss = 0
        predlist = []
        labellist = []
        for i , (data1, data2, sumtensor, label) in enumerate(self.val_loader):

            data1, data2, sumtensor, label = data1.to(self.device), data2.to(self.device), sumtensor.to(self.device), label.to(self.device)
            pred = self.clf(data1, data2, sumtensor)
            loss = F.cross_entropy(pred, label, weight = self.cls_weights)
            val_loss += loss.item()
            pred = torch.argmax(pred, dim=1)
            predlist.extend(pred.cpu().numpy())
            labellist.extend(label.cpu().numpy())

            if i%100 == 0:
                log = 'Validation epoch:{} iteration: {} loss:{}'.format(epoch, i, loss)
                io.cprint(log)
        
        time_str = 'Validation end:' + ctime(time())
        io.cprint(time_str)

        val_loss /= len(self.val_loader)
        
        report = classification_report(labellist, predlist)
        prec = precision_score(labellist, predlist, average = 'weighted')
        recall = recall_score(labellist, predlist, average = 'weighted')
        pack = {
            'prec': prec,
            'recall': recall,
            'report': report,
            'loss': val_loss
        }
        return pack

    def train(self, io):
        best_AP = 0.0

        for epoch in range(self.epochs):
            io.cprint('---------------------Epoch %d/%d---------------------' % (epoch, args.epochs))
            
            pack = self.__training_loop(epoch, io)
            self.writer.add_scalar('train/loss', pack['loss'], epoch)
            self.writer.add_scalar('train/precision', pack['prec'], epoch)
            self.writer.add_scalar('train/recall', pack['recall'], epoch)
            log = 'Training epoch:{} loss:{} '.format(epoch, pack['loss'])
            log += '\n %s'%pack['report']
            io.cprint(log)

            pack = self.__validation_loop(epoch, io)
            self.writer.add_scalar('valid/loss', pack['loss'], epoch)
            self.writer.add_scalar('valid/precision', pack['prec'], epoch)
            self.writer.add_scalar('valid/recall', pack['recall'], epoch)
            log = 'Validation epoch:{} loss:{}'.format(epoch, pack['loss'])
            log += '\n %s'%pack['report']
            io.cprint(log)


            #----------save best valid precision model
            if pack['prec'] > best_AP:
                torch.save(self.clf.state_dict(), '{}/{}.pt'.format(ckpt_dir,'best_epoch'))
                best_AP = pack['prec']

            if epoch == self.epochs-1:
                torch.save(self.clf.state_dict(), '{}/{}.pt'.format(ckpt_dir,'latest_epoch'))

        self.writer.close()




if __name__ == '__main__':
    try:
        args = get_args()

    except ValueError:
        print("Missing or invalid arguments")
        sys.exit(0)

    ckpt_dir = 'checkpoints/%s'%args.exp_name+'/models'
    os.makedirs(ckpt_dir, exist_ok = True)

    out_dir = 'checkpoints/%s'%args.exp_name+'/out'
    os.makedirs(out_dir, exist_ok = True)

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint('Program start: %s' % ctime(time()))

    #------ logging the hyperparameters settings
    for arg in vars(args):
        io.cprint('{} : {}'.format(arg, getattr(args, arg) ))

    trainer = Trainer(args.exp_name, args.epochs, args.batch_size, 
                        args.dataroot, args.lr, args.num_classes, args.resume)

    trainer.train(io)



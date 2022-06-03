
'''
This file is the starting point of training a classifier 
of gender or hair color classificazion individually
'''

import sys
import os
from time import time, ctime
import torch
from torch.utils.tensorboard import SummaryWriter

import warnings

from utils.utils import *
from data.data import *
from net.net import Encoder, Decoder

warnings.filterwarnings("ignore")

class Trainer:
    '''
    class to handle training
    '''
    def __init__(self, exp_name, epochs, batch_size, dataroot, lr, isresume) -> None:
        '''
        initializes the object
        Parameters:
        -----------
        epochs: no of iterations to train 
        batch_size: no of images in a batch
        dataroot: root dir of data
        lr: learning rate
        isresume: if to resume training
        '''
        
        #--------- set hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.writer = SummaryWriter(log_dir='runs/'+ exp_name)
        self.lr = lr
        
        #--------- get data loaders
        self.train_loader, self.val_loader, self.test_loader = get_MNIST(dataroot, batch_size, num_workers=0)

        print('train loader size', len(self.train_loader))
        print('val loader size', len(self.val_loader))

        #---------Initialize model, optimizer, loss function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder(input_dim=1, encoded_dim=128).to( self.device)
        self.decoder = Decoder(input_dim=1, encoded_dim=128).to( self.device)
       
        if isresume and os.path.exists(ckpt_dir+'/best_epoch.pt'):
            self.encoder.load_state_dict(torch.load(ckpt_dir+'/best_epoch.pt'))

        cls_weights = get_weights_MNIST('data/map_sum.npy')
        self.cls_weights = torch.from_numpy(cls_weights)
        params =  list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.01)

        

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

        self.encoder.train()
        self.decoder.train()
        train_loss = 0.0
        fake, org = [], []
        for i , (data1, label) in enumerate(self.train_loader):
            
            data1 = data1.to(self.device)
            
            latent = self.encoder(data1)
            out = self.decoder(latent)
            #import pdb; pdb.set_trace()
            
            loss = F.smooth_l1_loss(out, data1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.detach().item()
            
            if i%100 == 0:
                log = 'Training epoch:{} iteration: {} loss:{}'.format(epoch, i, loss)
                io.cprint(log)
                fake.extend(out.detach().cpu().numpy())
                org.extend(data1.detach().cpu().numpy())

        time_str = 'Train end:' + ctime(time())
        io.cprint(time_str)

        train_loss /= len(self.train_loader)

        pack = {
            'loss': train_loss
        }
        # vis results
        # vis_results(fake, org, out_dir+str(epoch)+'_{}.png')
        return pack

    def __validation_loop(self, epoch, io):
        time_str = 'Validation start:' + ctime(time())
        io.cprint(time_str)

        self.encoder.eval()
        self.decoder.eval()
        val_loss = 0.0
        fake, org = [], []
        for i , (data1, label) in enumerate(self.val_loader):
            data1 = data1.to(self.device)
            
            latent = self.encoder(data1)
            out = self.decoder(latent)
            #import pdb; pdb.set_trace()
            
            loss = F.mse_loss(out, data1)
            val_loss += loss.detach().item()

            if i%100 == 0:
                log = 'Validation epoch:{} iteration: {} loss:{}'.format(epoch, i, loss)
                io.cprint(log)
                fake.extend(out.detach().cpu().numpy())
                org.extend(data1.detach().cpu().numpy())
        
        time_str = 'Validation end:' + ctime(time())
        io.cprint(time_str)

        val_loss /= len(self.val_loader)
        
        pack = {
            'loss': val_loss
        }
        # vis results
        vis_results(fake, org, out_dir+str(epoch)+'_{}.png')
        return pack

    def train(self, io):
        min_loss = float('Inf')

        for epoch in range(self.epochs):
            io.cprint('---------------------Epoch %d/%d---------------------' % (epoch, args.epochs))
            
            pack = self.__training_loop(epoch, io)
            self.writer.add_scalar('train/loss', pack['loss'], epoch)
            log = 'Training epoch:{} loss:{}'.format(epoch, pack['loss'])
            io.cprint(log)

            pack = self.__validation_loop(epoch, io)
            self.writer.add_scalar('valid/loss', pack['loss'], epoch)
            log = 'Validation epoch:{} loss:{}'.format(epoch, pack['loss'])
            io.cprint(log)


            #----------save best valid precision model
            if pack['loss'] < min_loss:
                torch.save(self.encoder.state_dict(), '{}/{}.pt'.format(ckpt_dir,'best_epoch'))
                min_loss = pack['loss']

            if epoch == self.epochs-1:
                torch.save(self.encoder.state_dict(), '{}/{}.pt'.format(ckpt_dir,'latest_epoch'))

        self.writer.close()




if __name__ == '__main__':
    try:
        args = get_args()

    except ValueError:
        print("Missing or invalid arguments")
        sys.exit(0)

    ckpt_dir = 'checkpoints/%s'%args.exp_name+'/models'
    os.makedirs(ckpt_dir, exist_ok = True)

    out_dir = 'checkpoints/%s'%args.exp_name+'/out/'
    os.makedirs(out_dir, exist_ok = True)

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint('Program start: %s' % ctime(time()))

    #------ logging the hyperparameters settings
    for arg in vars(args):
        io.cprint('{} : {}'.format(arg, getattr(args, arg) ))

    trainer = Trainer(args.exp_name, args.epochs, args.batch_size, args.dataroot, args.lr, args.resume)

    trainer.train(io)



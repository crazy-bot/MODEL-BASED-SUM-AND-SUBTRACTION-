import argparse
from bdb import set_trace
from code import InteractiveConsole
import torch
import numpy as np
import torch.nn.functional as F
import cv2

class IOStream():
    '''
    This is an utility class for logging
    '''
    def __init__(self, path):
        '''
        Initializes the class

        parameters
        ----------
        path: The file path of desired log file
        '''
        self.f = open(path, 'a')

    def cprint(self, text):
        '''
        prints the text in console and flush in log file

        parameters
        ----------
        text (str): log text
        '''
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        '''
        closes the log file handler
        '''
        self.f.close()


def vis_results(pred, label, outdir):
    #import pdb; pdb.set_trace()
    b = len(pred)
    for i in range(b):
        combo = np.hstack([label[i][0]*255, pred[i][0]*255])
        cv2.imwrite(outdir.format(i), combo)

# def get_pairsum(data, label):
#     '''
#     data: (B, C, H, W)
#     labels: (B, 1) [0,1,2,3,4,5,6,7,8,9]
#     '''
#     B = data.size()[0]
#     assert B%2 == 0, 'batch_size should be even'
#     data1, label1 = data[:B//2,:,:,:], label[:B//2,:,:,:]
#     data2, label2 = data[B//2:,:,:,:], label[B//2:,:,:,:]
#     labels = torch.FloatTensor(np.arange(19))

#     input = []
#     target = []
#     for i in range(B//2):






def get_args():
    '''
    this method sets all the arguments with default values for command 
    line interface for training
    
    Returns
    -------
    args: parsed arguments into object
    '''
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--dataroot', type=str, default='data', metavar='N',
                        help='Path of the root dir of MNIST dataset')

    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
                        
    parser.add_argument('--batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch)')
                        
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of episode to train ')
                        
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    
    parser.add_argument('--num_classes', type=int, default=10,
                        help='num of target labels')
    
    parser.add_argument('--resume', type=bool, default=False,
                        help='to resume training or not')
    
    args = parser.parse_args()
    return args

def get_testargs():
    '''
    this method sets all the arguments with default values for command 
    line interface for testing

    Returns
    -------
    args: parsed arguments into object
    '''
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--dataroot', type=str, default='exp', metavar='N',
                        help='Path of the root dir of image dataset')
                        
    parser.add_argument('--classification_type', type=str, default='gender', metavar='N',
                        choices = ['gender', 'haircolor', 'multitask'], help='which classification you want to train')
    
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
                        
    parser.add_argument('--ckpt', type=str, default='best', metavar='batch_size',
                        choices = ['best', 'latest'], help='Size of batch)')
                        
    args = parser.parse_args()
    return args
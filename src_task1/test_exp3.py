
'''
This file is the starting point of training a classifier 
of gender or hair color classificazion individually
'''

import sys
import os
from time import time, ctime
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

import warnings

from utils.utils import *
from data.data import *
from net.net import AutoEncoderModel

warnings.filterwarnings("ignore")


class Tester:
    def __init__(self, dataroot, batch_size, num_classes, ckpt_dir) -> None:
        
        self.dataroot = dataroot
        self.target_names = np.arange(num_classes)
        #--------- loading dataset and creating loader
        dset = MNISTPair(root=dataroot, is_train=False)        
        self.test_loader = DataLoader(dset, batch_size=batch_size, num_workers=4)        
        print('test loader size', len(self.test_loader))

        #---------Initialize model, optimizer, loss function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clf = AutoEncoderModel(input_dim=1, encoded_dim=128, out_channel=num_classes).to( self.device)
       
        self.clf.load_state_dict(torch.load(ckpt_dir))
        self.clf.eval()

    def test(self, io):
        time_str = 'Testing start:' + ctime(time())
        io.cprint(time_str)

        predlist = []
        labellist = []
        for i , (data1, data2, label) in enumerate(self.test_loader):
            data1, data2, label = data1.to(self.device), data2.to(self.device), label.to(self.device)
            
            x1, x2, pred = self.clf(data1, data2)
            pred = torch.argmax(pred, dim=1)

            predlist.extend(pred.cpu().numpy())
            labellist.extend(label.cpu().numpy())
        
        time_str = 'Testing end:' + ctime(time())
        io.cprint(time_str)
        #import pdb; pdb.set_trace()
        report = classification_report(labellist, predlist)
        log = 'classification_report: \n {} '.format(report)
        io.cprint(log)



if __name__ == '__main__':
    try:
        args = get_testargs()

    except ValueError:
        print("Missing or invalid arguments")
        sys.exit(0)

    ckpt_dir = 'checkpoints/%s'%args.exp_name+'/models/%s'%args.ckpt+'_epoch.pt'
    assert os.path.exists(ckpt_dir),'model doest not exist: %s'%ckpt_dir

    io = IOStream('checkpoints/' + args.exp_name + '/test.log')
    io.cprint('Program start: %s' % ctime(time()))

    #------ logging the hyperparameters settings
    for arg in vars(args):
        io.cprint('{} : {}'.format(arg, getattr(args, arg) ))

    tester = Tester(args.dataroot, args.batch_size,  args.num_classes, ckpt_dir)

    tester.test(io)



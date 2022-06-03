
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
from net.net import TwoTowerModelExtended

warnings.filterwarnings("ignore")


class Tester:
    def __init__(self, dataroot, batch_size, num_classes, ckpt_dir, issum) -> None:
        
        self.dataroot = dataroot
        self.target_names = np.arange(num_classes)

        #--------- loading dataset and creating loader
        dset = MNISTPairv2Test(root=dataroot, issum=issum)        
        self.test_loader = DataLoader(dset, batch_size=batch_size, num_workers=4)   
        self.dset = dset     
        print('test loader size', len(self.test_loader))

        #---------Initialize model, optimizer, loss function
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clf = TwoTowerModelExtended(in_channel=1, out_channel=num_classes).to( self.device)
        self.clf.load_state_dict(torch.load(ckpt_dir))
        self.clf.eval()

    def test(self, io, issum):
        time_str = 'Testing start:' + ctime(time())
        io.cprint(time_str)

        predlist, labellist, flaglist, l1list, l2list = [],[],[],[], []

        for i , (data1, data2, flag, label, l1, l2) in enumerate(self.test_loader):
            data1, data2, label = data1.to(self.device), data2.to(self.device), label.to(self.device)
            if issum:
                flag = torch.ones([data1.shape[0], 1], dtype=torch.float32, device=self.device)
            else:
                flag = torch.zeros([data1.shape[0], 1], dtype=torch.float32, device=self.device)
            
            pred = self.clf(data1, data2, flag)
            pred = torch.argmax(pred, dim=1)
            
            predlist.extend(pred.cpu().numpy())
            labellist.extend(label.cpu().numpy())
            flaglist.extend(flag.detach().cpu().numpy())
            l1list.extend(l1.numpy())
            l2list.extend(l2.numpy())
        
        time_str = 'Testing end:' + ctime(time())
        io.cprint(time_str)
        #import pdb; pdb.set_trace()

        # save output
        flag = 'test_sum' if issum else 'test_subtract'
        f = open(out_dir + '/%s'%flag, 'w')
        txt = 'flag     label1      label2      GT      pred \n'
        f.write(txt)
        for i in range(len(predlist)):
            target = self.dset.idxtolabel[labellist[i]]
            predlabel = self.dset.idxtolabel[predlist[i]]
            txt = '%d         %d        %d        %d        %d\n'%(flaglist[i], l1list[i], l2list[i], target, predlabel)
            f.write(txt)
        f.close()

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

    out_dir = 'checkpoints/%s'%args.exp_name+'/out'
    os.makedirs(out_dir, exist_ok = True)

    io = IOStream('checkpoints/' + args.exp_name + '/test.log')
    io.cprint('Program start: %s' % ctime(time()))

    #------ logging the hyperparameters settings
    for arg in vars(args):
        io.cprint('{} : {}'.format(arg, getattr(args, arg) ))

    tester = Tester(args.dataroot, args.batch_size,  args.num_classes, ckpt_dir, args.issum)

    tester.test(io, args.issum)



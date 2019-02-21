from tensorboardX import SummaryWriter
import os, sys

# https://github.com/nashory/pggan-pytorch/blob/master/tf_recorder.py
class tf_recorder:
    def __init__(self, network_name):
        os.system('mkdir -p repo/tensorboard')
        for i in range(1000):
            self.targ = 'repo/tensorboard/{}_{}'.format(network_name, i)
            if not os.path.exists(self.targ):
                self.writer = SummaryWriter(self.targ)
                break
        self.niter = 0
                
    def add_scalar(self, index, val):
        self.writer.add_scalar(index, val, self.niter)

    def add_scalars(self, index, group_dict):
        self.writer.add_scalar(index, group_dict, self.niter)

    def iter(self):
        self.niter += 1
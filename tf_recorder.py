from tensorboardX import SummaryWriter
import os, sys
import os.path

# https://github.com/nashory/pggan-pytorch/blob/master/tf_recorder.py
class tf_recorder:
    def __init__(self, network_name, logdir='repo/tensorboard'):
        os.system('mkdir -p {}'.format(logdir))
        for i in range(1000):
            self.targ = os.path.join(logdir, '{}_{}'.format(network_name, i))
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

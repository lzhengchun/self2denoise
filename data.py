import numpy as np 
import h5py, threading, random, imageio
import queue as Queue
import h5py, torch
from skimage.feature import peak_local_max
from skimage.util import random_noise

class bkgdGen(threading.Thread):
    def __init__(self, data_generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = data_generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            # block if necessary until a free slot is available
            self.queue.put(item, block=True, timeout=None)
        self.queue.put(None)

    def next(self):
        # block if necessary until an item is available
        next_item = self.queue.get(block=True, timeout=None)
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

def gen_train_batch_bg(mbsize, psz=512, dev='cuda', nvar=0.1):
    with h5py.File('./dataset/glass-gt-s1024-1152.hdf5', 'r') as h5fd:
        img_gt = h5fd['data'][:-1]
    img_ns = random_noise(img_gt/255., mode = 'gaussian', var=nvar).astype(np.float32)

    while True:
        sidx = np.random.randint(0, img_ns.shape[0], mbsize)
        rst  = np.random.randint(0, img_ns.shape[1]-psz, mbsize)
        cst  = np.random.randint(0, img_ns.shape[2]-psz, mbsize)
        mb_ns = np.array([img_ns[_s, np.newaxis, _r:_r+psz, _c:_c+psz] for _s, _r, _c in zip(sidx, rst, cst)])
        mb_gt = np.array([img_gt[_s, np.newaxis, _r:_r+psz, _c:_c+psz] for _s, _r, _c in zip(sidx, rst, cst)])

        yield torch.from_numpy(mb_ns).to(dev), torch.from_numpy(mb_gt).to(dev)

def get1batch4test(sidx=None, mbsize=None, psz=512, dev='cuda', nvar=0.1):
    with h5py.File('./dataset/glass-gt-s1024-1152.hdf5', 'r') as h5fd:
        img_gt = h5fd['data'][-1:] # use the last image for model validation

    img_ns = random_noise(img_gt/255., mode = 'gaussian', var=nvar).astype(np.float32)

    if sidx is None:
        sidx = np.random.randint(0, img_ns.shape[0], mbsize)

    rst  = np.random.randint(0, img_ns.shape[1]-psz, len(sidx))
    cst  = np.random.randint(0, img_ns.shape[2]-psz, len(sidx))
    mb_ns = np.array([img_ns[_s, np.newaxis, _r:_r+psz, _c:_c+psz] for _s, _r, _c in zip(sidx, rst, cst)])
    mb_gt = np.array([img_gt[_s, np.newaxis, _r:_r+psz, _c:_c+psz] for _s, _r, _c in zip(sidx, rst, cst)])

    return torch.from_numpy(mb_ns).to(dev), torch.from_numpy(mb_gt).to(dev)

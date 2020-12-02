#! /home/beams/ZHENGCHUN.LIU/usr/miniconda3/envs/torch/bin/python
from model import unet, model_init, DnCNN
# from n2sUnet import Unet
from mask import Masker
import torch, argparse, os, time, sys, shutil, skimage.io, h5py
from util import str2bool, save2img
from data import bkgdGen, gen_train_batch_bg, get1batch4test
import numpy as np

parser = argparse.ArgumentParser(description='denoise with self-supervised learning')
parser.add_argument('-gpus',   type=str, default="", help='list of visiable GPUs')
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
parser.add_argument('-lr',     type=float,default=5e-4, help='learning rate')
parser.add_argument('-mbsize', type=int, default=16, help='mini batch size')
parser.add_argument('-psz',    type=int, default=256, help='patch size')
parser.add_argument('-maxep',  type=int, default=1000, help='max training epoches')
parser.add_argument('-nvar',   type=float, default=0.1, help='noise variance')
parser.add_argument('-print',  type=str2bool, default=False, help='1:print to terminal; 0: redirect to file')

args, unparsed = parser.parse_known_args()

if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)

if len(args.gpus) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")

itr_out_dir = args.expName + '-itrOut'
if os.path.isdir(itr_out_dir): 
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir) # to save temp output

# redirect print to a file
if args.print == 0:
    sys.stdout = open(os.path.join(itr_out_dir, 'iter-prints.log'), 'w') 

mb_data_iter = bkgdGen(data_generator=gen_train_batch_bg(mbsize=args.mbsize, \
                                      psz=args.psz, nvar=args.nvar), \
                       max_prefetch=16)   

def main(args):
    model = unet()
    # model = DnCNN(1, num_of_layers = 8)
    _ = model.apply(model_init) # init model weights and bias
    
    masker = Masker(width=4, mode='zero')

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(torch_devs)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    for epoch in range(args.maxep+1):
        time_it_st = time.time()
        X_mb, y_mb = mb_data_iter.next() 
        time_data = 1000 * (time.time() - time_it_st)

        mdl_input, mask = masker.mask(X_mb, epoch)
    
        model.train() # sets the module in training mode.
        optimizer.zero_grad()
        pred = model.forward(mdl_input)
        loss = criterion(pred*mask, X_mb*mask)
        loss.backward()
        optimizer.step() 

        time_e2e = 1000 * (time.time() - time_it_st)
        itr_prints = '[Info] @ %.1f Epoch: %05d, loss: %.4f, elapse: %.2fs/itr' % (\
                    time.time(), epoch, loss.cpu().detach().numpy(), (time.time() - time_it_st), )
        print(itr_prints)

        if epoch % 100 == 0:
            if epoch == 0:
                val_ns, val_gt = get1batch4test(sidx=range(1), psz=2048, nvar=args.nvar)
                save2img(val_ns[0,0].cpu().numpy(), '%s/ns.png' % (itr_out_dir))
                save2img(val_gt[0,0].cpu().numpy(), '%s/gt.png' % (itr_out_dir))

            model.eval() # sets the module in inference mode.
            with torch.no_grad():
                mdn = masker.infer_full_image(val_ns, model)
                ddn = model.forward(val_ns)

            save2img(mdn[0,0].cpu().numpy(), '%s/mdn-it%05d.png' % (itr_out_dir, epoch))
            save2img(ddn[0,0].cpu().numpy(), '%s/ddn-it%05d.png' % (itr_out_dir, epoch))

            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), "%s/mdl-it%05d.pth" % (itr_out_dir, epoch))
            else:
                torch.save(model.state_dict(), "%s/mdl-it%05d.pth" % (itr_out_dir, epoch))
        sys.stdout.flush()
        
if __name__ == "__main__":
    main(args)

import torch
import torchmetrics
from torch import nn
import module.resnet as resnet
import module.utils as utils
import module.resnet
from torchinfo import summary
from torchvision import transforms
from module.data import CfImageDataset
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import argparse
import module.aug as aug

torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(description="Train a network for ...")
    opt = parser.parse_args()

    # add code for datasets (we always use train and validation/ test set)
    data_transforms = aug.horizontal_vertical_flip()

    train_dataset = CfImageDataset("F:/DATA/classify-leaves/train.csv",
                                   "F:/DATA/classify-leaves/",
                                   mode="train",
                                   transform=data_transforms)
    valid_dataset = CfImageDataset("F:/DATA/classify-leaves/train.csv",
                                   "F:/DATA/classify-leaves/",
                                   mode="valid")

    # instantiate network (which has been imported from *networks.py*)
    net = resnet.ResNet(64, 176, 4)

    # create losses (criterion in pytorch)
    criterion_L1 = nn.CrossEntropyLoss()

    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
        ...

    # create optimizers
    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    ...

    # load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 0
    if opt.resume:
        ckpt = load_checkpoint(opt.path_to_checkpoint)  # custom method for loading last checkpoint
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        optim.load_state_dict(ckpt['optim'])
        print("last checkpoint restored")
        ...

    # if we want to run experiment on multiple GPUs we move the models there
    net = torch.nn.DataParallel(net)
    ...

    # typically we use tensorboardX to keep track of experiments
    writer = SummaryWriter(...)

    # now we start the main loop
    n_iter = start_n_iter
    for epoch in range(start_epoch, opt.epochs):
        # set models to train mode
        net.train()
        ...

        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(BackgroundGenerator(train_data_loader, ...)),
                    total=len(train_data_loader))
        start_time = time.time()

        # for loop going through dataset
        for i, data in pbar:
            # data preparation
            img, label = data
            if use_cuda:
                img = img.cuda()
                label = label.cuda()
            ...

            # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
            prepare_time = start_time - time.time()

            # forward and backward pass
            optim.zero_grad()
            ...
            loss.backward()
            optim.step()
            ...

            # udpate tensorboardX
            writer.add_scalar(..., n_iter)
            ...

            # compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
                process_time / (process_time + prepare_time), epoch, opt.epochs))
            start_time = time.time()

        # maybe do a test pass every x epochs
        if epoch % x == x - 1:
            # bring models to evaluation mode
            net.eval()
            ...
            # do some tests
            pbar = tqdm(enumerate(BackgroundGenerator(test_data_loader, ...)),
                        total=len(test_data_loader))
            for i, data in pbar:
                ...

            # save checkpoint if needed
            ...
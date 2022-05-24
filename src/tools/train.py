import argparse
import os
import pickle
import time
from datetime import datetime
from docopt import docopt

import numpy as np
import torch
import torchmetrics
from tqdm import tqdm
from yacs.config import CfgNode

from src.config.defaults import combine_cfgs
from src.data.dataset import build_dataloader
from src.modeling.meta_arch.build import build_model
from src.modeling.solvers.build import build_loss, build_metric, build_optimizer
from src.tools.helper import save_checkpoint


def train(cfg: CfgNode):

    torch.backends.cudnn.benchmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    assert cfg.OUTPUT_DIR != ''
    output_path = cfg.OUTPUT_DIR

    timestamp = datetime.now().isoformat(sep="T", timespec="auto")
    name_timestamp = timestamp.replace(":", "_")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Make backup folders if they do not exist
    backup_dir = os.path.join(output_path, 'model_backup')
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)

    # Make result folders if they do not exist
    results_dir = os.path.join(output_path, 'result')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    cfg.dump(stream=open(os.path.join(results_dir, f'config_{name_timestamp}.yaml'), 'w'))

    state_fpath = os.path.join(output_path, 'model.pt')

    # Performance path where we'll save our metrics to trace.p
    perf_path = os.path.join(results_dir, 'trace.p')
    perf_trace = []

    train_loader = build_dataloader(cfg.DATASET, mode='train')
    valid_loader = build_dataloader(cfg.DATASET, mode='valid')

    model = build_model(cfg.MODEL)
    use_cuda = torch.cuda.is_available()
    solver_cfg = cfg.SOLVER
    loss_fn = build_loss(solver_cfg.LOSS)
    updater = build_optimizer(model, solver_cfg.OPTIMIZER)
    train_metrics = build_metric(solver_cfg.METRIC)
    valid_metrics = build_metric(solver_cfg.METRIC)
    train_loss = torchmetrics.MeanMetric()
    valid_loss = torchmetrics.MeanMetric()

    def train_step(x, y):
        pred = model(x)
        loss = loss_fn(pred, y)
        updater.zero_grad(set_to_none=True)
        loss.backward()
        updater.step()
        with torch.no_grad():
            train_metrics.update(pred, y)
            train_loss.update(loss)

    def eval_step(x, y):
        pred = model(x)
        loss = loss_fn(pred, y)
        valid_metrics.update(pred, y)
        valid_loss.update(loss)

    if cfg.RESUME_PATH != '':
        checkpoint = torch.load(cfg.RESUME_PATH, map_location='cpu')
        epoch_start = checkpoint['epoch']
        valid_best = checkpoint['valid_best']
        unimproved = checkpoint['unimproved']
        n_iter_start = checkpoint['n_iter']
        model.load_state_dict(checkpoint['model'])
        updater.load_state_dict(checkpoint['optimizer'])
    else:
        valid_best = -np.inf
        unimproved = 0
        epoch_start = 0
        n_iter_start = 0
    if use_cuda:
        model = model.cuda()
        train_loss = train_loss.to(torch.device("cuda", 0))
        valid_loss = valid_loss.to(torch.device("cuda", 0))
        train_metrics = train_metrics.to(torch.device("cuda", 0))
        valid_metrics = valid_metrics.to(torch.device("cuda", 0))
    max_epoch = solver_cfg.MAX_EPOCHS
    min_epoch = solver_cfg.MIN_EPOCHS
    patience = solver_cfg.PATIENCE
    n_iter = n_iter_start
    for ei in range(epoch_start, max_epoch):
        model.train()
        if ei >= min_epoch and unimproved > patience:
            break
        else:
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            start_time = time.time()
            for i, data in pbar:
                img, label = data
                if use_cuda:
                    img = img.cuda()
                    label = label.cuda()
                prepare_time = start_time - time.time()
                train_step(img, label)
                n_iter += 1
                # compute computation time and *compute_efficiency*
                process_time = start_time - time.time() - prepare_time
                pbar.set_description("Train: Compute efficiency: {:.2f}, epoch: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(
                    process_time / (process_time + prepare_time),
                    ei, max_epoch,
                    train_loss.compute(), train_metrics.compute()))
                start_time = time.time()
            train_epoch_loss = train_loss.compute().cpu().numpy()
            train_epoch_metrics = train_metrics.compute().cpu().numpy()

        check_epoch = solver_cfg.CHECK_EPOCHS
        if ei % check_epoch == check_epoch - 1:
            model.eval()
            pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
            with torch.no_grad():
                for i, data in pbar:
                    img, label = data
                    if use_cuda:
                        img = img.cuda()
                        label = label.cuda()

                    eval_step(img, label)
                    pbar.set_description("Valid: epoch: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(
                        ei, max_epoch,
                        valid_loss.compute(), valid_metrics.compute()))

            valid_epoch_loss = valid_loss.compute().cpu().numpy()
            valid_epoch_metrics = valid_metrics.compute().cpu().numpy()

            if valid_epoch_metrics > valid_best:
                unimproved = 0
                valid_best = valid_epoch_metrics
                save_state = {
                    'model': model.state_dict(),
                    'epoch': ei,
                    'n_iter': n_iter,
                    'optimizer': updater.state_dict(),
                    'valid_best': valid_best,
                    'unimproved': unimproved
                }
                print("Saving Best (epoch %d)" % ei)
                torch.save(save_state, state_fpath)

            else:
                unimproved += 1
            save_state = {
                    'model': model.state_dict(),
                    'epoch': ei,
                    'n_iter': n_iter,
                    'optimizer': updater.state_dict(),
                    'valid_best': valid_best,
                    'unimproved': unimproved
                }
            print("Making a backup (epoch %d)" % ei)
            backup_fpath = os.path.join(backup_dir, "model_bak_%06d.pt" % (ei,))
            save_checkpoint(save_state, backup_fpath, max_keep=10)
            perf_trace.append(
                {
                    'epoch': ei,
                    'train_loss': train_epoch_loss,
                    'train_acc': train_epoch_metrics,
                    'val_err': valid_epoch_loss,
                    'val_acc': valid_epoch_metrics,
                }
            )
            pickle.dump(perf_trace, open(perf_path, 'wb'))
        train_loss.reset()
        valid_loss.reset()
        train_metrics.reset()
        valid_metrics.reset()


def main():
    parser = argparse.ArgumentParser(description='pytorch training')
    parser.add_argument('-o', default='', help='output path', type=str)
    parser.add_argument("--cfg", help='yaml config file path', type=str)
    args = parser.parse_args()
    cfg = combine_cfgs(args.cfg)
    cfg.OUTPUT_DIR = args.o
    train(cfg)


if __name__ == '__main__':
    main()



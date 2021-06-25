"""
This file is a modified version of https://github.com/sigsep/open-unmix-pytorch/blob/master/scripts/train.py
"""

import argparse
import model
import testx
import data
import torch
import time
from pathlib import Path
import tqdm
import json
import utils
import sklearn.preprocessing
import numpy as np
import random
import os
import copy
import museval
import norbert

import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

import model_utls


tqdm.monitor_interval = 0


def train(args, unmix, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    unmix.train()
    unmix.stft.center = True
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for data in pbar:
        pbar.set_description("Training batch")
        x = data[0]  # mix
        y = data[1]  # target
        z = data[2]  # text
        x, y, z = x.to(device), y.to(device), z.to(device)
        optimizer.zero_grad()
        if args.alignment_from:
            inputs = (x, z, data[3].to(device))  # add attention weights to input
        else:
            inputs = (x, z)
        Y_hat = unmix(inputs)
        Y = unmix.transform(y)
        loss_fn = torch.nn.L1Loss(reduction='sum')
        loss = loss_fn(Y_hat, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unmix.parameters(), max_norm=2, norm_type=1)
        optimizer.step()
        losses.update(loss.item(), Y.size(1))
    return losses.avg


def valid(args, unmix, device, valid_sampler):
    losses = utils.AverageMeter()

    unmix.eval()
    unmix.stft.center = True
    with torch.no_grad():
        for data in valid_sampler:
            x = data[0]  # mix
            y = data[1]  # vocals
            z = data[2]  # text
            x, y, z = x.to(device), y.to(device), z.to(device)
            if args.alignment_from:
                inputs = (x, z, data[3].to(device))  # add attention weight to input
            else:
                inputs = (x, z)
            Y_hat = unmix(inputs)
            Y = unmix.transform(y)
            loss_fn = torch.nn.L1Loss(reduction='sum')  # in sms project, the loss is defined before looping over epochs
            loss = loss_fn(Y_hat, Y)

            losses.update(loss.item(), Y.size(1))
        return losses.avg #, sdr_avg.avg, sar_avg.avg, sir_avg.avg


def get_statistics(args, dataset):

    # dataset is an instance of a torch.utils.data.Dataset class

    scaler = sklearn.preprocessing.StandardScaler()  # tool to compute mean and variance of data

    # define operation that computes magnitude spectrograms
    spec = torch.nn.Sequential(
        model.STFT(n_fft=args.nfft, n_hop=args.nhop),
        model.Spectrogram(mono=True)
    )
    # return a deep copy of dataset:
    # constructs a new compound object and recursively inserts copies of the objects found in the original
    dataset_scaler = copy.deepcopy(dataset)

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None  # no scaling of sources before mixing
    dataset_scaler.random_chunks = False  # no random chunking of tracks
    dataset_scaler.random_track_mix = False  # no random accompaniments for vocals
    dataset_scaler.random_interferer_mix = False
    dataset_scaler.seq_duration = None  # if None, the original whole track from musdb is loaded

    # make a progress bar:
    # returns an iterator which acts exactly like the original iterable,
    # but prints a dynamically updating progressbar every time a value is requested.
    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)

    for ind in pbar:
        out = dataset_scaler[ind]  # x is mix and y is target source in time domain, z is text and ignored here
        x = out[0]
        y = out[1]
        pbar.set_description("Compute dataset statistics")
        X = spec(x[None, ...])  # X is mono magnitude spectrogram, ... means as many ':' as needed

        # X is spectrogram of one full track
        # at this point, X has shape (nb_frames, nb_samples, nb_channels, nb_bins) = (N, 1, 1, F)
        # nb_frames: time steps, nb_bins: frequency bands, nb_samples: batch size

        # online computation of mean and std on X for later scaling
        # after squeezing, X has shape (N, F)
        scaler.partial_fit(np.squeeze(X))  # np.squeeze: remove single-dimensional entries from the shape of an array

    # set inital input scaler values
    # scale_ and mean_ have shape (nb_bins,), standard deviation and mean are computed on each frequency band separately
    # if std of a frequency bin is smaller than m = 1e-4 * (max std of all freq. bins), set it to m
    std = np.maximum(   # maximum compares two arrays element wise and returns the maximum element wise
        scaler.scale_,
        1e-4*np.max(scaler.scale_)  # np.max = np.amax, it returns the max element of one array
    )
    return scaler.mean_, std


def main():
    parser = argparse.ArgumentParser(description='Open Unmix Trainer')

    # which target do we want to train?
    parser.add_argument('--target', type=str, default='vocals',
                        help='target source (will be passed to the dataset)')

    # experiment tag which will determine output folder in trained models, tensorboard name, etc.
    parser.add_argument('--tag', type=str)


    # allow to pass a comment about the experiment
    parser.add_argument('--comment', type=str, help='comment about the experiment')

    args, _ = parser.parse_known_args()

    # Dataset paramaters
    parser.add_argument('--dataset', type=str, default="musdb",
                        choices=[
                            'musdb_lyrics', 'timit_music', 'blended', 'nus', 'nus_train'
                        ],
                        help='Name of the dataset.')

    parser.add_argument('--root', type=str, help='root path of dataset')
    parser.add_argument('--output', type=str, default="trained_models/{}/".format(args.tag),
                        help='provide output path base folder name')

    parser.add_argument('--wst-model', type=str, help='Path to checkpoint folder for warmstart')

    # Trainig Parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, defaults to 1e-3')
    parser.add_argument('--patience', type=int, default=140,
                        help='maximum number of epochs to train (default: 140)')
    parser.add_argument('--lr-decay-patience', type=int, default=80,
                        help='lr decay patience for plateau scheduler')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.3,
                        help='gamma of learning rate scheduler decay')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')

    parser.add_argument('--alignment-from', type=str, default=None)
    parser.add_argument('--fake-alignment', action='store_true', default=False)


    # Model Parameters
    parser.add_argument('--unidirectional', action='store_true', default=False,
                        help='Use unidirectional LSTM instead of bidirectional')
    parser.add_argument('--nfft', type=int, default=4096,
                        help='STFT fft size and window size')
    parser.add_argument('--nhop', type=int, default=1024,
                        help='STFT hop size')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='hidden size parameter of dense bottleneck layers')
    parser.add_argument('--bandwidth', type=int, default=16000,
                        help='maximum model bandwidth in herz')
    parser.add_argument('--nb-channels', type=int, default=2,
                        help='set number of channels for model (1, 2)')
    parser.add_argument('--nb-workers', type=int, default=0,
                        help='Number of workers for dataloader.')
    parser.add_argument('--nb-audio-encoder-layers', type=int, default=2)
    parser.add_argument('--nb-layers', type=int, default=3)
    # name of the model class in model.py that should be used
    parser.add_argument('--architecture', type=str)
    # select attention type if applicable for selected model
    parser.add_argument('--attention', type=str)

    # Misc Parameters
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='less verbose during training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args, _ = parser.parse_known_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    print("Using Torchaudio: ", utils._torchaudio_available())
    dataloader_kwargs = {'num_workers': args.nb_workers, 'pin_memory': True} if use_cuda else {}

    writer = SummaryWriter(logdir=os.path.join('tensorboard', args.tag))

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset, valid_dataset, args = data.load_datasets(parser, args)

    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)


    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data.collate_fn, drop_last=True,
        **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, collate_fn=data.collate_fn, **dataloader_kwargs
    )

    if args.wst_model:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, train_dataset)

    max_bin = utils.bandwidth_to_max_bin(
        valid_dataset.sample_rate, args.nfft, args.bandwidth
    )

    train_args_dict = vars(args)
    train_args_dict['max_bin'] = int(max_bin)  # added to config
    train_args_dict['vocabulary_size'] = valid_dataset.vocabulary_size  # added to config

    train_params_dict = copy.deepcopy(vars(args))  # return args as dictionary with no influence on args

    # add to parameters for model loading but not to config file
    train_params_dict['scaler_mean'] = scaler_mean
    train_params_dict['scaler_std'] = scaler_std

    model_class = model_utls.ModelLoader.get_model(args.architecture)
    model_to_train = model_class.from_config(train_params_dict)
    model_to_train.to(device)

    optimizer = torch.optim.Adam(
        model_to_train.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10
    )

    es = utils.EarlyStopping(patience=args.patience)

    # if a model is specified: resume training
    if args.wst_model:
        model_path = Path(os.path.join('trained_models', args.wst_model)).expanduser()
        with open(Path(model_path, args.target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)


        model_to_train.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # train for another arg.epochs
        t = tqdm.trange(
            results['epochs_trained'],
            results['epochs_trained'] + args.epochs + 1,
            disable=args.quiet
        )
        train_losses = results['train_loss_history']
        valid_losses = results['valid_loss_history']
        train_times = results['train_time_history']
        best_epoch = 0

    # else start from 0
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()

        train_loss = train(args, model_to_train, device, train_sampler, optimizer)
        #valid_loss, sdr_val, sar_val, sir_val = valid(args, model_to_train, device, valid_sampler)
        valid_loss = valid(args, model_to_train, device, valid_sampler)

        writer.add_scalar("Training_cost", train_loss, epoch)
        writer.add_scalar("Validation_cost", valid_loss, epoch)

        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(
            train_loss=train_loss, val_loss=valid_loss
        )

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_train.state_dict(),
                'best_loss': es.best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target=args.target
        )

        # save params
        params = {
            'epochs_trained': epoch,
            'args': vars(args),
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'valid_loss_history': valid_losses,
            'train_time_history': train_times,
            'num_bad_epochs': es.num_bad_epochs
        }

        with open(Path(target_path,  args.target + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break


if __name__ == "__main__":
    main()

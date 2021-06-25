"""
Save the alignment matrices (hard attention weights) for the MUSDB dataset using the specified model (tag)
"""
import os
import pickle
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

import testx
import data
import model

from estimate_alignment import optimal_alignment_path

tag = 'JOINT3'  # tag of alignment model

target = 'vocals'

torch.manual_seed(0)

model_path = 'trained_models/{}'.format(tag)
device = 'cpu'
print("Device:", device)

# load model
unmix = testx.load_model(target, model_path, device)
unmix.return_alphas = True
unmix.stft.center = True

try:
    with open(os.path.join(model_path, target + '.json'), 'r') as stream:
        config = json.load(stream)
        keys = config['args'].keys()

        samplerate = config['args']['samplerate']
        text_units = config['args']['text_units']
        nb_channels = config['args']['nb_channels']
        nfft = config['args']['nfft']
        nhop = config['args']['nhop']
        data_set = config['args']['dataset']
        space_token_only = config['args']['space_token_only'] if 'space_token_only' in keys else False
except (FileNotFoundError):
    print('no config file found!')
    quit()



test_set = data.MUSDBLyricsDataTest(samplerate=samplerate, text_units=text_units,
                                    space_token_only=True)

val_set = data.MUSDBLyricsDataVal(samplerate=samplerate, text_units=text_units, space_token_only=True,
                                  return_name=True)

train_set = data.MUSDBLyricsDataTrain(samplerate=samplerate, text_units=text_units, add_silence=False,
                                      random_track_mixing=False,
                                      space_token_only=True, return_name=True)


pickle_in = open('dicts/idx2cmu_phoneme.pickle', 'rb')
idx2symbol = pickle.load(pickle_in)

# go through data sets and save alignment path
# use clean vocals for training and validation set
# use mixtures for test set

# make dirs to save alignments
base_path = 'evaluation/{}/musdb_alignments/'.format(tag)
if not os.path.isdir(base_path):
    os.makedirs(base_path)
    os.makedirs(os.path.join(base_path, 'train'))
    os.makedirs(os.path.join(base_path, 'val'))
    os.makedirs(os.path.join(base_path, 'test'))


# TEST SET
for idx in range(len(test_set)):
    track = test_set[idx]

    mix = track['mix'].unsqueeze(dim=0)
    true_vocals = track['vocals'].unsqueeze(dim=0)
    true_accompaniment = track['accompaniment']
    text = track['text'].unsqueeze(dim=0)
    name = track['name'][2:]

    with torch.no_grad():
        vocals_estimate, alphas, scores = unmix((mix.to(device), text.to(device)))

    optimal_path = optimal_alignment_path(scores, mode='max_numpy', init=2000)

    optimal_path = torch.from_numpy(optimal_path).type(torch.float32)

    torch.save(optimal_path, os.path.join(base_path, 'test', name + '.pt'))
    print(idx, name)


# TRAIN SET
for idx in range(len(train_set)):
    data = train_set[idx]

    mix = data[0].unsqueeze(dim=0)  # mix
    true_vocals = data[1].unsqueeze(dim=0)  # vocals
    text = data[2].unsqueeze(dim=0)  # text
    name = data[3]  # track name

    with torch.no_grad():
        vocals_estimate, alphas, scores = unmix((true_vocals.to(device), text.to(device)))

    optimal_path = optimal_alignment_path(scores, mode='max_numpy', init=2000)
    optimal_path = torch.from_numpy(optimal_path).type(torch.float32)
    torch.save(optimal_path, os.path.join(base_path, 'train', name + '.pt'))
    print(idx, name)

# VAL SET
for idx in range(len(val_set)):
    data = val_set[idx]

    mix = data[0].unsqueeze(dim=0)  # mix
    true_vocals = data[1].unsqueeze(dim=0)  # vocals
    text = data[2].unsqueeze(dim=0)  # text
    name = data[3]  # track name

    with torch.no_grad():
        vocals_estimate, alphas, scores = unmix((true_vocals.to(device), text.to(device)))

    optimal_path = optimal_alignment_path(scores, mode='max_numpy', init=2000)
    optimal_path = torch.from_numpy(optimal_path).type(torch.float32)
    torch.save(optimal_path, os.path.join(base_path, 'val', name + '.pt'))
    print(idx, name)
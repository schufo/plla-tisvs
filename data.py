"""This file is a modified version of https://github.com/sigsep/open-unmix-pytorch/blob/master/openunmix/data.py
It contains the dataset classes used for training and testing"""

from utils import load_audio, load_info
from pathlib import Path
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import librosa as lb

import glob
import argparse
import random
import musdb
import torch
import tqdm
import os
import pickle
import csv


class MixSNR(object):
    """
    ..........
    """

    def __init__(self):
        pass

    def __call__(self, target_snr, vocals, accompaniment, vocals_start=0, vocals_length=None):

        if vocals_length is None:
            vocals_length = vocals.size()[1]

        vocals_energy = torch.sum(torch.mean(vocals, dim=0) ** 2)
        # compute accompaniment energy only on part where vocals are present
        acc_energy = torch.sum(torch.mean(accompaniment[:, vocals_start: vocals_start + vocals_length], dim=0) ** 2)

        if acc_energy > 0.1 and vocals_energy > 0.1:
            snr_current = 10 * torch.log10(vocals_energy / acc_energy)
            snr_difference = target_snr - snr_current
            scaling = (10 ** (snr_difference / 10))
            vocals_scaled = vocals * torch.sqrt(scaling)
            mix = vocals_scaled + accompaniment
            mix_max = abs(mix).max()
            mix = mix / mix_max
            vocals_scaled = vocals_scaled / mix_max
            accompaniment = accompaniment / mix_max
            return mix, vocals_scaled, accompaniment
        else:
            mix = vocals + accompaniment
            mix_max = abs(mix).max()
            mix = mix / mix_max
            vocals = vocals / mix_max
            accompaniment = accompaniment / mix_max
            return mix, vocals, accompaniment


class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio


def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g


def _augment_channelswap(audio):
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio


def load_datasets(parser, args):
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """

    if args.dataset == 'musdb_lyrics':
        parser.add_argument('--samplerate', type=int, default=16000)
        parser.add_argument('--text-units', type=str, default='cmu_phonemes')
        parser.add_argument('--n', action='store_false', default=True)
        parser.add_argument('--no-x', action='store_false', default=True)
        parser.add_argument('--s', action='store_false', default=True)
        parser.add_argument('--no-d', action='store_false', default=True)
        parser.add_argument('--no-random-track-mix', action='store_true', default=False)
        parser.add_argument('--red-phonemes', type=str, default=None)
        parser.add_argument('--space-token-only', action='store_true', default=False)  # space token at start and end (no $)
        parser.add_argument('--add-silence', action='store_true', default=False)
        parser.add_argument('--snr', type=float, default=None)
        args = parser.parse_args()

        train_dataset = MUSDBLyricsDataTrain(samplerate=args.samplerate,
                                             text_units=args.text_units,
                                             space_token_only=args.space_token_only,
                                             add_silence=args.add_silence,
                                             n=args.n,
                                             x=args.no_x,
                                             s=args.s,
                                             d=args.no_d,
                                             random_track_mixing=(not args.no_random_track_mix),
                                             alignment_from=args.alignment_from,
                                             fake_alignment=args.fake_alignment,
                                             target=args.target,
                                             mix_snr=args.snr)

        valid_dataset = MUSDBLyricsDataVal(samplerate=args.samplerate,
                                           text_units=args.text_units,
                                           space_token_only=args.space_token_only,
                                           n=args.n,
                                           x=args.no_x,
                                           s=args.s,
                                           d=args.no_d,
                                           alignment_from=args.alignment_from,
                                           fake_alignment=args.fake_alignment,
                                           target=args.target)

    elif args.dataset == 'timit_music':
        parser.add_argument('--samplerate', type=int, default=16000)  # add sample rate to config
        parser.add_argument('--text-units', type=str, default='cmu_phonemes')
        parser.add_argument('--red-phonemes', type=str, default=None)
        parser.add_argument('--fixed-length-timit', action='store_true', default=False)  # all exmaples 8.2s (much silence)
        parser.add_argument('--space-token-only', action='store_true', default=False)  # space token at start and end (no $)

        args = parser.parse_args()

        train_dataset = TIMITMusicTrain(args.text_units, fixed_length=args.fixed_length_timit,
                                        space_token_only=args.space_token_only)

        valid_dataset = TIMITMusicVal(args.text_units, fixed_length=args.fixed_length_timit,
                                      space_token_only=args.space_token_only)

    elif args.dataset == 'blended':
        parser.add_argument('--samplerate', type=int, default=16000)  # add sample rate to config
        parser.add_argument('--text-units', type=str, default='cmu_phonemes')
        parser.add_argument('--red-phonemes', type=str, default=None)
        parser.add_argument('--fixed-length-timit', action='store_true', default=False)  # all exmaples 8.2s (much silence)
        parser.add_argument('--space-token-only', action='store_true', default=False)  # space token at start and end (no $)
        parser.add_argument('--add-silence', action='store_true', default=False)

        parser.add_argument('--n', action='store_false', default=True)
        parser.add_argument('--x', action='store_false', default=True)
        parser.add_argument('--s', action='store_false', default=True)
        parser.add_argument('--d', action='store_false', default=True)
        parser.add_argument('--no-random-track-mix', action='store_true', default=False)
        parser.add_argument('--speech-examples', type=int, default=4320)  # number of speech examples to be added

        args = parser.parse_args()

        singing_train_data = MUSDBLyricsDataTrain(samplerate=args.samplerate,
                                             text_units=args.text_units,
                                             space_token_only=args.space_token_only,
                                             add_silence=args.add_silence,
                                             n=args.n,
                                             x=args.x,
                                             s=args.s,
                                             d=args.d,
                                             random_track_mixing=(not args.no_random_track_mix))

        speech_train_data = TIMITMusicTrain(args.text_units, space_token_only=args.space_token_only,
                                            fixed_length=args.fixed_length_timit)

        indices = list(range(args.speech_examples))
        selected_speech_data = torch.utils.data.Subset(speech_train_data, indices)

        train_dataset = torch.utils.data.ConcatDataset([singing_train_data, selected_speech_data])

        valid_dataset = MUSDBLyricsDataVal(samplerate=args.samplerate,
                                           text_units=args.text_units,
                                           space_token_only=args.space_token_only,
                                           n=args.n,
                                           x=args.x,
                                           s=args.s,
                                           d=args.d)

    elif args.dataset == 'nus':
        parser.add_argument('--snr', type=float, default=0.)
        parser.add_argument('--samplerate', type=int, default=16000)
        parser.add_argument('--text-units', type=str, default='cmu_phonemes')
        parser.add_argument('--space-token-only', action='store_true', default=False)  # space token at start and end (no $)
        args = parser.parse_args()

        train_dataset = NUS(snr=args.snr, training=True)

        valid_data = MUSDBLyricsDataVal(samplerate=args.samplerate,
                                           text_units=args.text_units,
                                           space_token_only=args.space_token_only)

        # pick 48 examples from validation dataset
        indices = list(torch.randint(0, len(valid_data), (48,)).numpy())
        valid_dataset = torch.utils.data.Subset(valid_data, indices)
        valid_dataset.sample_rate = args.samplerate
        valid_dataset.vocabulary_size = valid_data.vocabulary_size

    elif args.dataset == 'nus_train':
        parser.add_argument('--snr', type=float, default=0.)
        parser.add_argument('--samplerate', type=int, default=16000)
        parser.add_argument('--text-units', type=str, default='cmu_phonemes')
        parser.add_argument('--space-token-only', action='store_true', default=False)  # space token at start and end (no $)
        args = parser.parse_args()

        train_dataset = NUSTrain(snr=args.snr)

        valid_data = MUSDBLyricsDataVal(samplerate=args.samplerate,
                                        text_units=args.text_units,
                                        space_token_only=args.space_token_only,
                                        space_between_phonemes=False)

        # pick 48 examples from validation dataset
        indices = list(torch.randint(0, len(valid_data), (48,)).numpy())
        valid_dataset = torch.utils.data.Subset(valid_data, indices)
        valid_dataset.sample_rate = args.samplerate
        valid_dataset.vocabulary_size = valid_data.vocabulary_size

    return train_dataset, valid_dataset, args


class MUSDBLyricsDataTrain(torch.utils.data.Dataset):

    def __init__(self,
                 samplerate,
                 text_units,
                 add_silence=False,
                 random_track_mixing=True,
                 space_token_only=False,
                 n=True,
                 x=True,
                 s=True,
                 d=True,
                 transform=None,
                 return_name=False,
                 alignment_from=None,
                 fake_alignment=False,
                 target='vocals',
                 mix_snr=None):

        super(MUSDBLyricsDataTrain, self).__init__()

        self.sample_rate = samplerate
        self.mix_with_snr = MixSNR()
        self.text_units = text_units
        self.space_token_only = space_token_only
        self.add_silence = add_silence
        self.return_name = return_name
        self.alignment_from = alignment_from
        self.fake_alignment = fake_alignment  # if True, align last feature vector of H to all audio frames
        self.target = target
        self.mix_snr = mix_snr

        if alignment_from:
            self.path_to_attention_weights = 'evaluation/{}/musdb_alignments/train/'.format(alignment_from)

        if samplerate == 16000:
            self.data_set_root = '../Datasets/MUSDB_w_lyrics'
        elif samplerate == 44100:
            self.data_set_root = '../Datasets/MUSDB_w_lyrics441'

        self.random_track_mixing = random_track_mixing
        if self.random_track_mixing:
            pickle_in = open(os.path.join(self.data_set_root, 'train/train_accompaniments_12s.pickle'), 'rb')
            self.sources_12s_list = pickle.load(pickle_in)  # the file names are the same for each source

        if text_units == 'cmu_phonemes' or 'voice_activity':
            self.path_to_text = os.path.join(self.data_set_root, 'train/text_cmu_phonemes/')
            self.vocabulary_size = 44
        if text_units == 'ones':
            self.path_to_text = os.path.join(self.data_set_root, 'train/text_cmu_phonemes/')
            self.vocabulary_size = 44

        self.path_to_audio = os.path.join(self.data_set_root, 'train/audio')

        file_list = []

        if n:
            pickle_in = open(os.path.join(self.data_set_root, 'train/train_files_n.pickle'), 'rb')
            n_files = pickle.load(pickle_in)
            for file in n_files:
                file_list.append(file)
        if x:
            pickle_in = open(os.path.join(self.data_set_root, 'train/train_files_x.pickle'), 'rb')
            x_files = pickle.load(pickle_in)
            for file in x_files:
                file_list.append(file)
        if s:
            pickle_in = open(os.path.join(self.data_set_root, 'train/train_files_s.pickle'), 'rb')
            s_files = pickle.load(pickle_in)
            for file in s_files:
                file_list.append(file)
        if d:
            pickle_in = open(os.path.join(self.data_set_root, 'train/train_files_d.pickle'), 'rb')
            d_files = pickle.load(pickle_in)
            for file in d_files:
                file_list.append(file)

        self.file_dict = {idx: file for (idx, file) in enumerate(file_list)}

        self.transform = transform

    def __len__(self):
        return len(self.file_dict)

    def __getitem__(self, idx):

        file = self.file_dict[idx]

        if self.add_silence:
            self.space_token_only = True

        # text as index encoding with shape (nb_characters)
        text = torch.load(os.path.join(self.path_to_text, file[2:] + '.pt'))

        if self.space_token_only:
            # if no vocals in snippet the text is the space token only (index 3)
            if file[0] == 'x':
                text_ = torch.tensor([3])
            else:
                text_ = torch.ones((text.size()[0] + 2)) * 3 # add space token (index 3) to start and end
                text_[1:-1] = text
        else:
            text_ = torch.ones((text.size()[0] + 2)) * 2
            text_[1:-1] = text  # add random sound token ('%' with index 2) to start and end of text

        if self.text_units == 'ones':
            text_ = torch.ones_like(text_)
        if self.text_units == 'voice_activity':
            text_[text_ != 3] = 5

        vocals = torch.load(os.path.join(self.path_to_audio, 'vocals', file + '.pt'))

        vocals_start = 0
        vocals_signal_length = vocals.size()[1]

        if self.add_silence:
            vocal_length = vocals.size()[1] / self.sample_rate
            if vocal_length < 11:
                if torch.rand(1) > 0.5:
                    # at silence at end
                    vocals = torch.cat([vocals, torch.zeros((2, 11* self.sample_rate - vocals_signal_length))], dim=1)
                    vocals_start = 0
                else:
                    # add silence at start
                    vocals = torch.cat([torch.zeros((2, 11* self.sample_rate - vocals_signal_length)), vocals], dim=1)
                    vocals_start = 11* self.sample_rate - vocals_signal_length

        if self.target == 'accompaniment':
            accompaniment = torch.load(os.path.join(self.path_to_audio, 'accompaniment', file + '.pt'))

        elif not self.random_track_mixing:
            drums = torch.load(os.path.join(self.path_to_audio, 'drums', file + '.pt'))
            drums = _augment_gain(drums)
            bass = torch.load(os.path.join(self.path_to_audio, 'bass', file + '.pt'))
            bass = _augment_gain(bass)
            other = torch.load(os.path.join(self.path_to_audio, 'other', file + '.pt'))
            other = _augment_gain(other)
            accompaniment = drums + bass + other

        elif self.random_track_mixing:

            length_vocals = vocals.size()[1]
            drums_idx = torch.randint(high=len(self.sources_12s_list), size=(1,), dtype=torch.int)
            drums_file = self.sources_12s_list[drums_idx]
            drums = torch.load(os.path.join(self.path_to_audio, 'drums_12s', drums_file))
            drums = drums[:, :length_vocals].type(torch.float32)
            drums = _augment_gain(drums)
            bass_idx = torch.randint(high=len(self.sources_12s_list), size=(1,), dtype=torch.int)
            bass_file = self.sources_12s_list[bass_idx]
            bass = torch.load(os.path.join(self.path_to_audio, 'bass_12s', bass_file))
            bass = bass[:, :length_vocals].type(torch.float32)
            bass = _augment_gain(bass)
            other_idx = torch.randint(high=len(self.sources_12s_list), size=(1,), dtype=torch.int)
            other_file = self.sources_12s_list[other_idx]
            other = torch.load(os.path.join(self.path_to_audio, 'other_12s', other_file))
            other = other[:, :length_vocals].type(torch.float32)
            other = _augment_gain(other)
            accompaniment = drums + bass + other

        if self.target == 'vocals':
            if self.mix_snr is None:
                vocals = _augment_gain(vocals, low=0.25, high=0.9)
                mix = vocals + accompaniment
            else:
                mix, vocals, accompaniment = self.mix_with_snr(self.mix_snr, vocals, accompaniment,
                                                               vocals_start=vocals_start,
                                                               vocals_length=vocals_signal_length)
            target_source = vocals
        elif self.target == 'accompaniment':
            mix = vocals + accompaniment
            target_source = accompaniment

        if self.alignment_from:
            attention_weights = torch.load(os.path.join(self.path_to_attention_weights, file[2:] + '.pt'))
            if self.fake_alignment:
                attention_weights = torch.zeros_like(attention_weights)
                attention_weights[:, -1] = 1
            return mix, target_source, text_, attention_weights
        elif self.return_name:
            return mix, target_source, text_, file[2:]
        else:
            return mix, target_source, text_


class MUSDBLyricsDataVal(torch.utils.data.Dataset):

    def __init__(self,
                 samplerate,
                 text_units,
                 space_token_only=False,
                 n=True,
                 x=True,
                 s=True,
                 d=True,
                 transform=None,
                 return_name=False,
                 alignment_from=None,
                 fake_alignment=False,
                 target='vocals'):

        super(MUSDBLyricsDataVal, self).__init__()

        self.sample_rate = samplerate
        self.text_units = text_units
        self.space_token_only = space_token_only
        self.return_name = return_name
        self.alignment_from = alignment_from
        self.fake_alignment = fake_alignment
        self.target = target

        if alignment_from:
            self.path_to_attention_weights = 'evaluation/{}/musdb_alignments/val/'.format(alignment_from)

        if samplerate == 16000:
            self.data_set_root = '../Datasets/MUSDB_w_lyrics'
        elif samplerate == 44100:
            self.data_set_root = '../Datasets/MUSDB_w_lyrics441'

        if text_units == 'cmu_phonemes' or 'voice_activity':
            self.path_to_text = os.path.join(self.data_set_root, 'val/text_cmu_phonemes/')
            self.vocabulary_size = 44
        if text_units == 'ones':
            self.path_to_text = os.path.join(self.data_set_root, 'val/text_cmu_phonemes/')
            self.vocabulary_size = 44

        self.path_to_audio = os.path.join(self.data_set_root, 'val/audio')

        file_list = []

        if n:
            pickle_in = open(os.path.join(self.data_set_root, 'val/val_files_n.pickle'), 'rb')
            n_files = pickle.load(pickle_in)
            for file in n_files:
                file_list.append(file)
        if x:
            pickle_in = open(os.path.join(self.data_set_root, 'val/val_files_x.pickle'), 'rb')
            x_files = pickle.load(pickle_in)
            for file in x_files:
                file_list.append(file)
        if s:
            pickle_in = open(os.path.join(self.data_set_root, 'val/val_files_s.pickle'), 'rb')
            s_files = pickle.load(pickle_in)
            for file in s_files:
                file_list.append(file)
        if d:
            pickle_in = open(os.path.join(self.data_set_root, 'val/val_files_d.pickle'), 'rb')
            d_files = pickle.load(pickle_in)
            for file in d_files:
                file_list.append(file)

        self.file_dict = {idx: file for (idx, file) in enumerate(file_list)}

        self.transform = transform

    def __len__(self):
        return len(self.file_dict)

    def __getitem__(self, idx):

        file = self.file_dict[idx]

        # text as index encoding
        text = torch.load(os.path.join(self.path_to_text, file[2:] + '.pt'))
        if self.space_token_only:
            # if no vocals in snippet the text is the space token only (index 3)
            if file[0] == 'x':
                text_ = torch.tensor([3])
            else:
                text_ = torch.ones((text.size()[0] + 2)) * 3 # add space token (index 3) to start and end
                text_[1:-1] = text
        else:
            text_ = torch.ones((text.size()[0] + 2)) * 2
            text_[1:-1] = text  # add random sound token ('%' with index 2) to start and end of text

        if self.text_units == 'ones':
            text_ = torch.ones_like(text_)
        if self.text_units == 'voice_activity':
            text_[text_ != 3] = 5

        # vocals in time domain
        vocals = torch.load(os.path.join(self.path_to_audio, 'vocals', file + '.pt'))
        mix = torch.load(os.path.join(self.path_to_audio, 'mix', file + '.pt'))

        if self.target == 'vocals':
            target_source = vocals
        elif self.target == 'accompaniment':
            target_source = mix - vocals

        if self.alignment_from:
            attention_weights = torch.load(os.path.join(self.path_to_attention_weights, file[2:] + '.pt'))
            if self.fake_alignment:
                attention_weights = torch.zeros_like(attention_weights)
                attention_weights[:, -1] = 1
            return mix, target_source, text_, attention_weights
        elif self.return_name:
            return mix, target_source, text_, file[2:]
        else:
            return mix, target_source, text_


class MUSDBLyricsDataTest(torch.utils.data.Dataset):

    def __init__(self,
                 samplerate,
                 text_units,
                 train_mode=False,
                 space_token_only=False,
                 n=True,
                 x=True,
                 s=True,
                 d=True,
                 transform=None,
                 alignment_from=None,
                 fake_alignment=False,
                 mix_snr=None):
        
        super(MUSDBLyricsDataTest, self).__init__()

        self.sample_rate = samplerate
        self.text_units = text_units
        self.space_token_only = space_token_only
        self.alignment_from = alignment_from
        self.fake_alignment = fake_alignment
        self.mix_snr = mix_snr

        if alignment_from:
            self.path_to_attention_weights = 'evaluation/{}/musdb_alignments/test/'.format(alignment_from)

        if samplerate == 16000:
            self.data_set_root = '../Datasets/MUSDB_w_lyrics'
        elif samplerate == 44100:
            self.data_set_root = '../Datasets/MUSDB_w_lyrics441'

        if text_units == 'characters':
            self.path_to_text = os.path.join(self.data_set_root, 'test/text/')
            self.vocabulary_size = 32
        if text_units == 'cmu_phonemes' or 'voice_activity':
            self.path_to_text = os.path.join(self.data_set_root, 'test/text_cmu_phonemes/')
            self.vocabulary_size = 44
        if text_units == 'ones':
            self.path_to_text = os.path.join(self.data_set_root, 'test/text_cmu_phonemes/')
            self.vocabulary_size = 44

        self.path_to_audio = os.path.join(self.data_set_root, 'test/audio')

        file_list = []

        if n:
            pickle_in = open(os.path.join(self.data_set_root, 'test/test_files_n.pickle'), 'rb')
            n_files = pickle.load(pickle_in)
            for file in n_files:
                file_list.append(file)
        if x:
            pickle_in = open(os.path.join(self.data_set_root, 'test/test_files_x.pickle'), 'rb')
            x_files = pickle.load(pickle_in)
            for file in x_files:
                file_list.append(file)
        if s:
            pickle_in = open(os.path.join(self.data_set_root, 'test/test_files_s.pickle'), 'rb')
            s_files = pickle.load(pickle_in)
            for file in s_files:
                file_list.append(file)
        if d:
            pickle_in = open(os.path.join(self.data_set_root, 'test/test_files_d.pickle'), 'rb')
            d_files = pickle.load(pickle_in)
            for file in d_files:
                file_list.append(file)

        self.file_dict = {idx: file for (idx, file) in enumerate(file_list)}

        self.mix_with_snr = MixSNR()
        self.transform = transform

    def __len__(self):
        return len(self.file_dict)

    def __getitem__(self, idx):

        file = self.file_dict[idx]

        # text as index encoding
        text = torch.load(os.path.join(self.path_to_text, file[2:] + '.pt'))
        if self.space_token_only:
            # if no vocals in snippet the text is the space token only (index 3)
            if file[0] == 'x':
                text_ = torch.tensor([3])
            else:
                text_ = torch.ones((text.size()[0] + 2)) * 3 # add space token (index 3) to start and end
                text_[1:-1] = text
        else:
            text_ = torch.ones((text.size()[0] + 2)) * 2
            text_[1:-1] = text  # add random sound token ('%' with index 2) to start and end of text

        if self.text_units == 'ones':
            text_ = torch.ones_like(text_)
        if self.text_units == 'voice_activity':
            text_[text_ != 3] = 5

        # time domain sources with shape (2, nb_timesteps)
        vocals = torch.load(os.path.join(self.path_to_audio, 'vocals', file + '.pt'))
        mix = torch.load(os.path.join(self.path_to_audio, 'mix', file + '.pt'))
        acc = torch.load(os.path.join(self.path_to_audio, 'accompaniments', file + '.pt'))

        if self.alignment_from:
            attention_weights = torch.load(os.path.join(self.path_to_attention_weights, file[2:] + '.pt'))
            if self.fake_alignment:
                attention_weights = torch.zeros_like(attention_weights)
                attention_weights[:, -1] = 1
            if self.mix_snr is not None:
                mix, vocals, acc = self.mix_with_snr(self.mix_snr, vocals, acc)
            test_track = {'name': file, 'mix': mix, 'vocals': vocals, 'accompaniment': acc, 'text': text_.type(torch.float32),
                          'attention_weights': attention_weights}
        else:
            if self.mix_snr is not None:
                mix, vocals, acc = self.mix_with_snr(self.mix_snr, vocals, acc)
            test_track = {'name': file, 'mix': mix, 'vocals': vocals, 'accompaniment': acc, 'text': text_.type(torch.float32)}

        return test_track


class TIMITMusicTrain(torch.utils.data.Dataset):

    def __init__(self,
                 text_units,
                 fixed_length=False,
                 space_token_only=False):

        super(TIMITMusicTrain).__init__()

        # set to true if all files should have fixed length of 8.5 s (=> much silence in true vocals)
        self.fixed_length = fixed_length
        self.text_units = text_units
        self.space_token_only = space_token_only

        self.sample_rate = 16000  # TIMIT is only available at 16 kHz

        if text_units == 'cmu_phonemes':
            self.path_to_text_sequences = '../Datasets/TIMIT/cmu_phoneme_sequences_idx_open_unmix/train'
            self.vocabulary_size = 44
        if text_units == 'ones':
            self.path_to_text_sequences = '../Datasets/TIMIT/cmu_phoneme_sequences_idx_open_unmix/train'
            self.vocabulary_size = 44


        # music related
        path_to_music = '../Datasets/Music_instrumentals/train/torch_snippets'
        self.list_of_music_files = sorted([f for f in glob.glob(path_to_music + "/*.pt", recursive=True)])

        self.mix_with_snr = MixSNR()

    def __len__(self):
        return 4320  # number of TIMIT utterances assigned to training set

    def __getitem__(self, idx):

        # get speech file
        speech = torch.load('../Datasets/TIMIT/TIMIT_torch/train/{}.pt'.format(idx))

        # randomly choose a music file from list and load it
        music_idx = torch.randint(low=0, high=len(self.list_of_music_files), size=(1,))
        music_file = self.list_of_music_files[music_idx]
        music = torch.load(music_file)

        if self.fixed_length:
            # pad the speech signal to same length as music
            speech_len = speech.size()[1]
            music_len = music.size()[1]
            padding_at_start = int(torch.randint(0, music_len - speech_len, size=(1,)))
            padding_at_end = music_len - padding_at_start - speech_len
            speech_padded = np.pad(array=speech.numpy(), pad_width=((0, 0), (padding_at_start, padding_at_end)),
                               mode='constant', constant_values=0)

        else:
            speech_len = speech.size()[1]
            music_len = music.size()[1]
            max_pad = min((music_len - speech_len) // 2, self.sample_rate)
            padding_at_start = int(torch.randint(0, max_pad, size=(1,)))
            padding_at_end = int(torch.randint(0, max_pad, size=(1,)))
            speech_padded = np.pad(array=speech.numpy(), pad_width=((0, 0), (padding_at_start, padding_at_end)),
                                   mode='constant', constant_values=0)
            music = music[:, 0:speech_padded.shape[1]]

        text = torch.load(os.path.join(self.path_to_text_sequences, '{}.pt'.format(idx)))
        if self.space_token_only:
            # put space token instead of silence token at start and end of text sequence
            # this option should not be used for pre-training on speech,
            # otherwise the alinment will not be learned
            # when training on MUSDB and added spech examples, this option can be selected
            # (training on MUSDB without pre-training and using the silence token
            # does not enable learning the alignment)
            text[0] = 3
            text[-1] = 3

        if self.text_units == 'ones':
            text = torch.ones_like(text)

        target_snr = torch.rand(size=(1,)) * (-8)
        mix, speech, music = self.mix_with_snr(target_snr, torch.from_numpy(speech_padded).type(torch.float32),
                                        music, padding_at_start, speech_len)

        return mix, speech, text


class TIMITMusicVal(torch.utils.data.Dataset):

    def __init__(self,
                 text_units,
                 fixed_length=False,
                 space_token_only=False):

        super(TIMITMusicVal).__init__()

        self.fixed_length = fixed_length
        self.text_units = text_units
        self.space_token_only = space_token_only

        self.sample_rate = 16000  # TIMIT is only available at 16 kHz

        if text_units == 'cmu_phonemes':
            self.path_to_text_sequences = '../Datasets/TIMIT/cmu_phoneme_sequences_idx_open_unmix/val'
            self.vocabulary_size = 44
        if text_units == 'ones':
            self.path_to_text_sequences = '../Datasets/TIMIT/cmu_phoneme_sequences_idx_open_unmix/val'
            self.vocabulary_size = 44

        # music related
        path_to_music = '../Datasets/Music_instrumentals/val/torch_snippets'
        self.list_of_music_files = sorted([f for f in glob.glob(path_to_music + "/*.pt", recursive=True)])

        self.mix_with_snr = MixSNR()

    def __len__(self):
        return 240  # number of TIMIT utterances assigned to val set

    def __getitem__(self, idx):
        # get speech file
        speech = torch.load('../Datasets/TIMIT/TIMIT_torch/val/{}.pt'.format(idx))

        # randomly choose a music file from list and load a snippet with speech length + x
        music_file = self.list_of_music_files[idx]
        music = torch.load(music_file)

        if self.fixed_length:
            # pad the speech signal to same length as music
            speech_len = speech.size()[1]
            music_len = music.size()[1]
            padding_at_start = int(torch.randint(0, music_len - speech_len, size=(1,)))
            padding_at_end = music_len - padding_at_start - speech_len
            speech_padded = np.pad(array=speech.numpy(), pad_width=((0, 0), (padding_at_start, padding_at_end)),
                               mode='constant', constant_values=0)

        else:
            speech_len = speech.size()[1]
            music_len = music.size()[1]
            max_pad = min((music_len - speech_len) // 2, self.sample_rate)
            padding_at_start = int(torch.randint(0, max_pad, size=(1,)))
            padding_at_end = int(torch.randint(0, max_pad, size=(1,)))
            speech_padded = np.pad(array=speech.numpy(), pad_width=((0, 0), (padding_at_start, padding_at_end)),
                                   mode='constant', constant_values=0)
            music = music[:, 0:speech_padded.shape[1]]

        text = torch.load(os.path.join(self.path_to_text_sequences, '{}.pt'.format(idx)))
        if self.space_token_only:
            # put space token instead of silence token at start and end of text sequence
            text[0] = 3
            text[-1] = 3
        if self.text_units == 'ones':
            text = torch.ones_like(text)

        target_snr = -5
        mix, speech, music = self.mix_with_snr(target_snr, torch.from_numpy(speech_padded).type(torch.float32),
                                        music, padding_at_start, speech_len)

        return mix, speech, text


class TIMITMusicTest(torch.utils.data.Dataset):

    def __init__(self,
                 text_units,
                 fixed_length=False,
                 space_token_only=False):

        super(TIMITMusicTest).__init__()

        self.fixed_length = fixed_length
        self.text_units = text_units
        self.space_token_only = space_token_only

        self.sample_rate = 16000  # TIMIT is only available at 16 kHz

        if text_units == 'cmu_phonemes':
            self.path_to_text_sequences = '../Datasets/TIMIT/cmu_phoneme_sequences_idx_open_unmix/test'
            self.vocabulary_size = 44
        if text_units == 'ones':
            self.path_to_text_sequences = '../Datasets/TIMIT/cmu_phoneme_sequences_idx_open_unmix/test'
            self.vocabulary_size = 44

        # music related
        path_to_music = '../Datasets/Music_instrumentals/test/torch_snippets'
        self.list_of_music_files = sorted([f for f in glob.glob(path_to_music + "/*.pt", recursive=True)])

        self.mix_with_snr = MixSNR()

    def __len__(self):
        return 1344  # number of TIMIT utterances assigned to val set

    def __getitem__(self, idx):

        # get speech file
        speech = torch.load('../Datasets/TIMIT/TIMIT_torch/test/{}.pt'.format(idx))

        # randomly choose a music file from list and load a snippet with speech length + x
        music_idx = idx % len(self.list_of_music_files)
        music_file = self.list_of_music_files[music_idx]

        music = torch.load(music_file)

        if self.fixed_length:
            # pad the speech signal to same length as music
            speech_len = speech.size()[1]
            music_len = music.size()[1]
            padding_at_start = int(torch.randint(0, music_len - speech_len, size=(1,)))
            padding_at_end = music_len - padding_at_start - speech_len
            speech_padded = np.pad(array=speech.numpy(), pad_width=((0, 0), (padding_at_start, padding_at_end)),
                                mode='constant', constant_values=0)
        else:
            speech_len = speech.size()[1]
            music_len = music.size()[1]
            max_pad = min((music_len - speech_len) // 2, self.sample_rate)
            padding_at_start = int(torch.randint(0, max_pad, size=(1,)))
            print("padding at start", padding_at_start)
            padding_at_end = int(torch.randint(0, max_pad, size=(1,)))
            speech_padded = np.pad(array=speech.numpy(), pad_width=((0, 0), (padding_at_start, padding_at_end)),
                                   mode='constant', constant_values=0)
            music = music[:, 0:speech_padded.shape[1]]

        text = torch.load(os.path.join(self.path_to_text_sequences, '{}.pt'.format(idx)))
        if self.space_token_only:
            # put space token instead of silence token at start and end of text sequence
            text[0] = 3
            text[-1] = 3
        if self.text_units == 'ones':
            text = torch.ones_like(text)

        target_snr = -5
        mix, speech, music = self.mix_with_snr(target_snr, torch.from_numpy(speech_padded).type(torch.float32),
                                        music, padding_at_start, speech_len)

        name = music_file.split('/')[-1][:-3] + str(idx)

        test_track = {'name': name, 'mix': mix, 'vocals': speech, 'accompaniment': music, 'text': text}

        return test_track


class Hansen(torch.utils.data.Dataset):

    def __init__(self):

        super(Hansen).__init__()

        self.sample_rate = 16000
        pickle_in = open('dicts/cmu_phoneme2idx.pickle', 'rb')
        cmu_phoneme2idx = pickle.load(pickle_in)
        self.idx2cmu_phoneme = {value:key for key, value in cmu_phoneme2idx.items()}


        self.audio_files = sorted(glob.glob('../Datasets/HansensDataset/mixed/pytorch_16k/*.pt'))
        self.text_onset_files = sorted(glob.glob('../Datasets/HansensDataset/wordonsets/*.tsv'))
        self.text_phoneme_idx_files = sorted(glob.glob('../Datasets/HansensDataset/lyrics_phoneme_idx_pytorch/*.pt'))

    def __len__(self):
        return  len(self.audio_files)

    def __getitem__(self, idx):

        audio_file = self.audio_files[idx]
        text_onset_file = self.text_onset_files[idx]
        text_phoneme_idx_file = self.text_phoneme_idx_files[idx]

        name = audio_file.split('/')[-1][:-3]

        audio = torch.load(audio_file)
        text_phoneme_idx = torch.load(text_phoneme_idx_file)
        text_phoneme_idx[0] = 3
        text_phoneme_idx[-1] = 3

        text_symb = [self.idx2cmu_phoneme[idx] for idx in list(text_phoneme_idx.numpy())]

        true_onsets = []
        true_offsets = []

        with open(text_onset_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                onset = np.float32((row[0].replace(',', '.')))
                true_onsets.append(onset)
                offset = np.float32(row[1].replace(',', '.'))
                true_offsets.append(offset)

        test_track = {'name': name, 'audio': audio, 'text_phoneme_symbols': text_symb,
                      'text_phoneme_idx': text_phoneme_idx,
                      'true_onsets': true_onsets, 'true_offsets': true_offsets}

        return test_track

class Jamendo(torch.utils.data.Dataset):

    def __init__(self):

        super(Jamendo).__init__()

        self.sample_rate = 16000
        pickle_in = open('dicts/cmu_phoneme2idx.pickle', 'rb')
        cmu_phoneme2idx = pickle.load(pickle_in)
        self.idx2cmu_phoneme = {value:key for key, value in cmu_phoneme2idx.items()}

        self.audio_files = sorted(glob.glob('../Datasets/jamendolyrics/audio_pytorch_16k/*.pt'))
        self.text_onset_files = sorted(glob.glob('../Datasets/jamendolyrics/annotations/*.txt'))
        self.text_phoneme_idx_files = sorted(glob.glob('../Datasets/jamendolyrics/lyrics_phoneme_idx_pytorch/*.pt'))

    def __len__(self):
        return  len(self.audio_files)

    def __getitem__(self, idx):

        audio_file = self.audio_files[idx]
        text_onset_file = self.text_onset_files[idx]
        text_phoneme_idx_file = self.text_phoneme_idx_files[idx]

        name = audio_file.split('/')[-1][:-3]

        audio = torch.load(audio_file)
        text_phoneme_idx = torch.load(text_phoneme_idx_file)

        text_symb = [self.idx2cmu_phoneme[idx] for idx in list(text_phoneme_idx.numpy())]

        true_onsets = []

        with open(text_onset_file) as file:
            lines = file.readlines()
            for line in lines:
                onset = np.float32((line.replace('\n', '')))
                true_onsets.append(onset)

        test_track = {'name': name, 'audio': audio, 'text_phoneme_symbols': text_symb,
                      'text_phoneme_idx': text_phoneme_idx,
                      'true_onsets': true_onsets}

        return test_track

class NUS(torch.utils.data.Dataset):

    def __init__(self, acapella=False, snr=5, training=False):
        """

        Args:
            acapella: If True, the vocals will not be mixed with music
            snr: mixing SNR in dB if not acapella
            training: If True, the examples will be cut into ~6 second long segments which are randomly selected
        """
        super(NUS).__init__()

        self.sample_rate = 16000
        self.acapella = acapella
        self.snr = snr  # ratio to mix vocals and music
        self.training = training

        self.audio_files = sorted(glob.glob('../Datasets/nus-smc-corpus_48/audio_pytorch_16k/*.pt'))
        self.text_onset_files = sorted(glob.glob('../Datasets/nus-smc-corpus_48/*/sing/*.txt'))
        self.text_phoneme_idx_files = sorted(glob.glob('../Datasets/nus-smc-corpus_48/lyrics_phoneme_idx_pytorch/*.pt'))
        if not acapella:
            # we excluded 'PR - Oh No' and 'Hollow_Ground_-_Ill_Fate' because they were shorter than the vocals
            self.music_files = sorted(glob.glob('../Datasets/MUSDB_accompaniments/test/whole_songs/*.pt'))
            self.mix_snr = MixSNR()
    def __len__(self):
        return  len(self.audio_files)

    def __getitem__(self, idx):

        audio_file = self.audio_files[idx]
        text_onset_file = self.text_onset_files[idx]
        text_phoneme_idx_file = self.text_phoneme_idx_files[idx]

        name = audio_file.split('/')[-1][:-3]

        audio = torch.load(audio_file)
        if not self.acapella:
            music = torch.load(self.music_files[idx])
            vocals_length = audio.size(0)
            music = music[:vocals_length]
            audio, vocals, music = self.mix_snr(self.snr, audio.unsqueeze(0), music.unsqueeze(0))
            # audio = vocals + music
            audio = audio.squeeze(0)
            vocals = vocals.squeeze(0)
            name = name + self.music_files[idx].split('/')[-1][:-3]

        text_phoneme_idx = torch.load(text_phoneme_idx_file)

        true_onsets = []
        with open(text_onset_file) as file:
            lines = file.readlines()
            for line in lines:
                annotation = line.split(' ')
                if annotation[-1].replace('\n', '') == 'sp':
                    # ignore duration-less word boundary annotation
                    continue
                onset = np.float32((annotation[0]))
                true_onsets.append(onset)

        if name[:6] == 'ADIZ09':
            # correct wrong annotations for training
            true_onsets = [x + 2.8 for x in true_onsets]

        if self.training:
            # cut out a random segment of ~ 6 seconds length

            # find the latest onset which is 6 seconds before the last onset
            max_start = np.where(true_onsets <= true_onsets[-1] - 6)[0][-1]

            start_onset_idx = torch.randint(0, max_start, (1,))
            start_sample = int(true_onsets[start_onset_idx] * self.sample_rate)

            # find onset that is ~6 seconds after start onset
            end_onset_idx = np.where(true_onsets > true_onsets[start_onset_idx] + 6)[0][0]
            end_sample = int(true_onsets[end_onset_idx] * self.sample_rate)

            audio = audio[start_sample:end_sample][None, :]
            vocals = vocals[start_sample:end_sample][None, :]
            text = text_phoneme_idx[start_onset_idx:end_onset_idx]

            return audio, vocals, text

        test_track = {'name': name, 'audio': audio,
                      'text_phoneme_idx': text_phoneme_idx,
                      'true_onsets': true_onsets}

        return test_track


class NUSTrain(torch.utils.data.Dataset):

    def __init__(self, snr=0):

        super(NUS).__init__()

        self.sample_rate = 16000

        self.audio_files = sorted(glob.glob('../Datasets/nus-smc-corpus_48/audio_pytorch_16k/*.pt'))
        self.text_onset_files = sorted(glob.glob('../Datasets/nus-smc-corpus_48/*/sing/*.txt'))
        self.text_phoneme_idx_files = sorted(glob.glob('../Datasets/nus-smc-corpus_48/lyrics_phoneme_idx_pytorch/*.pt'))
        #self.music_files = sorted(glob.glob('../Datasets/MUSDB_accompaniments/test/whole_songs/*.pt'))
        path_to_music = '../Datasets/Music_instrumentals/train/torch_snippets'
        self.music_files = sorted([f for f in glob.glob(path_to_music + "/*.pt", recursive=True)])
        self.mix_snr = MixSNR()
        self.snr = snr

    def __len__(self):
        return  len(self.audio_files)

    def __getitem__(self, idx):

        audio_file = self.audio_files[idx]
        text_onset_file = self.text_onset_files[idx]
        text_phoneme_idx_file = self.text_phoneme_idx_files[idx]
        name = audio_file.split('/')[-1][:-3]
        vocals = torch.load(audio_file)
        text_phoneme_idx = torch.load(text_phoneme_idx_file)

        # load a music signal
        musix_idx = torch.randint(0, len(self.music_files), (1,))
        music = torch.load(self.music_files[musix_idx])[0, :]
        music_length = music.size(0)


        true_onsets = []
        with open(text_onset_file) as file:
            lines = file.readlines()
            for line in lines:
                annotation = line.split(' ')
                if annotation[-1].replace('\n', '') == 'sp':
                    # ignore duration-less word boundary annotation
                    continue
                onset = np.float32((annotation[0]))
                true_onsets.append(onset)

        if name[:6] == 'ADIZ09':
            # correct wrong annotations
            true_onsets = [x + 2.8 for x in true_onsets]

        # cut out a random segment of ~ 6 seconds length from vocals and cut text info accordingly
        # find the latest onset which is 6 seconds before the last onset
        max_start = np.where(true_onsets <= true_onsets[-1] - 6)[0][-1]

        start_onset_idx = torch.randint(0, max_start, (1,))
        start_sample = int(true_onsets[start_onset_idx] * self.sample_rate)

        # find onset that is ~6 seconds after start onset
        end_onset_idx = np.where(true_onsets > true_onsets[start_onset_idx] + 6)[0][0]
        end_sample = int(true_onsets[end_onset_idx] * self.sample_rate)

        vocals = vocals[start_sample:end_sample]
        text = text_phoneme_idx[start_onset_idx:end_onset_idx]

        text = torch.nn.functional.pad(text, pad=(1,1), mode='constant', value=1)  # add silence token at start and end

        vocals_length = vocals.size(0)
        if vocals_length < music_length:
            padding_at_start = int(torch.randint(0, music_length - vocals_length, size=(1,)))
            padding_at_end = music_length - padding_at_start - vocals_length
            vocals = torch.nn.functional.pad(vocals, pad=(padding_at_start, padding_at_end))
        elif vocals_length > music_length:
            padding_at_start = int(torch.randint(0, vocals_length - music_length, size=(1,)))
            padding_at_end = vocals_length - padding_at_start - music_length
            music = torch.nn.functional.pad(music, pad=(padding_at_start, padding_at_end))

        mix, vocals, music = self.mix_snr(self.snr, vocals.unsqueeze(0), music.unsqueeze(0),
                                          vocals_start=padding_at_start, vocals_length=vocals_length)

        name = name + self.music_files[idx].split('/')[-1][:-3]

        return mix, vocals, text


def collate_fn(sample_list):

    # make it work with and without text as third data set output

    batch_size = len(sample_list)
    mix_list = [sample_list[n][0].t() for n in range(batch_size)]
    vocals_list = [sample_list[n][1].t() for n in range(batch_size)]
    side_info_list = [sample_list[n][2].t() for n in range(batch_size)]

    # pad characters to length of longest character sequence in batch and stack them along dim=0
    mix = pad_sequence(mix_list, batch_first=True, padding_value=0)
    vocals = pad_sequence(vocals_list, batch_first=True, padding_value=0)
    side_info = pad_sequence(side_info_list, batch_first=True, padding_value=0)

    vocals = vocals.transpose(2, 1)
    mix = mix.transpose(2, 1)

    if len(sample_list[0]) == 4:
        M_max = max([sample_list[n][3].size(1) for n in range(batch_size)])
        N_max = max([sample_list[n][3].size(0) for n in range(batch_size)])
        attention_weights = torch.zeros((batch_size, N_max, M_max), dtype=torch.float32)
        for b in range(batch_size):
            attention_weights_b = sample_list[b][3]
            N, M = attention_weights_b.size()
            attention_weights[b, :N, :M] = attention_weights_b
        return mix, vocals, side_info.type(torch.float32), attention_weights
    else:
        # shape mix, vocals: (batch_size, nb_channels, nb_time_steps)
        # shape side_info: (batch_size, nb_elements)
        return mix, vocals, side_info



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Open Unmix Trainer')
    parser.add_argument(
        '--dataset', type=str, default="musdb",
        choices=[
            'musdb', 'aligned', 'sourcefolder',
            'trackfolder_var', 'trackfolder_fix', 'musdb_lyrics'
        ],
        help='Name of the dataset.'
    )

    parser.add_argument(
        '--root', type=str, help='root path of dataset'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help=('write out a fixed dataset of samples')
    )

    parser.add_argument('--target', type=str, default='vocals')

    # I/O Parameters
    parser.add_argument(
        '--seq-dur', type=float, default=5.0,
        help='Duration of <=0.0 will result in the full audio'
    )

    parser.add_argument('--batch-size', type=int, default=16)

    args, _ = parser.parse_known_args()
    train_dataset, valid_dataset, args = load_datasets(parser, args)

    # Iterate over training dataset
    total_training_duration = 0
    for k in tqdm.tqdm(range(len(train_dataset))):
        x, y = train_dataset[k]
        total_training_duration += x.shape[1] / train_dataset.sample_rate
        if args.save:
            import soundfile as sf
            sf.write(
                "test/" + str(k) + 'x.wav',
                x.detach().numpy().T,
                44100,
            )
            sf.write(
                "test/" + str(k) + 'y.wav',
                y.detach().numpy().T,
                44100,
            )

    print("Total training duration (h): ", total_training_duration / 3600)
    print("Number of train samples: ", len(train_dataset))
    print("Number of validation samples: ", len(valid_dataset))

    # iterate over dataloader
    train_dataset.seq_duration = args.seq_dur
    train_dataset.random_chunks = True

    if args.dataset == 'musdb_lyrics':
        train_sampler = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn,
        )

    else:
        train_sampler = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        )

    for x, y in tqdm.tqdm(train_sampler):
        pass

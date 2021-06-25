"""Save TIMIT audio files as PyTorch tensors"""

import numpy as np
import torch
import timit_utils as tu

import os

# insert the path to your TIMIT corpus here
corpus = tu.Corpus('../Datasets/TIMIT/TIMIT/TIMIT')

timit_training_set = corpus.train
timit_test_set = corpus.test

path_for_saving_audio = '../Datasets/TIMIT/TIMIT_torch'


def get_timit_train_sentence(idx):
    # the training set for this project comprises the first 4320 sentences of the TIMIT training partition
    # the persons are not sorted by dialect regions when accessed with .person_by_index, which ensures that all
    # dialect regions are represented in both the training and validation set
    person_idx = int(np.floor(idx / 10))
    person = timit_training_set.person_by_index(person_idx)
    sentence_idx = idx % 10
    sentence = person.sentence_by_index(sentence_idx)
    audio = sentence.raw_audio
    words = sentence.words_df.index.values
    word_onsets = sentence.words_df['start'].values
    phonemes = sentence.phones_df.index.values

    return audio, words, phonemes


def get_timit_val_sentence(idx):
    # the validation set for this project comprises the last 300 sentences of the TIMIT training partition excluding
    # the first two sentences per speaker (SA1, SA2) resulting in 240 utterance in total.
    # the persons are not sorted by dialect regions when accessed with .person_by_index, which ensures that all
    # dialect regions are represented in both the training and validation set
    person_idx = int(np.floor(idx / 8)) + 432
    person = timit_training_set.person_by_index(person_idx)
    sentence_idx = (idx % 8) + 2  # to ignore sentences 0 and 1 (SA1 and SA2), because they are also in training set
    sentence = person.sentence_by_index(sentence_idx)
    audio = sentence.raw_audio
    words = sentence.words_df.index.values
    word_onsets = sentence.words_df['start'].values
    phonemes = sentence.phones_df.index.values

    return audio, words, phonemes


def get_timit_test_sentence(idx):

    person_idx = int(np.floor(idx / 8))
    person = timit_test_set.person_by_index(person_idx)
    sentence_idx = (idx % 8) + 2  # to ignore sentences 0 and 1 (SA1 and SA2), because they are also in training set
    sentence = person.sentence_by_index(sentence_idx)
    audio = sentence.raw_audio
    words = sentence.words_df.index.values
    word_onsets = sentence.words_df['start'].values
    phonemes = sentence.phones_df.index.values

    return audio, words, phonemes


# training sentences
for idx in range(4320):

    speech, words, phonemes = get_timit_train_sentence(idx)

    speech_torch = torch.from_numpy(speech).type(torch.float32)

    # copy the single channel to make a two channel signal
    speech_torch = speech_torch.repeat(2, 1)

    speech_path = os.path.join(path_for_saving_audio, 'train', '{}.pt'.format(idx))
    torch.save(speech_torch, speech_path)


# validation sentences
for idx in range(240):

    speech, words, phonemes = get_timit_val_sentence(idx)

    speech_torch = torch.from_numpy(speech).type(torch.float32)

    # copy the single channel to make a two channel signal
    speech_torch = speech_torch.repeat(2, 1)

    speech_path = os.path.join(path_for_saving_audio, 'val', '{}.pt'.format(idx))
    torch.save(speech_torch, speech_path)


# test sentences
for idx in range(1344):

    speech, words, phonemes = get_timit_test_sentence(idx)

    speech_torch = torch.from_numpy(speech).type(torch.float32)

    # copy the single channel to make a two channel signal
    speech_torch = speech_torch.repeat(2, 1)

    speech_path = os.path.join(path_for_saving_audio, 'test', '{}.pt'.format(idx))
    torch.save(speech_torch, speech_path)
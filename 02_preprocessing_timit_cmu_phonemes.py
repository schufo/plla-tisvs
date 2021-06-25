"""
Save timit text as sequence of CMU phonemes with space character between words
represented by their indexes in the vocabulary. The term CMU phonemes refers to
APRAbet phonemes obtained at http://www.speech.cs.cmu.edu/tools/lextool.html.
Requires dicts/timit_word2cmu_phonemes.pickle.
"""
import numpy as np
import torch
import timit_utils as tu

import os
import pickle
import sys

# insert path to your TIMIT corpus here
corpus = tu.Corpus('../Datasets/TIMIT/TIMIT/TIMIT')

timit_training_set = corpus.train
timit_test_set = corpus.test


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
    # the validation set for this project comprises the last 300 sentences of the TIMIT training partition minus
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


# load timit_word2cmu_phonemes: a dictionary that translates the words of the TIMIT vocabulary to phonemes
pickle_in = open('dicts/timit_word2cmu_phonemes.pickle', 'rb')
timit_word2cmu_phonemes = pickle.load(pickle_in)

path_for_saving_text_cmu_phone_files = '../Datasets/TIMIT/cmu_phoneme_sequences_idx_open_unmix/'

# #: padding, $: silence, >: space, %: random sound, -: silence (no lyrics)
cmu_vocabulary = ['#', '$', '%', '>', '-', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G',
                  'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH',
                  'UW', 'V', 'W', 'Y', 'Z', 'ZH']

cmu_phoneme2idx = {}
for idx, phoneme in enumerate(cmu_vocabulary):
    cmu_phoneme2idx[phoneme] = idx
print(cmu_phoneme2idx)

idx2cmu_phoneme = {}
for idx, phoneme in enumerate(cmu_vocabulary):
    idx2cmu_phoneme[idx] = phoneme
print(idx2cmu_phoneme)

pickle_out = open(os.path.join('dicts', "cmu_vocabulary.pickle"), "wb")
pickle.dump(cmu_vocabulary, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join('dicts', "cmu_phoneme2idx.pickle"), "wb")
pickle.dump(cmu_phoneme2idx, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join('dicts', "idx2cmu_phoneme.pickle"), "wb")
pickle.dump(idx2cmu_phoneme, pickle_out)
pickle_out.close()


# -----------------------------------------------------------------------------------------------------------

# save each TIMIT training sentence as sequence of CMU phonemes in index representation
for idx in range(4320):

    speech, words, phonemes = get_timit_train_sentence(idx)

    phoneme_sequence = []

    for word in words:
        phones = timit_word2cmu_phonemes[word]
        for p in phones:
            phoneme_sequence.append(p)
        phoneme_sequence.append('>')

    # remove the last space token
    phoneme_sequence = phoneme_sequence[:-1]

    phoneme_idx = np.array([cmu_phoneme2idx[p] for p in phoneme_sequence])

    # add a silence token (idx=1) to start and end of character sequence
    phoneme_idx = np.pad(phoneme_idx, (1, 1), mode='constant', constant_values=1)

    phoneme_idx = torch.from_numpy(phoneme_idx)
    file_name = os.path.join(path_for_saving_text_cmu_phone_files, 'train', '{}.pt'.format(idx))
    torch.save(phoneme_idx, file_name)


# validation sentences
for idx in range(240):

    speech, words, phonemes = get_timit_val_sentence(idx)

    phoneme_sequence = []

    for word in words:
        phones = timit_word2cmu_phonemes[word]
        for p in phones:
            phoneme_sequence.append(p)
        phoneme_sequence.append('>')

    # remove the last space token
    phoneme_sequence = phoneme_sequence[:-1]

    phoneme_idx = np.array([cmu_phoneme2idx[p] for p in phoneme_sequence])

    # add a silence token (idx=1) to start and end of character sequence
    phoneme_idx = np.pad(phoneme_idx, (1, 1), mode='constant', constant_values=1)

    phoneme_idx = torch.from_numpy(phoneme_idx)
    file_name = os.path.join(path_for_saving_text_cmu_phone_files, 'val', '{}.pt'.format(idx))
    torch.save(phoneme_idx, file_name)


# test sentences
for idx in range(1344):

    speech, words, phonemes = get_timit_test_sentence(idx)

    phoneme_sequence = []

    for word in words:
        phones = timit_word2cmu_phonemes[word]
        for p in phones:
            phoneme_sequence.append(p)
        phoneme_sequence.append('>')

    # remove the last space token
    phoneme_sequence = phoneme_sequence[:-1]

    phoneme_idx = np.array([cmu_phoneme2idx[p] for p in phoneme_sequence])

    # add a silence token (idx=1) to start and end of character sequence
    phoneme_idx = np.pad(phoneme_idx, (1, 1), mode='constant', constant_values=1)

    phoneme_idx = torch.from_numpy(phoneme_idx)
    file_name = os.path.join(path_for_saving_text_cmu_phone_files, 'test', '{}.pt'.format(idx))
    torch.save(phoneme_idx, file_name)
"""
Convert the lyrics transcripts into phoneme sequences in index representation and save as torch.Tensors
"""
import pickle
import os
import glob

import torch

# should be equal to path_to_save_data in 03_preprocessing_musdb_audio_txt_char.py
path_to_dataset = '../Datasets/MUSDB_w_lyrics'

# read text file with words and correspoding phonemes, make dict musdb_word2cmu_phoneme
words2cmu_phonemes_file = open('dicts/MUSDB_words_CMU_phonemes.txt')
lines = words2cmu_phonemes_file.readlines()

musdb_word2cmu_phoneme = {'-': '-'}

for line in lines:
    line = line.replace('\n', '').split('\t')

    word = line[0].lower()
    phonemes = line[1]
    musdb_word2cmu_phoneme[word] = phonemes

print(musdb_word2cmu_phoneme)
# save musdb_word2cmu_phoneme
pickle_out = open(os.path.join('dicts', "musdb_word2cmu_phoneme.pickle"), "wb")
pickle.dump(musdb_word2cmu_phoneme, pickle_out)
pickle_out.close()

# load cmu_phoneme2idx and idx2cmu_phoneme created by 02_preprocessing_timit_cmu_phonemes.py
# this means that the CMU phonemes have the same indices for the TIMIT and the MUSDB data sets
pickle_in = open('dicts/cmu_phoneme2idx.pickle', 'rb')
cmu_phoneme2idx = pickle.load(pickle_in)
pickle_in = open('dicts/idx2cmu_phoneme.pickle', 'rb')
idx2cmu_phoneme = pickle.load(pickle_in)


# -----------------------------------------------------------------------------------------------------------

# go through dataset in train/text, val/text, test/text and load text files
for subset in ['train', 'val', 'test']:

    path_to_save_cmu_phonemes = os.path.join(path_to_dataset, subset, 'text_cmu_phonemes')
    if not os.path.isdir(path_to_save_cmu_phonemes):
        os.makedirs(path_to_save_cmu_phonemes)

    path_to_line_transcripts = os.path.join(path_to_dataset, subset, 'text')
    transcripts = glob.glob(path_to_line_transcripts + '/*.txt')

    # go through word level text transcripts
    for transcript in transcripts:
        transcript_file = open(transcript)
        words = transcript_file.read()
        words = words.split('>')
        phonemes = []
        print(words)
        for word in words:
            if word == '':
                continue
            word_phonemes = musdb_word2cmu_phoneme[word.replace('\n', '')].split(' ')
            for p in word_phonemes:
                phonemes.append(p)
            phonemes.append('>')  # add space token after each word
        phonemes = phonemes[:-1]  # remove last space token
        print(phonemes)
        phonemes_idx = torch.tensor([cmu_phoneme2idx[p] for p in phonemes]).type(torch.float32)
        print(phonemes_idx)

        file_name = transcript.split('/')[-1][:-4]
        torch.save(phonemes_idx, os.path.join(path_to_save_cmu_phonemes, file_name + '.pt'))

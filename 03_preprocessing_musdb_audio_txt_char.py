"""
This script reads the MUSDB lyrics annotation files, cuts the audio
into snippets according to annotated lines, and
saves audio and text files accordingly (both as torch files).

Please note that when this file was written,
the vocals category annotations were done with different
letters than in the publicly available version of the MUSDB lyrics.
The categories translate as follows: a-->n, b-->s, c-->d, d-->x (public format --> old format).
This script can be used with the a, b, c, d annotation style but the
annotations will be translated to the old format and the folder
structure and other scripts use the old format as well.
"""

import musdb
import librosa as lb
import torch

import os
import pickle
import yaml
# ignore warning about unsafe loaders in pyYAML 5.1 (used in musdb)
# https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
yaml.warnings({'YAMLLoadWarning': False})

path_to_musdb = '../Datasets/MUSDB18'
path_to_train_lyrics = '../Datasets/MUSDB_w_lyrics/lyrics_transcripts/train'
path_to_test_lyrics = '../Datasets/MUSDB_w_lyrics/lyrics_transcripts/test'

pickle_in = open('../Datasets/MUSDB_w_lyrics/char2idx.pickle', 'rb')
char2idx = pickle.load(pickle_in)

target_sr = 16000

path_to_save_data = '../Datasets/MUSDB_w_lyrics'

# ------------------------------------------------------------------------------------------------------------------
# make folder structure

path = os.path.join(path_to_save_data, 'test', 'text')
if not os.path.isdir(path):
    os.makedirs(path, exist_ok=True)
for stem in ['vocals', 'mix', 'accompaniments']:
    for type in ['n', 'x', 's', 'd']:
        path = os.path.join(path_to_save_data, 'test', 'audio', stem, type)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

path = os.path.join(path_to_save_data, 'val', 'text')
if not os.path.isdir(path):
    os.makedirs(path, exist_ok=True)
for stem in ['vocals', 'mix']:
    for type in ['n', 'x', 's', 'd']:
        path = os.path.join(path_to_save_data, 'val', 'audio', stem, type)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

path = os.path.join(path_to_save_data, 'train', 'text')
if not os.path.isdir(path):
    os.makedirs(path, exist_ok=True)
for stem in ['vocals', 'accompaniment', 'drums', 'bass', 'other']:
    for type in ['n', 'x', 's', 'd']:
        path = os.path.join(path_to_save_data, 'train', 'audio', stem, type)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
os.makedirs(os.path.join(path_to_save_data, 'train', 'audio', 'drums_12s'), exist_ok=True)
os.makedirs(os.path.join(path_to_save_data, 'train', 'audio', 'bass_12s'), exist_ok=True)
os.makedirs(os.path.join(path_to_save_data, 'train', 'audio', 'other_12s'), exist_ok=True)
os.makedirs(os.path.join(path_to_save_data, 'train', 'audio', 'accompaniment_12s'), exist_ok=True)
# ------------------------------------------------------------------------------------------------------------------

musdb_corpus = musdb.DB(path_to_musdb)
training_tracks = musdb_corpus.load_mus_tracks(subsets=['train'])
test_tracks = musdb_corpus.load_mus_tracks(subsets=['test'])


# validation set as open unmix but replace non english tracks by another track from the same artist:
# replaced Fergessen - Nos Palpitants by Fergessen - The Wind
# relplaced Meaxic - Take A Step by Meaxic - You Listen
validation_tracks = ['Actions - One Minute Smile',
                     'Clara Berry And Wooldog - Waltz For My Victims',
                     'Johnny Lokke - Promises & Lies',
                     'Patrick Talbot - A Reason To Leave',
                     'Triviul - Angelsaint',
                     'Alexander Ross - Goodbye Bolero',
                     'Fergessen - The Wind',
                     'Leaf - Summerghost',
                     'Skelpolu - Human Mistakes',
                     'Young Griffo - Pennies',
                     'ANiMAL - Rockshow',
                     'James May - On The Line',
                     'Meaxic - You Listen',
                     'Traffic Experiment - Sirens']

# -----------------------------------------------------------------------------------------------------------------
# process MUSDB training partition and make training and validation files
train_files_n = []
train_files_s = []
train_files_d = []
train_files_x = []

val_files_n = []
val_files_s = []
val_files_d = []
val_files_x = []

train_accompaniment_12s = []
train_bass_12s = []
train_drums_12s = []
train_other_12s = []

snippet_type_conversion = {'a': 'n', 'b': 's', 'c': 'd', 'd': 'x'}

for track in training_tracks:

    track_name = track.name

    # make file name for audio and text files of current track
    file_name = track.name.split('-')
    file_name = file_name[0][0:6] + "_" + file_name[1][1:6]
    file_name = file_name.replace(" ", "_")

    # make boolean indicating whether current track is in validation set
    val_set_track = track_name in validation_tracks

    # -----------------------------------------------------------------------------------------------------------------
    # generate accompaniment snippets of 12 s length of all tracks in training partition
    if not val_set_track:
        for target in ['accompaniment', 'drums', 'bass', 'other']:
            accompaniment_audio = track.targets[target].audio
            accompaniment_audio_resampled = lb.core.resample(accompaniment_audio.T, track.rate, target_sr)
            acc_snippets = lb.util.frame(accompaniment_audio_resampled, frame_length=12 * target_sr,
                                         hop_length=12 * target_sr)

            number_of_snippets = acc_snippets.shape[-1]

            for counter in range(number_of_snippets):
                # audio_torch has shape (2, ???) = (channels, samples)
                audio_torch = torch.tensor(acc_snippets[:, :, counter]).type(torch.float32)
                torch.save(audio_torch, os.path.join(path_to_save_data, 'train', 'audio', '{}_12s'.format(target),
                                                 file_name + '_{}.pt'.format(counter)))
                if target == 'accompaniment':
                    train_accompaniment_12s.append(file_name + '_{}.pt'.format(counter))
                elif target == 'drums':
                    train_drums_12s.append(file_name + '_{}.pt'.format(counter))
                elif target == 'bass':
                    train_bass_12s.append(file_name + '_{}.pt'.format(counter))
                elif target == 'other':
                    train_other_12s.append(file_name + '_{}.pt'.format(counter))
    # -----------------------------------------------------------------------------------------------------------------

    path_to_track_lyrics = os.path.join(path_to_train_lyrics, track_name + '.txt')

    # ignore files without lyrics annotations
    if not os.path.isfile(path_to_track_lyrics):
        print("No lyrics for", track, ", it was skipped")
        continue

    lyrics_file = open(path_to_track_lyrics)
    lyrics_lines = lyrics_file.readlines()

    vocals_audio = track.targets['vocals'].audio

    if val_set_track:
        other_audio = track.audio

        # resample
        acc_audio_resampled = lb.core.resample(other_audio.T, track.rate, target_sr)

        vocals_audio_resampled = lb.core.resample(vocals_audio.T, track.rate, target_sr)

        # go through lyrics lines and split audio as annotated
        for counter, line in enumerate(lyrics_lines):

            # ignore rejected lines
            if line[0] == '*':
                continue

            annotations = line.split(' ', maxsplit=3)

            start_m = int(annotations[0].split(':')[0])  # start time minutes
            start_s = int(annotations[0].split(':')[1])  # start time seconds
            start_time = start_m * 60 + start_s  # start time in seconds

            end_m = int(annotations[1].split(':')[0])  # end time minutes
            end_s = int(annotations[1].split(':')[1])  # end time seconds
            end_time = end_m * 60 + end_s  # end time in seconds

            acc_audio_snippet = acc_audio_resampled[:, start_time * target_sr: end_time * target_sr]
            vocals_audio_snippet = vocals_audio_resampled[:, start_time * target_sr: end_time * target_sr]

            acc_audio_snippet_torch = torch.tensor(acc_audio_snippet).type(torch.float32)
            vocals_audio_snippet_torch = torch.tensor(vocals_audio_snippet).type(torch.float32)

            snippet_type = annotations[2]  # a, b, c, d

            snippet_type = snippet_type_conversion[snippet_type]  # change to old format n, s, d, x

            text = annotations[3].replace('\n', '').replace(' ', '>')
            text_idx = torch.tensor([char2idx[char] for char in text]).type(torch.float32)

            snippet_file_name = file_name + '_{}'.format(counter)

            partition = 'val'
            other = 'mix'

            # save audio
            path_to_save_vocals = os.path.join(path_to_save_data, partition, 'audio', 'vocals', snippet_type,
                                               snippet_file_name)
            path_to_save_other = os.path.join(path_to_save_data, partition, 'audio', other, snippet_type,
                                              snippet_file_name)
            torch.save(acc_audio_snippet_torch, path_to_save_other + '.pt')
            torch.save(vocals_audio_snippet_torch, path_to_save_vocals + '.pt')

            # save text
            path_to_save_text = os.path.join(path_to_save_data, partition, 'text', snippet_file_name + '.txt')
            path_to_save_text_idx = os.path.join(path_to_save_data, partition, 'text', snippet_file_name + '.pt')
            with open(path_to_save_text, 'w') as txt_file:
                txt_file.write(text)
                txt_file.close()
            torch.save(text_idx, path_to_save_text_idx)

            if snippet_type == 'n':
                val_files_n.append('n/{}'.format(snippet_file_name))
            if snippet_type == 'x':
                val_files_x.append('x/{}'.format(snippet_file_name))
            if snippet_type == 's':
                val_files_s.append('s/{}'.format(snippet_file_name))
            if snippet_type == 'd':
                val_files_d.append('d/{}'.format(snippet_file_name))

    # process training songs
    else:
        acc_audio = track.targets['accompaniment'].audio
        drums_audio = track.targets['drums'].audio
        bass_audio = track.targets['bass'].audio
        other_audio = track.targets['other'].audio

        # resample
        vocals_audio_resampled = lb.core.resample(vocals_audio.T, track.rate, target_sr)
        acc_audio_resampled = lb.core.resample(acc_audio.T, track.rate, target_sr)
        drums_audio_resampled = lb.core.resample(drums_audio.T, track.rate, target_sr)
        bass_audio_resampled = lb.core.resample(bass_audio.T, track.rate, target_sr)
        other_audio_resampled = lb.core.resample(other_audio.T, track.rate, target_sr)

        # go through lyrics lines and split audio as annotated
        for counter, line in enumerate(lyrics_lines):

            # ignore rejected lines
            if line[0] == '*':
                continue

            annotations = line.split(' ', maxsplit=3)

            start_m = int(annotations[0].split(':')[0])  # start time minutes
            start_s = int(annotations[0].split(':')[1])  # start time seconds
            start_time = start_m * 60 + start_s  # start time in seconds

            end_m = int(annotations[1].split(':')[0])  # end time minutes
            end_s = int(annotations[1].split(':')[1])  # end time seconds
            end_time = end_m * 60 + end_s  # end time in seconds

            acc_audio_snippet = acc_audio_resampled[:, start_time * target_sr: end_time * target_sr]
            vocals_audio_snippet = vocals_audio_resampled[:, start_time * target_sr: end_time * target_sr]
            drums_audio_snippet = drums_audio_resampled[:, start_time * target_sr: end_time * target_sr]
            bass_audio_snippet = bass_audio_resampled[:, start_time * target_sr: end_time * target_sr]
            other_audio_snippet = other_audio_resampled[:, start_time * target_sr: end_time * target_sr]

            acc_audio_snippet_torch = torch.tensor(acc_audio_snippet).type(torch.float32)
            vocals_audio_snippet_torch = torch.tensor(vocals_audio_snippet).type(torch.float32)
            drums_audio_snippet_torch = torch.tensor(drums_audio_snippet).type(torch.float32)
            bass_audio_snippet_torch = torch.tensor(bass_audio_snippet).type(torch.float32)
            other_audio_snippet_torch = torch.tensor(other_audio_snippet).type(torch.float32)

            snippet_type = annotations[2]  # a, b, c, d

            snippet_type = snippet_type_conversion[snippet_type]  # change to old format n, s, d, x

            text = annotations[3].replace('\n', '').replace(' ', '>')
            text_idx = torch.tensor([char2idx[char] for char in text]).type(torch.float32)

            snippet_file_name = file_name + '_{}'.format(counter)

            partition = 'train'
            other = 'accompaniments'

            # save audio
            path_to_save_vocals = os.path.join(path_to_save_data, partition, 'audio', 'vocals', snippet_type,
                                               snippet_file_name)
            path_to_save_acc = os.path.join(path_to_save_data, partition, 'audio', 'accompaniment', snippet_type,
                                              snippet_file_name)
            path_to_save_drums = os.path.join(path_to_save_data, partition, 'audio', 'drums', snippet_type,
                                            snippet_file_name)
            path_to_save_bass = os.path.join(path_to_save_data, partition, 'audio', 'bass', snippet_type,
                                            snippet_file_name)
            path_to_save_other = os.path.join(path_to_save_data, partition, 'audio', 'other', snippet_type,
                                            snippet_file_name)

            torch.save(acc_audio_snippet_torch, path_to_save_acc + '.pt')
            torch.save(vocals_audio_snippet_torch, path_to_save_vocals + '.pt')
            torch.save(drums_audio_snippet_torch, path_to_save_drums + '.pt')
            torch.save(bass_audio_snippet_torch, path_to_save_bass + '.pt')
            torch.save(other_audio_snippet_torch, path_to_save_other + '.pt')

            # save text
            path_to_save_text = os.path.join(path_to_save_data, partition, 'text', snippet_file_name + '.txt')
            path_to_save_text_idx = os.path.join(path_to_save_data, partition, 'text', snippet_file_name + '.pt')
            with open(path_to_save_text, 'w') as txt_file:
                txt_file.write(text)
                txt_file.close()
            torch.save(text_idx, path_to_save_text_idx)

            if snippet_type == 'n':
                train_files_n.append('n/{}'.format(snippet_file_name))
            if snippet_type == 'x':
                train_files_x.append('x/{}'.format(snippet_file_name))
            if snippet_type == 's':
                train_files_s.append('s/{}'.format(snippet_file_name))
            if snippet_type == 'd':
                train_files_d.append('d/{}'.format(snippet_file_name))

# # save lists with file names
pickle_out = open(os.path.join(path_to_save_data, "val", "val_files_n.pickle"), "wb")
pickle.dump(val_files_n, pickle_out)
pickle_out.close()
pickle_out = open(os.path.join(path_to_save_data, "val", "val_files_x.pickle"), "wb")
pickle.dump(val_files_x, pickle_out)
pickle_out.close()
pickle_out = open(os.path.join(path_to_save_data, "val", "val_files_s.pickle"), "wb")
pickle.dump(val_files_s, pickle_out)
pickle_out.close()
pickle_out = open(os.path.join(path_to_save_data, "val", "val_files_d.pickle"), "wb")
pickle.dump(val_files_d, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(path_to_save_data, "train", "train_files_n.pickle"), "wb")
pickle.dump(train_files_n, pickle_out)
pickle_out.close()
pickle_out = open(os.path.join(path_to_save_data, "train", "train_files_x.pickle"), "wb")
pickle.dump(train_files_x, pickle_out)
pickle_out.close()
pickle_out = open(os.path.join(path_to_save_data, "train", "train_files_s.pickle"), "wb")
pickle.dump(train_files_s, pickle_out)
pickle_out.close()
pickle_out = open(os.path.join(path_to_save_data, "train", "train_files_d.pickle"), "wb")
pickle.dump(train_files_d, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(path_to_save_data, "train", "train_accompaniments_12s.pickle"), "wb")
pickle.dump(train_accompaniment_12s, pickle_out)
pickle_out.close()
pickle_out = open(os.path.join(path_to_save_data, "train", "train_drums_12s.pickle"), "wb")
pickle.dump(train_drums_12s, pickle_out)
pickle_out.close()
pickle_out = open(os.path.join(path_to_save_data, "train", "train_bass_12s.pickle"), "wb")
pickle.dump(train_bass_12s, pickle_out)
pickle_out.close()
pickle_out = open(os.path.join(path_to_save_data, "train", "train_other_12s.pickle"), "wb")
pickle.dump(train_other_12s, pickle_out)
pickle_out.close()

print("Train files n:", train_files_n)
print("Train files x:", train_files_x)
print("Train files s:", train_files_s)
print("Train files d:", train_files_d)
print("Val files n:", val_files_n)
print("Val files x:", val_files_x)
print("Val files s:", val_files_s)
print("Val files d:", val_files_d)
print("Train accompaniments 12s:", train_accompaniment_12s)
# -----------------------------------------------------------------------------------------------------------------


# process MUSDB test partition and make test files
test_files_n = []
test_files_s = []
test_files_d = []
test_files_x = []
for track in test_tracks:

    track_name = track.name

    # make file name for audio and text files of current track
    file_name = track.name.split('-')
    file_name = file_name[0][0:6] + "_" + file_name[1][1:6]
    file_name = file_name.replace(" ", "_")

    path_to_track_lyrics = os.path.join(path_to_test_lyrics, track_name + '.txt')

    # ignore files without lyrics annotations
    if not os.path.isfile(path_to_track_lyrics):
        print("No lyrics for", track, ", it was skipped")
        continue

    lyrics_file = open(path_to_track_lyrics)
    lyrics_lines = lyrics_file.readlines()

    mix_audio = track.audio
    vocals_audio = track.targets['vocals'].audio
    accompaniment_audio = track.targets['accompaniment'].audio

    # resample
    mix_audio_resampled = lb.core.resample(mix_audio.T, track.rate, target_sr)
    vocals_audio_resampled = lb.core.resample(vocals_audio.T, track.rate, target_sr)
    accompaniment_audio_resampled = lb.core.resample(accompaniment_audio.T, track.rate, target_sr)

    # go through lyrics lines and split audio as annotated
    for counter, line in enumerate(lyrics_lines):

        # ignore rejected lines
        if line[0] == '*':
            continue

        annotations = line.split(' ', maxsplit=3)

        start_m = int(annotations[0].split(':')[0])  # start time minutes
        start_s = int(annotations[0].split(':')[1])  # start time seconds
        start_time = start_m * 60 + start_s  # start time in seconds

        end_m = int(annotations[1].split(':')[0])  # end time minutes
        end_s = int(annotations[1].split(':')[1])  # end time seconds
        end_time = end_m * 60 + end_s  # end time in seconds

        mix_audio_snippet = mix_audio_resampled[:, start_time * target_sr: end_time * target_sr]
        vocals_audio_snippet = vocals_audio_resampled[:, start_time * target_sr: end_time * target_sr]
        accompaniment_audio_snippet = accompaniment_audio_resampled[:, start_time * target_sr: end_time * target_sr]

        mix_audio_snippet_torch = torch.tensor(mix_audio_snippet).type(torch.float32)
        vocals_audio_snippet_torch = torch.tensor(vocals_audio_snippet).type(torch.float32)
        accompaniment_audio_snippet_torch = torch.tensor(accompaniment_audio_snippet).type(torch.float32)

        snippet_type = annotations[2]  # a, b, c, d

        snippet_type = snippet_type_conversion[snippet_type]  # change to old format n, s, d, x

        text = annotations[3].replace('\n', '').replace(' ', '>')
        text_idx = torch.tensor([char2idx[char] for char in text]).type(torch.float32)

        snippet_file_name = file_name + '_{}'.format(counter)

        # save audio
        path_to_save_vocals = os.path.join(path_to_save_data, 'test', 'audio', 'vocals', snippet_type,
                                           snippet_file_name)
        path_to_save_mix = os.path.join(path_to_save_data, 'test', 'audio', 'mix', snippet_type, snippet_file_name)
        path_to_save_acc = os.path.join(path_to_save_data, 'test', 'audio', 'accompaniments', snippet_type,
                                        snippet_file_name)

        torch.save(mix_audio_snippet_torch, path_to_save_mix + '.pt')
        torch.save(vocals_audio_snippet_torch, path_to_save_vocals + '.pt')
        torch.save(accompaniment_audio_snippet_torch, path_to_save_acc + '.pt')

        # save text
        path_to_save_text = os.path.join(path_to_save_data, 'test', 'text', snippet_file_name + '.txt')
        path_to_save_text_idx = os.path.join(path_to_save_data, 'test', 'text', snippet_file_name + '.pt')
        with open(path_to_save_text, 'w') as txt_file:
            txt_file.write(text)
            txt_file.close()
        torch.save(text_idx, path_to_save_text_idx)

        if snippet_type == 'n':
            test_files_n.append('n/{}'.format(snippet_file_name))
        if snippet_type == 'x':
            test_files_x.append('x/{}'.format(snippet_file_name))
        if snippet_type == 's':
            test_files_s.append('s/{}'.format(snippet_file_name))
        if snippet_type == 'd':
            test_files_d.append('d/{}'.format(snippet_file_name))


# save lists with file names
pickle_out = open(os.path.join(path_to_save_data, "test", "test_files_n.pickle"), "wb")
pickle.dump(test_files_n, pickle_out)
pickle_out.close()
pickle_out = open(os.path.join(path_to_save_data, "test", "test_files_x.pickle"), "wb")
pickle.dump(test_files_x, pickle_out)
pickle_out.close()
pickle_out = open(os.path.join(path_to_save_data, "test", "test_files_s.pickle"), "wb")
pickle.dump(test_files_s, pickle_out)
pickle_out.close()
pickle_out = open(os.path.join(path_to_save_data, "test", "test_files_d.pickle"), "wb")
pickle.dump(test_files_d, pickle_out)
pickle_out.close()

print("Test files n:", test_files_n)
print("Test files x:", test_files_x)
print("Test files s:", test_files_s)
print("Test files d:", test_files_d)

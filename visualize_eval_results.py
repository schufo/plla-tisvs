"""
This script can be used to print tables with the evaluation results organized according to vocals categories
"""
import pandas as pd
import museval
import musdb
import os
import json
import numpy as np

# choose which target to visualize
target = 'vocals'  # vocals, accompaniment, all

# which statistic to compute on the data
statistic = 'median'  # mean or median

# choose which tags to visualize
tags = ['SEQ', 'SEQ-BL1', 'SEQ-BL2']

# ----------------------------------------------------------------------------------------------------------------
# statistics per track
results_over_tracks = pd.DataFrame({'tag': [], 'SDR': [], 'SIR': [], 'SAR': []})

# statistics over ALL frames
results_over_frames = pd.DataFrame({'tag': [],
                                         'SDR_n': [], 'SIR_n': [], 'SAR_n': [], 'PES_n': [], 'EPS_n': [],
                                         'SDR_s': [], 'SIR_s': [], 'SAR_s': [], 'PES_s': [], 'EPS_s': [],
                                         'SDR_d': [], 'SIR_d': [], 'SAR_d': [], 'PES_d': [], 'EPS_d': [],
                                         'SDR_x': [], 'SIR_x': [], 'SAR_x': [], 'PES_x': [], 'EPS_x': [],
                                         'SDR_nsd': [], 'SIR_nsd': [], 'SAR_nsd': [], 'PES_nsd': [], 'EPS_nsd': []})

# silence results over all frames
results_silence_over_frames = pd.DataFrame({'tag': [], 'PES': [], 'EPS': []})


track_names = set()  # all test tracks in shortened form
museval_data_path = os.path.join('evaluation', tags[0], 'bss_eval_results.pandas')
museval_data = pd.read_pickle(museval_data_path)
for track in museval_data['track'].values:
    track_names.add(track[2:10])

for tag in tags:

    tag_dict = {'tag': tag,
                'SDR_n': [], 'SIR_n': [], 'SAR_n': [], 'PES_n': [], 'EPS_n': [],
                'SDR_s': [], 'SIR_s': [], 'SAR_s': [], 'PES_s': [], 'EPS_s': [],
                'SDR_d': [], 'SIR_d': [], 'SAR_d': [], 'PES_d': [], 'EPS_d': [],
                'SDR_x': [], 'SIR_x': [], 'SAR_x': [], 'PES_x': [], 'EPS_x': [],
                'SDR_nsd': [], 'SIR_nsd': [], 'SAR_nsd': [], 'PES_nsd': [], 'EPS_nsd': []}

    silence_over_frames_dict = {'tag': tag, 'PES': [], 'EPS': []}

    # load museval summary
    museval_data_path = os.path.join('evaluation', tag, 'bss_eval_results.pandas')
    museval_data = pd.read_pickle(museval_data_path)
    # remove nan
    museval_data = museval_data.dropna(axis=0)

    # load silent frames results
    silent_frames_data_path = os.path.join('evaluation', tag, 'silent_frames_results.json')
    silent_frames_data = pd.read_json(silent_frames_data_path, orient='records')

    silence_over_frames_dict['PES'] = silent_frames_data[(silent_frames_data['target'] == 'vocals')].mean(axis=0, skipna=True)['PES']
    silence_over_frames_dict['EPS'] = silent_frames_data[(silent_frames_data['target'] == 'vocals')].mean(axis=0, skipna=True)['EPS']

    results_silence_over_frames = results_silence_over_frames.append(pd.DataFrame([silence_over_frames_dict]),
                                                                     ignore_index=True, sort=False)


    # statistics over all frames within a vocals type
    for vocals_type in ['n', 's', 'd', 'x', 'nsd']:

        if statistic == 'mean':
            if vocals_type == 'nsd':
                vocals_mean = silent_frames_data[(silent_frames_data['target'] == 'vocals') &
                                                 (silent_frames_data['track'].str.contains('n/') |
                                                  silent_frames_data['track'].str.contains('s/') |
                                                  silent_frames_data['track'].str.contains('d/'))] \
                                                  .mean(axis=0, skipna=True)
            else:
                vocals_mean = silent_frames_data[(silent_frames_data['target'] == 'vocals') &
                                             (silent_frames_data['track'].str.contains('{}/'.format(vocals_type)))]\
                                              .mean(axis=0, skipna=True)
            # add silent frames results to method results dict
            tag_dict['PES_{}'.format(vocals_type)] = vocals_mean['PES']
            tag_dict['EPS_{}'.format(vocals_type)] = vocals_mean['EPS']

        for metric in ['SDR', 'SIR', 'SAR']:

            if vocals_type == 'nsd':
                values = museval_data[(museval_data['metric']== metric) &
                                      (museval_data['track'].str.contains('n/') |
                                       museval_data['track'].str.contains('s/') |
                                       museval_data['track'].str.contains('d/')) &
                                      (museval_data['target'] == 'vocals')]['score'].values
            else:
                values = museval_data[(museval_data['metric']==metric) &
                                  (museval_data['track'].str.contains('{}/'.format(vocals_type))) &
                                  (museval_data['target'] == 'vocals')]['score'].values

            if statistic == 'mean':
                summary_statistic = np.mean(values)
            elif statistic == 'median':
                summary_statistic = np.median(values)

            tag_dict['{}_{}'.format(metric, vocals_type)] = summary_statistic


    results_over_frames = results_over_frames.append(pd.DataFrame([tag_dict]), ignore_index=True, sort=False)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print('Results over frames:')
    print(results_over_frames)
    print('Results on silence over all frames:')
    print(results_silence_over_frames)


print('Results over frames:')
print(results_over_frames.to_latex(
    float_format="{:0.2f}".format, index=False))
print('Results on silence over all frames:')
print(results_silence_over_frames.to_latex(
    float_format="{:0.2f}".format, index=False))





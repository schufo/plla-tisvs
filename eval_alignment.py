"""
Compute alignment evaluation metrics for alignment estimates computed with estimate_alignment.py
and save them as json file.
"""

import os
import json

import numpy as np
import torch
import data

# from https://github.com/f90/jamendolyrics
def duration_correct(ref_timestamps, pred_timestamps, total_duration):
    # Compute for given ref and prediction timestamps and the total song duration,
    # what percentage of time the predicted position coincides with the true position (between 0 and 1)
    assert(len(ref_timestamps) == len(pred_timestamps))
    assert(np.max(ref_timestamps) <= total_duration and np.max(pred_timestamps) <= total_duration)
    correct = 0.0
    ref_prev = 0.0
    pred_prev = 0.0
    for i in range(len(ref_timestamps)):
        # Compare intersection of intervals [ref_prev, ref_curr] and [pred_prev, pred_curr]
        corr_interval_start = max(ref_prev, pred_prev)  # latest interval start
        corr_interval_end = min(ref_timestamps[i], pred_timestamps[i])  # earliest interval stop
        correct += max(corr_interval_end - corr_interval_start, 0)  # time of correct interval length

        ref_prev = ref_timestamps[i]
        pred_prev = pred_timestamps[i]
    # One last calculation for final interval until end of track
    corr_interval_start = max(ref_prev, pred_prev)
    corr_interval_end = total_duration
    correct += max(corr_interval_end - corr_interval_start, 0)

    percentage_correct = correct / total_duration
    return percentage_correct


if __name__ == '__main__':

    tags = ['JOINT3']
    test_sets = ['Hansen', 'Jamendo' , 'NUS_acapella', 'NUS_snr5', 'NUS_snr0', 'NUS_snr-5']

    for tag in tags:
        for test_set in test_sets:

            path_to_estimates = 'evaluation/{}/alignments/{}'.format(tag, test_set)

            if test_set == 'Hansen':
                dataset = data.Hansen()
            elif test_set == 'Jamendo':
                dataset = data.Jamendo()
            elif test_set[:3] == 'NUS':
                dataset = data.NUS(acapella=test_set=='NUS_acapella')


            results = {}

            for idx in range(len(dataset)):
                test_example = dataset[idx]
                name = test_example['name']
                if test_set[:3] == 'NUS' and name[:6] == 'ADIZ09':
                    # skip this song because of incorrect phoneme onset annotations
                    continue

                true_onsets = np.array(test_example['true_onsets'])

                if test_set[:3] == 'NUS' and name[:6] == 'ZHIY02' and tag == 'MFA':
                    true_onsets = true_onsets[:-2]  # last two phonemes were lost in MFA
                #true_offsets = np.array(test_example['true_offsets'])
                audio = test_example['audio']

                # load estimates
                estimated_onsets = np.load(os.path.join(path_to_estimates, name + '_onsets.npy'))

                # absolute error
                absolute_error = abs(true_onsets - estimated_onsets)
                mean_abs_error = np.mean(absolute_error)
                median_abs_error = np.median(absolute_error)

                # percentage correctly aligned within a tolerance (like Mauch et al. 2012)
                perc_within_tolerance = {}  # tolerance: percentage
                if test_set[:3] == 'NUS':
                    tolerances = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0]
                else:
                    tolerances = np.round_(np.linspace(0.1, 1.0, 10), decimals=1)
                for tolerance in tolerances:
                    perc_correct = np.mean(absolute_error < tolerance)
                    perc_within_tolerance[tolerance] = perc_correct

                # percentage of correct segments (Fujihara et al. 2011)
                duration = audio.size(0) / 16000
                perc_correct_seg = duration_correct(true_onsets, estimated_onsets, duration)

                song_results = {'mean_abs_error': mean_abs_error, 'median_abs_error': median_abs_error,
                                'perc_within_tolerance': perc_within_tolerance, 'perc_correct_seg': perc_correct_seg}

                # add song results to results dict
                results[name[:17]] = song_results

            # compute averages over the data set and add to the results dict
            mean_mean_ae = np.mean([results[song]['mean_abs_error'] for song in results.keys()])
            mean_median_ae = np.mean([results[song]['median_abs_error'] for song in results.keys()])
            mean_perc_correct_seg = np.mean([results[song]['perc_correct_seg'] for song in results.keys()])

            mean_perc_within_tol = {}
            for tolerance in tolerances:
                mean_perc_within_tolerance = np.mean([results[song]['perc_within_tolerance'][tolerance] for song in results.keys()])
                mean_perc_within_tol[tolerance] = mean_perc_within_tolerance

            results['mean_mean_ae'] = mean_mean_ae
            results['mean_median_ae'] = mean_median_ae
            results['mean_perc_within_tol'] = mean_perc_within_tol
            results['mean_perc_correct_seg'] = mean_perc_correct_seg

            with open(os.path.join(path_to_estimates, 'alignment_results.json'), 'w') as file:
                json.dump(results, file)
            print(results)


"""
This file is a modified version of https://github.com/sigsep/open-unmix-pytorch/blob/master/openunmix/evaluate.py
It can be used to evaluate the source separation results using objective measures SDR, SAR, SIR, EPS, PES
"""

import argparse
import musdb
import museval
import testx
import multiprocessing
import functools
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import tqdm
import resampy
import json
import os
import soundfile as sf

import data
import model
import silent_frames_evaluation

def evaluate(references, estimates, output_dir, track_name, sample_rate, win=1.0, hop=1.0, mode='v4'):

    """
    Compute the BSS_eval metrics as well as PES and EPS. It is following the design concept of museval.eval_mus_track
    :param references: dict of reference sources {target_name: signal}, signal has shape: (nb_timesteps, np_channels)
    :param estimates: dict of user estimates {target_name: signal}, signal has shape: (nb_timesteps, np_channels)
    :param output_dir: path to output directory used to save evaluation results
    :param track_name: name that is assigned to TrackStore object for evaluated track
    :param win: evaluation window length in seconds, default 1
    :param hop: evaluation window hop length in second, default 1
    :param sample_rate: sample rate of test tracks (should be same as rate the model has been trained on)
    :param mode: BSSEval version, default to `v4`
    :return:
        bss_eval_data: museval.TrackStore object containing bss_eval evaluation scores
        silent_frames_data: Pandas data frame containing EPS and PES scores
    """

    eval_targets = list(estimates.keys())

    estimates_list = []
    references_list = []
    for target in eval_targets:
        estimates_list.append(estimates[target])
        references_list.append(references[target])

    # eval bass_eval and EPS, PES metrics
    # save in TrackStore object
    bss_eval_data = museval.TrackStore(win=win, hop=hop, track_name=track_name)

    # skip examples with a silent source because BSSeval metrics are not defined in this case
    skip = False
    for target in eval_targets:
        reference_energy = np.sum(references[target]**2)
        estimate_energy = np.sum(estimates[target]**2)
        if reference_energy == 0 or estimate_energy == 0:
            skip = True
            SDR = ISR = SIR = SAR = (np.ones((1,)) * (-np.inf), np.ones((1,)) * (-np.inf))
            print("skip {}, {} source is all zero".format(track_name, target))

    if not skip:

        SDR, ISR, SIR, SAR = museval.evaluate(
            references_list,
            estimates_list,
            win=int(win * sample_rate),
            hop=int(hop * sample_rate),
            mode=mode,
            padding=True
        )

    # add evaluation of ESP and PES
    PES, EPS, _, __ = silent_frames_evaluation.eval_silent_frames(
        true_source=np.array(references_list),
        predicted_source=np.array(estimates_list),
        window_size=int(win * sample_rate),
        hop_size=int(hop * sample_rate)
    )

    # iterate over all targets
    for i, target in enumerate(eval_targets):
        values = {
            "SDR": SDR[i].tolist(),
            "SIR": SIR[i].tolist(),
            "ISR": ISR[i].tolist(),
            "SAR": SAR[i].tolist(),
        }

        bss_eval_data.add_target(
            target_name=target,
            values=values
        )

    silent_frames_data = pd.DataFrame({'target': [], 'PES': [], 'EPS': [], 'track': []})
    for i, target in enumerate(eval_targets):
       silent_frames_data = silent_frames_data.append({'target': target, 'PES': PES[i], 'EPS': EPS[i], 'track': track_name}, ignore_index=True)

    # save evaluation results if output directory is defined
    if output_dir:
        # validate against the schema
        bss_eval_data.validate()

        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(
                    os.path.join(output_dir, track_name.replace('/', '_')) + '.json', 'w+'
            ) as f:
                f.write(bss_eval_data.json)
        except (IOError):
            pass

    return bss_eval_data, silent_frames_data


def separate_and_evaluate2(
    track,
    targets,
    model_name,
    niter,
    alpha,
    softmask,
    output_dir,
    eval_dir,
    samplerate,
    device='cpu',
    args=None

):
    mix = track['mix']
    true_vocals = track['vocals']
    true_accompaniment = track['accompaniment']
    text = track['text'].unsqueeze(dim=0)
    track_name = track['name']

    mix_numpy = mix.numpy().T

    if args.alignment_from:
        attention_weights = track['attention_weights'].unsqueeze(dim=0)
        inputs = (mix_numpy, text, attention_weights)
    else:
        inputs = (mix_numpy, text)

    estimates = testx.separate(
        inputs=inputs,
        targets=targets,
        model_name=model_name,
        niter=niter,
        alpha=alpha,
        softmask=softmask,
        device=device,
        args=args,
        accompaniment_model=args.accompaniment_model
    )

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # make another script that reassembles kind of whole tracks of snippets
        sf.write(os.path.join(output_dir, track_name.replace('/', '_') + '.wav'), estimates['vocals'], samplerate)

    references = {'vocals': true_vocals.numpy().T, 'accompaniment': true_accompaniment.numpy().T}
    bss_eval_scores, silent_frames_scores = evaluate(references=references,
                                                     estimates=estimates,
                                                     output_dir=eval_dir,
                                                     track_name=track_name,
                                                     sample_rate=samplerate)

    return bss_eval_scores, silent_frames_scores


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='MUSDB18 Evaluation',
        add_help=False
    )

    # changed default to vocals
    parser.add_argument(
        '--targets',
        nargs='+',
        default=['vocals'],
        type=str,
        help='provide targets to be processed. \
              If none, all available targets will be computed'
    )

    parser.add_argument(
        '--tag',
        type=str,
        help ='tag of model/variant to evaluate that was assigned at training time'
    )


    parser.add_argument(
        '--test-snr',
        type=int,
        default= None,
        help ='SNR if evaluation mixes should have a custom SNR'
    )

    args, _ = parser.parse_known_args()

    if args.test_snr == None:
        default_eval_tag = args.tag
    else:
        default_eval_tag = args.tag + "_snr_" + str(args.test_snr)

    parser.add_argument(
        '--eval-tag',
        type=str,
        default= default_eval_tag,
        help ='tag for evaluation folder etc.'
    )

    # changed
    parser.add_argument(
        '--model',
        type=str,
        default='trained_models/{}'.format(args.tag),
        help='path to mode base directory of pretrained models'
    )

    parser.add_argument(
        '--optimal-path-attention',
        action='store_true',
        default=False,
        help='Set to True if alphas should be the optimal path through scores during evaluation'
    )

    parser.add_argument(
        '--estimates-out',
        action='store_true',
        default=False,
        help='Set to True if estimates should be stored'
    )

    parser.add_argument(
        '--outdir',
        type=str,
        help='Results path where audio evaluation results are stored'
    )

    args, _ = parser.parse_known_args()

    outdir = None
    if args.estimates_out and not args.outdir:
        outdir = 'evaluation/{}/estimates'.format(args.eval_tag)
    if args.estimates_out and args.outdir:
        outdir = args.outdir

    parser.add_argument(
        '--evaldir',
        type=str,
        default='evaluation/{}/eval_results'.format(args.eval_tag),
        help='Results path for museval scores'
    )

    # changed --root to --testset
    parser.add_argument(
        '--testset',
        type=str,
        help='Test set name'
    )

    # removed --subset

    parser.add_argument(
        '--cores',
        type=int,
        default=1
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA inference'
    )

    parser.add_argument(
        '--is-wav',
        action='store_true', default=False,
        help='flags wav version of the dataset'
    )

    args, _ = parser.parse_known_args()

    if args.testset == 'musdb':
        mus = musdb.DB(
        root='../Datasets/MUSDB18',
        subsets='test',
        is_wav=args.is_wav
        )
        test_tracks = mus.tracks

    elif args.testset == 'musdb_lyrics':

        try:
            # load configuration that was generated at training time
            with open(Path(args.model, args.targets[0] + '.json'), 'r') as stream:
                config = json.load(stream)
                keys = config['args'].keys()
                samplerate = config['args']['samplerate']
                text_units_default = config['args']['text_units']
                space_token_only = config['args']['space_token_only'] if 'space_token_only' in keys else False
                alignment_from = config['args']['alignment_from'] if 'alignment_from' in keys else None
                fake_alignment = config['args']['fake_alignment'] if 'fake_alignment' in keys else False
        except (FileNotFoundError):
            text_units_default = 'characters'
            samplerate = 44100
            fake_alignment = False
            space_token_only = True
            alignment_from = None
            print("No config found, set n, x, s, d to True,  test_units=characters, samplerate=44100")


        # use same data set specifications as during training but allow overwriting on command line
        parser.add_argument('--text-units', type=str, default=text_units_default)

        parser.add_argument('--alignment-from', type=str, default=alignment_from)
        args, _ = parser.parse_known_args()

        # here one "test track" is actually one snippet
        test_tracks = data.MUSDBLyricsDataTest(samplerate=samplerate, text_units=args.text_units,
                                               n=True, x=True, s=True, d=True, space_token_only=space_token_only,
                                               alignment_from=args.alignment_from, fake_alignment=fake_alignment,
                                               mix_snr=args.test_snr)

    args, _ = parser.parse_known_args()
    args = testx.inference_args(parser, args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    if args.cores > 1:
        pool = multiprocessing.Pool(args.cores)
        results = museval.EvalStore()
        scores_list = list(
            pool.imap_unordered(
                func=functools.partial(
                    separate_and_evaluate2,
                    targets=args.targets,
                    model_name=args.model,
                    niter=args.niter,
                    alpha=args.alpha,
                    softmask=args.softmask,
                    output_dir=outdir,
                    eval_dir=args.evaldir,
                    device=device
                ),
                iterable=mus.tracks,
                chunksize=1
            )
        )
        pool.close()
        pool.join()
        for scores in scores_list:
            results.add_track(scores)

    else:
        print("evaluate model {} with sample rate {}".format(args.model, samplerate))
        results = museval.EvalStore()
        silent_frames_results = pd.DataFrame({'target': [], 'PES': [], 'EPS': [], 'track': []})
        for idx in tqdm.tqdm(range(len(test_tracks))):
            track = test_tracks[idx]
            bss_eval_scores, silent_frames_scores = separate_and_evaluate2(
                                                                           track,
                                                                           targets=args.targets,
                                                                           model_name=args.model,
                                                                           niter=args.niter,
                                                                           alpha=args.alpha,
                                                                           softmask=args.softmask,
                                                                           output_dir=outdir,
                                                                           eval_dir=args.evaldir,
                                                                           device=device,
                                                                           samplerate=samplerate,
                                                                           args=args
                                                                           )
            results.add_track(bss_eval_scores)
            silent_frames_results = silent_frames_results.append(silent_frames_scores, ignore_index=True)

    print(results)
    method = museval.MethodStore()
    method.add_evalstore(results, args.model)
    method.save('evaluation/{}/bss_eval_results.pandas'.format(args.eval_tag))

    # print mean over tracks for PES and EPS
    print("mean over evaluation frames, mean over channels, mean over tracks")
    for target in ['vocals', 'accompaniment']:
        print(target + ' ==>', silent_frames_results.loc[silent_frames_results['target'] == target].mean(axis=0, skipna=True))

    silent_frames_results.to_json('evaluation/{}/silent_frames_results.json'.format(args.eval_tag), orient='records')

"""
Implementation of the PES and EPS evaluation metrics. It is used in eval_separation.py
"""

import numpy as np

def pad_or_truncate(
    audio_reference,
    audio_estimates
):
    """Pad or truncate estimates by duration of references:
    - If reference > estimates: add zeros at the and of the estimated signal
    - If estimates > references: truncate estimates to duration of references
    Parameters
    ----------
    references : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing true reference sources
    estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing estimated sources
    Returns
    -------
    references : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing true reference sources
    estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing estimated sources
    """
    est_shape = audio_estimates.shape
    ref_shape = audio_reference.shape
    if est_shape[1] != ref_shape[1]:
        if est_shape[1] >= ref_shape[1]:
            audio_estimates = audio_estimates[:, :ref_shape[1], :]
        else:
            # pad end with zeros
            audio_estimates = np.pad(
                audio_estimates,
                [
                    (0, 0),
                    (0, ref_shape[1] - est_shape[1]),
                    (0, 0)
                ],
                mode='constant'
            )

    return audio_reference, audio_estimates

def eval_silent_frames(true_source, predicted_source, window_size: int, hop_size: int, eval_incomplete_last_frame=False,
                       eps_for_silent_target=True):
    """
    :param true_source: true source signal in the time domain, numpy array with shape (nsources, nsamples, nchannels)
    :param predicted_source: predicted source signal in the time domain, numpy array with shape (nsources, nsamples, nchannels)
    :param window_size: length (in samples) of the window used for the framewise bss_eval metrics computation
    :param hop_size: hop size (in samples) used for the framewise bss_eval metrics computation
    :param eval_incomplete_last_frame: if True, takes last frame into account even if it is shorter than the window,
    default: False
    :param eps_for_silent_target: if True, returns a value also if target source is silent, set to False for exact same
    behaviour as explained in the paper "Weakly Informed Audio Source Separation", default: True
    :return: pes: numpy array containing PES values for all passed sources,
                  mean over evaluation windows, mean over channels. shape=(nsources,)
             eps: numpy array containing EPS values for all passed sources.
                  mean over evaluation windows, mean over channels. shape=(nsources,)
             silent_true_source_frames: list of indices of frames with silent target source
             silent_prediction_frames: list of indices of frames with silent predicted source
    """

    # check inputs
    if true_source.ndim == 1:
        assert len(true_source) == len(predicted_source), "true source and predicted source must have same length if " \
                                                          "one-dimensional signals are passed"
        true_source = np.expand_dims(np.expand_dims(true_source, axis=0), axis=-1)
        predicted_source = np.expand_dims(np.expand_dims(predicted_source, axis=0), axis=-1)

    else:
        true_source, predicted_source = pad_or_truncate(true_source, predicted_source)

    nb_sources, signal_length, nb_channels = true_source.shape

    # compute number of evaluation frames
    number_eval_frames = int(np.ceil((signal_length - window_size) / hop_size)) + 1

    # prepare lists that will be filled with values and returned
    pes = []
    eps = []

    for src in range(nb_sources):

        # hold values for all channels of one source
        pes_list_source = []
        eps_list_source = []

        for ch in range(nb_channels):

            prediction = predicted_source[src, :, ch]
            truth = true_source[src, :, ch]

            last_frame_incomplete = False
            if signal_length % hop_size != 0:
                last_frame_incomplete = True

            # values for each frame will be gathered here
            pes_list_channel = []
            eps_list_channel = []

            # indices of frames with silence will be gathered here
            silent_true_source_frames = []
            silent_prediction_frames = []

            for n in range(number_eval_frames):

                # evaluate last frame if applicable
                if n == number_eval_frames - 1 and last_frame_incomplete:
                    if eval_incomplete_last_frame:
                        prediction_window = prediction[n * hop_size:]
                        true_window = truth[n * hop_size:]
                    else:
                        continue

                # evaluate other frames
                else:
                    prediction_window = prediction[n * hop_size: n * hop_size + window_size]
                    true_window = truth[n * hop_size: n * hop_size + window_size]

                # compute Predicted Energy at Silence (PES)
                if sum(abs(true_window)) == 0:
                    pes_win = 10 * np.log10(sum(prediction_window ** 2) + 10 ** (-12))
                    pes_list_channel.append(pes_win)
                    silent_true_source_frames.append(n)

                # compute Energy at Predicted Silence (EPS)
                if eps_for_silent_target:
                    if sum(abs(prediction_window)) == 0:
                        true_source_energy_at_silent_prediction = 10 * np.log10(sum(true_window ** 2) + 10 ** (-12))
                        eps_list_channel.append(true_source_energy_at_silent_prediction)
                        silent_prediction_frames.append(n)

                else:
                    if sum(abs(prediction_window)) == 0 and sum(abs(true_window)) != 0:
                        true_source_energy_at_silent_prediction = 10 * np.log10(sum(true_window ** 2) + 10 ** (-12))
                        eps_list_channel.append(true_source_energy_at_silent_prediction)
                        silent_prediction_frames.append(n)

            # take mean over all evaluation windows of current channel if values exist
            if len(pes_list_channel) != 0:
                pes_channel = np.mean(pes_list_channel)
                pes_list_source.append(pes_channel)
            if len(eps_list_channel) != 0:
                eps_channel = np.mean(eps_list_channel)
                eps_list_source.append(eps_channel)

        # take mean over channels of one source
        if len(pes_list_source) != 0:
            pes_source = np.mean(pes_list_source)
            pes.append(pes_source)
        else:
            pes.append(None)
        if len(eps_list_source) != 0:
            eps_source = np.mean(eps_list_source)
            eps.append(eps_source)
        else:
            eps.append(None)

    return np.array(pes), np.array(eps), silent_true_source_frames, silent_prediction_frames

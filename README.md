# Phoneme Level Lyrics Alignment and Text-Informed Singing Voice Separation

This is the code for the paper

> Schulze-Forster, K., Doire, C., Richard, G., & Badeau, R. "Phoneme Level Lyrics Alignment and Text-Informed Singing Voice Separation." *IEEE/ACM Transactions on Audio, Speech and Language Processing* (2021).  
>doi: [10.1109/TASLP.2021.3091817](https://doi.org/10.1109/TASLP.2021.3091817)

If you use parts of the code in your work, please cite the paper.

## Links
[:page_facing_up: Publicly available paper print](https://hal.telecom-paris.fr/hal-03255334)  
[:loud_sound: Audio examples](https://schufo.github.io/plla_tisvs/)  
[:memo: MUSDB18 lyrics transcripts](https://doi.org/10.5281/zenodo.3989267)

## Installation
Clone the repository to your machine:
<pre>
git clone https://github.com/schufo/plla-tisvs.git
</pre>

The project was implemented using Python 3.6 and a conda environment.
Your can create the environment with all dependencies with the following command:
<pre>
conda env create -f environment.yml
</pre>

Then activate the environment:
<pre>
conda activate plla_tisvs
</pre>

## Data preparation

To prepare the TIMIT and MUSDB audio and text data, the preprocessing scripts must be run in the indicated order.
Note that the paths to the datasets must be adapted by the user in the scripts.

## Training

Models for joint alignment and separation can be trained as follows:
<pre>
python train.py --tag 'pre_trained_joint' --architecture 'InformedOpenUnmix3' --attention 'dtw' --dataset 'timit_music' --text-units 'cmu_phonemes' --epochs 66 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001  --comment 'pre training on speech music mixtures'

python train.py --tag 'JOINT1' --architecture 'InformedOpenUnmix3' --wst-model 'pre_trained_joint' --attention 'dtw' --dataset 'musdb_lyrics' --text-units 'cmu_phonemes' --space-token-only --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment '...'

python train.py --tag 'JOINT2' --architecture 'InformedOpenUnmix3' --wst-model 'pre_trained_joint' --attention 'dtw' --dataset 'blended' --speech-examples 1000 --text-units 'cmu_phonemes' --space-token-only --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'like JOINT1 but added speech examples to trainig set'

python train.py --tag 'JOINT3' --architecture 'InformedOpenUnmix3' --wst-model 'pre_trained_joint' --attention 'dtw' --dataset 'blended' --speech-examples 1000 --text-units 'cmu_phonemes' --space-token-only --add-silence --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'like JOINT2 but added silence to singing voice examples'

python train.py --tag 'JOINT_SP' --architecture 'InformedOpenUnmix3' --attention 'dtw' --dataset 'timit_music' --text-units 'cmu_phonemes' --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'trained only on speech-music mixtures'
</pre>

Before the separation models can be trained using aligned lyrics, the alignments need to be available.
Alignments can be obtained from the above models using the script save_alignment_paths.py. 
Separation models can then be trained as follows:

<pre>
python train.py --tag 'SEQ' --architecture 'InformedOpenUnmix3NA2' --dataset 'musdb_lyrics' --text-units 'cmu_phonemes' --alignment-from 'JOINT3' --space-token-only --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'informed with aligned text'

python train.py --tag 'SEQ_BL1' --architecture 'InformedOpenUnmix3NA2' --dataset 'musdb_lyrics' --text-units 'ones' --alignment-from 'JOINT3' --fake-alignment --space-token-only --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'informed with a constant representation and a constant alignment'

python train.py --tag 'SEQ_BL2' --architecture 'InformedOpenUnmix3NA2' --dataset 'musdb_lyrics' --text-units 'voice_activity' --alignment-from 'JOINT3' --space-token-only --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'informed with voice activity information derived from aligned text'
</pre>

## Evaluation

### Separation
<pre>
python eval_separation.py --tag 'SEQ' --testset 'musdb_lyrics' --test-snr 5

# the original mixture is used as input when no test SNR is specified
python eval_separation.py --tag 'SEQ' --testset 'musdb_lyrics'
</pre>

### Alignment
1. Compute and save alignments:
<pre>
python estimate_alignment.py --tag 'JOINT3' --testset 'Hansen'

python estimate_alignment.py --tag 'JOINT3' --testset 'NUS_acapella'

python estimate_alignment.py --tag 'JOINT3' --testset 'NUS' --snr 0
</pre>

2. Compute alignment evaluation scores using eval_alignment.py

## Acknowledgment

This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 765068.

## Copyright

Copyright 2021 Kilian Schulze-Forster of Télécom Paris, Institut Polytechnique de Paris.
All rights reserved.

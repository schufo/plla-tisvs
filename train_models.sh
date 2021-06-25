

python train.py --tag 'pre_trained_joint' --architecture 'InformedOpenUnmix3' --attention 'dtw' --dataset 'timit_music' --text-units 'cmu_phonemes' --epochs 66 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001  --comment 'pre training on speech music mixtures'

python train.py --tag 'JOINT1' --architecture 'InformedOpenUnmix3' --wst-model 'pre_trained_joint' --attention 'dtw' --dataset 'musdb_lyrics' --text-units 'cmu_phonemes' --space-token-only --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment '...'

python train.py --tag 'JOINT2' --architecture 'InformedOpenUnmix3' --wst-model 'pre_trained_joint' --attention 'dtw' --dataset 'blended' --speech-examples 1000 --text-units 'cmu_phonemes' --space-token-only --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'like JOINT1 but added speech examples to trainig set'

python train.py --tag 'JOINT3' --architecture 'InformedOpenUnmix3' --wst-model 'pre_trained_joint' --attention 'dtw' --dataset 'blended' --speech-examples 1000 --text-units 'cmu_phonemes' --space-token-only --add-silence --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'like JOINT2 but added silence to singing voice examples'

python train.py --tag 'JOINT_SP' --architecture 'InformedOpenUnmix3' --attention 'dtw' --dataset 'timit_music' --text-units 'cmu_phonemes' --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'trained only on speech-music mixtures'

python train.py --tag 'SEQ' --architecture 'InformedOpenUnmix3NA2' --dataset 'musdb_lyrics' --text-units 'cmu_phonemes' --alignment-from 'JOINT3' --space-token-only --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'informed with aligned text'

python train.py --tag 'SEQ_BL1' --architecture 'InformedOpenUnmix3NA2' --dataset 'musdb_lyrics' --text-units 'ones' --alignment-from 'JOINT3' --fake-alignment --space-token-only --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'informed with a constant representation and a constant alignment'

python train.py --tag 'SEQ_BL2' --architecture 'InformedOpenUnmix3NA2' --dataset 'musdb_lyrics' --text-units 'voice_activity' --alignment-from 'JOINT3' --space-token-only --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'informed with voice activity information derived from aligned text'

python train.py --tag 'UMX1' --architecture 'OpenUnmix' --dataset 'musdb_lyrics' --text-units 'cmu_phonemes' --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'original Open Unmix model trained on the MUSDB lyrics dataset'

python train.py --tag 'UMX_pre_trained' --architecture 'OpenUnmix' --dataset 'timit_music' --text-units 'cmu_phonemes' --epochs 66 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'pre-trained original Open Unmix model on speech-musix mixtures'

python train.py --tag 'UMX2' --architecture 'OpenUnmix' --wst-model 'UMX_pre_trained' --dataset 'blended' --text-units 'cmu_phonemes' --speech-examples 1000 --add-silence --epochs 2000 --batch-size 16 --nb-channels 1 --nb-workers 4 --samplerate 16000 --nfft 512 --nhop 256 --weight-decay 0 --lr 0.001 --comment 'continue training pre-trained Open Unmix with added speech and added silence'

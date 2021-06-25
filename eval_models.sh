
# ---------------- Separation evaluation --------------------------------
python eval_separation.py --tag 'SEQ' --testset 'musdb_lyrics' --test-snr 5

# the original mixture is used as input when no test SNR is specified
python eval_separation.py --tag 'SEQ' --testset 'musdb_lyrics'


# ---------------- Alignment evaluation --------------------------------

python estimate_alignment.py --tag 'JOINT3' --testset 'Hansen'

python estimate_alignment.py --tag 'JOINT3' --testset 'NUS_acapella'

python estimate_alignment.py --tag 'JOINT3' --testset 'NUS' --snr 0

# to compute several evaluation metrics based on the estimates,
# the script eval_alignment can be used.
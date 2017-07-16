import sys
import theano.sandbox.cuda
theano.sandbox.cuda.use(sys.argv[1] if len(sys.argv) > 1 else 'gpu0')

from IQA_BIECON_release import train_iqa as tm

tm.train_biecon(
    config_file='IQA_BIECON_release/configs/NR_biecon.yaml',
    section='base_LIVE',
    tr_te_file='outputs/tr_te_live.txt',
    snap_path='outputs/NR/BIECON_exp/',
    epoch_loc=40, epoch_nr=100
)

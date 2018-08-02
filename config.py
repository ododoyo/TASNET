# config file for training or other process
# check the config file before your process
# and import this file to get parameters

# data config
samp_rate = 8000
frame_duration = 0.005
frame_size = int(samp_rate * frame_duration)
shift = frame_size
overlap_rate = 0
batch_size = 32
dev_batch_size = 32
min_queue_size = 64
load_file_num = 100
min_sent_len = 10
shorter_sent_len = 100
longer_sent_len = 800

# data_path
train_spkr1_list = "/home/fsl/workspace/SpeechSeparation/TASNET/list/wsj0_tr_spkr1.lst"
train_spkr2_list = "/home/fsl/workspace/SpeechSeparation/TASNET/list/wsj0_tr_spkr2.lst"
train_mixsnr_list = "/home/fsl/workspace/SpeechSeparation/TASNET/list/wsj0_tr_mixsnr.lst"
dev_spkr1_list = "/home/fsl/workspace/SpeechSeparation/TASNET/list/wsj0_cv_spkr1.lst"
dev_spkr2_list = "/home/fsl/workspace/SpeechSeparation/TASNET/list/wsj0_cv_spkr2.lst"
dev_mixsnr_list = "/home/fsl/workspace/SpeechSeparation/TASNET/list/wsj0_cv_mixsnr.lst"
debug_spkr1_list = "/home/fsl/workspace/SpeechSeparation/TASNET/list/wsj0_debug_spkr1.lst"
debug_spkr2_list = "/home/fsl/workspace/SpeechSeparation/TASNET/list/wsj0_debug_spkr2.lst"
debug_mixsnr_list = "/home/fsl/workspace/SpeechSeparation/TASNET/list/wsj0_debug_mixsnr.lst"

# job config
job_type = "train"
job_dir = "job/TASNET_trial"
gpu_list = [0]

# model config
num_basis = 500
num_layers = 4
hidden_size = 500  # num_units in uni-directional cell
bidirectional = True

# training param
seed = 123
resume = True
init_mean = 0.0
init_stddev = 0.02
max_grad_norm = 200
learning_rate = 1e-3
max_epoch = 100
pretrain_shorter_epoch = 5
log_period = 5
save_period = 500
dev_period = 500
early_stop_count = 10
decay_lr_count = 3
decay_lr = 0.5
min_learning_rate = 1e-6

# test config
test_spkr1_list = '/home/fsl/workspace/SpeechSeparation/TASNET/list/wsj0_tt_spkr1.lst'
test_spkr2_list = '/home/fsl/workspace/SpeechSeparation/TASNET/list/wsj0_tt_spkr2.lst'
test_mixsnr_list = '/home/fsl/workspace/SpeechSeparation/TASNET/list/wsj0_tt_mixsnr.lst'
test_name = 'wsj-test'

# load option
load_option = 1
load_path = ""

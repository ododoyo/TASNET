import os
import sys
import logging
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import traceback
import numpy as np
import tensorflow as tf
import soundfile as sf
import config as cfg
from utils.tools import *
from speech_data import SpeechReader
from model.TASNET import TASmodel
from mir_eval import bss_eval_sources

tf.logging.set_verbosity(tf.logging.ERROR)


if __name__ == "__main__":
    gpu_list = cfg.gpu_list
    num_gpu = 1  # ues one gpu to evaluate by default
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    tf.set_random_seed(cfg.seed)
    job_dir = cfg.test_name
    mix_dir = os.path.join(job_dir, 'mix')
    s1_dir = os.path.join(job_dir, 's1')
    s2_dir = os.path.join(job_dir, 's2')
    s1_est_dir = os.path.join(job_dir, 's1.est')
    s2_est_dir = os.path.join(job_dir, 's2.est')
    create_folders(mix_dir)
    create_folders(s1_dir)
    create_folders(s2_dir)
    create_folders(s1_est_dir)
    create_folders(s2_est_dir)
    s1_list = get_file_line(cfg.test_spkr1_list)
    s2_list = get_file_line(cfg.test_spkr2_list)
    snr_list = get_file_line(cfg.test_mixsnr_list)
    snr_ans = []
    bss_ans = []
    pesq_ans = []
    try:
        with tf.Graph().as_default():
            sess_config = tf.ConfigProto()
            sess_config.allow_soft_placement = True
            sess_config.gpu_options.allow_growth = True
            sess = tf.Session(config=sess_config)
            initializer = tf.random_normal_initializer(mean=cfg.init_mean,
                                                       stddev=cfg.init_stddev)
            with tf.variable_scope("TASNET", initializer=initializer):
                model = TASmodel(sess, cfg, num_gpu, initializer)
                model.restore_model()
                for s1_file, s2_file, mix_snr in zip(s1_list, s2_list, snr_list):
                    s1_file = s1_file.strip()
                    s2_file = s2_file.strip()
                    mix_snr = float(mix_snr.strip())
                    name = os.path.basename(s1_file)
                    s1_sig = read_wav(s1_file, samp_rate=cfg.samp_rate)
                    s2_sig = read_wav(s2_file, samp_rate=cfg.samp_rate)
                    seg_mix, seg_src, seq_len, mix_sig = prepare_feat(
                        s1_sig, s2_sig, cfg.frame_size, cfg.shift)
                    inputs = np.array([[seg_mix]])
                    targets = np.array([[seg_src]])
                    feed_seq_len = np.array([[seq_len]])
                    batch_data = (inputs, targets, feed_seq_len)
                    recon = model.get_reorder_recon(batch_data)
                    recon = recon[0][0]
                    recon_sig1 = recon[0].reshape((-1))[0:len(s1_sig)]
                    recon_sig2 = recon[1].reshape((-1))[0:len(s2_sig)]
                    audiowrite(os.path.join(mix_dir, name), mix_sig, samp_rate=cfg.samp_rate)
                    audiowrite(os.path.join(s1_dir, name), s1_sig, samp_rate=cfg.samp_rate)
                    audiowrite(os.path.join(s2_dir, name), s2_sig, samp_rate=cfg.samp_rate)
                    audiowrite(os.path.join(s1_est_dir, name), recon_sig1, samp_rate=cfg.samp_rate)
                    audiowrite(os.path.join(s2_est_dir, name), recon_sig2, samp_rate=cfg.samp_rate)
                    snr1 = get_SISNR(s1_sig, recon_sig1)
                    snr2 = get_SISNR(s2_sig, recon_sig2)
                    snr_ans.append([snr1, snr2])
                    src_ref = np.stack([s1_sig, s2_sig], axis=0)
                    src_est = np.stack([recon_sig1, recon_sig2], axis=0)
                    src_anchor = np.stack([mix_sig, mix_sig], axis=0)
                    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
                    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
                    bss_ans.append(np.concatenate(
                        [sdr, sir, sar, popt, sdr0, sir0, sar0, popt0], axis=0))
                    pesq1 = getPESQ(os.path.join(s1_dir, name),
                                    os.path.join(s1_est_dir, name), cfg.samp_rate)
                    pesq2 = getPESQ(os.path.join(s2_dir, name),
                                    os.path.join(s2_est_dir, name), cfg.samp_rate)
                    pesq1, pesq2 = float(pesq1), float(pesq2)
                    pesq_ans.append([pesq1, pesq2])
                    print("Sentence {}".format(name))
                    print("snr1: {}, snr2: {}".format(snr1, snr2))
                    print("sdr1: {}, sdr2: {}".format(sdr[0], sdr[1]))
                    print("pesq1: {}, pesq2: {}".format(pesq1, pesq2))
            bss_ans = np.array(bss_ans)
            snr_ans = np.array(snr_ans)
            pesq_ans = np.array(pesq_ans)
            np.savetxt("snr.ans", snr_ans, fmt="%.4f")
            np.savetxt("bss.ans", bss_ans, fmt="%.4f")
            np.savetxt("pesq.ans", pesq_ans, fmt="%.4f")
            print("mean of snr: {}".format(np.mean(snr_ans)))
            print("mean of sdr: {}".format(np.mean(bss_ans[:, 0:2])))
            print("mean of pesq: {}".format(np.mean(pesq_ans)))
    except Exception as e:
        logging.error("evaluating exception: %s" % e)
        traceback.print_exc()

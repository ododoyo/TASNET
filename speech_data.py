import os
import sys
import logging
import traceback
import random
import time
import threading
import numpy as np
import config as cfg
try:
    from Queue import Queue
except ImportError:
    from queue import Queue
from utils.tools import *


class Producer(threading.Thread):
    def __init__(self, reader):
        threading.Thread.__init__(self)
        self.reader = reader
        self.exitcode = 0
        self.stop_flag = False

    def run(self):
        try:
            min_queue_size = self.reader._config.min_queue_size
            while not self.stop_flag:
                idx = self.reader._next_load_idx
                if idx >= len(self.reader.data_list):
                    self.reader._batch_queue.put([])
                    break
                if self.reader._batch_queue.qsize() < min_queue_size:
                    batch_list = self.reader.load_samples()
                    for batch in batch_list:
                        self.reader._batch_queue.put(batch)
                else:
                    time.sleep(1)
        except Exception as e:
            logging.warning("producer exception: %s" % e)
            self.exitcode = 1
            traceback.print_exc()

    def stop(self):
        self.stop_flag = True

class SpeechReader(object):
    def __init__(self, config, data_list, batch_size=None, max_sent_len=-1,
                 min_sent_len=10, num_gpu=1, job_type='train'):
        self.num_gpu = num_gpu
        self.max_sent_len = max_sent_len
        self.min_sent_len = min_sent_len
        self.num_speakers = 2
        self.batch_size = batch_size
        if batch_size is None:
            self.batch_size = config.batch_size
        self.eps = 1e-8
        self._config = config
        self.data_list = self.read_data_list(data_list)
        self._job_type = job_type
        self._batch_queue = Queue()
        self.reset()

    def reset(self):
        self.sample_buffer = []
        self._next_load_idx = 0
        if self._job_type == "train":
            self.shuffle_data_list()
        self._producer = Producer(self)
        self._producer.start()

    def shuffle_data_list(self):
        random.shuffle(self.data_list)

    def get_file_line(self, file_path):
        line_list = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.strip().split()[0]
                line_list.append(line)
        return line_list

    def read_data_list(self, data_list):
        s1_file, s2_file, snr_file = data_list
        s1_list = self.get_file_line(s1_file)
        s2_list = self.get_file_line(s2_file)
        snr_list = self.get_file_line(snr_file)
        tuple_list = list(zip(s1_list, s2_list, snr_list))
        return tuple_list

    def mix2signal(self, sig1, sig2, snr):
        eps = self.eps
        sig1 = np.array(sig1, dtype=np.float32)
        sig2 = np.array(sig2, dtype=np.float32)
        min_len = min(len(sig1), len(sig2))
        sig1 = sig1[0:min_len]
        sig2 = sig2[0:min_len]

        w = 10 ** (snr / 20.0)
        # norm to unit energy
        sig1 = w * sig1 / (np.sqrt(np.sum(sig1 ** 2)) + eps)
        sig2 = sig2 / (np.sqrt(np.sum(sig2 ** 2)) + eps)
        mix = sig1 + sig2
        amp_max = max(max(np.abs(sig1)), max(np.abs(sig2)), max(np.abs(mix)), eps)
        sig1 = sig1 / amp_max * 0.9
        sig2 = sig2 / amp_max * 0.9
        mix = mix / amp_max * 0.9
        return mix, sig1, sig2

    def unit_norm(self, sig):
        eps = self.eps
        norm_w = np.sqrt(np.sum(sig ** 2)) + eps
        sig = sig / norm_w
        return sig, norm_w

    def check_zero_energy(self, src_seg):
        return False
        tmp1 = np.sum(src_seg, axis=(1, 2))
        tmp2 = np.sum(tmp1 == 0)
        return tmp2 != 0

    def load_one_mixture(self, tuple_file):
        sample_list = []
        s1_file, s2_file, mix_snr = tuple_file
        mix_snr = float(mix_snr)
        samp_rate = self._config.samp_rate
        frame_size = self._config.frame_size
        shift = self._config.shift
        # for pcm file
        # s1_sig, _ = read_raw_pcm(s1_file, channels=1, samplerate=samp_rate, subtype='PCM_16')
        # s2_sig, _ = read_raw_pcm(s2_file, channels=1, samplerate=samp_rate, subtype='PCM_16')
        # for wav file
        s1_sig = read_wav(s1_file, samp_rate=samp_rate)
        s2_sig = read_wav(s2_file, samp_rate=samp_rate)
        min_len = min(len(s1_sig), len(s2_sig))
        seq_len = samples_to_segment_len(min_len, frame_size, shift)
        if seq_len < self.min_sent_len:
            return []
        # mix_sig, s1_sig, s2_sig = self.mix2signal(s1_sig, s2_sig, mix_snr)
        mix_sig = s1_sig + s2_sig  # wsj0 corpus has been pre-processed for mixing
        seg_mix = segment_signal(mix_sig, frame_size, shift)
        seg_s1 = segment_signal(s1_sig, frame_size, shift)
        seg_s2 = segment_signal(s2_sig, frame_size, shift)
        seg_src = np.stack([seg_s1, seg_s2], axis=0)
        i = 0
        while self.max_sent_len > 0 and i + self.max_sent_len <= seq_len:
            one_sample = (seg_mix[i:i+self.max_sent_len],
                          seg_src[:, i:i+self.max_sent_len, :],
                          self.max_sent_len)
            sample_list.append(one_sample)
            i += (1 - self._config.overlap_rate) * self.max_sent_len
        if seq_len - i >= self.min_sent_len and self._job_type != "train":
            one_sample = (seg_mix[i:], seg_src[:, i:, :], seq_len - i)
            sample_list.append(one_sample)
        return sample_list

    def patch_batch_data(self):
        batch_size = self.batch_size
        group_size = batch_size * self.num_gpu
        feat_dim = self._config.frame_size
        num_groups = len(self.sample_buffer) // group_size
        if num_groups == 0:
            return []
        group_list = []
        choose_samples = [self.sample_buffer[i:i+group_size]
                          for i in range(0, group_size * num_groups, group_size)]
        self.sample_buffer = self.sample_buffer[group_size * num_groups:]
        for one_group in choose_samples:
            group_seg_mix = []
            group_seg_src = []
            group_seq_len = []
            for i in range(0, group_size, batch_size):
                one_batch = one_group[i:i+batch_size]
                max_len =  int(max(map(lambda x: x[2], one_batch)))
                batch_seg_mix = np.zeros((batch_size, max_len, feat_dim), dtype=np.float32)
                batch_seg_src = np.zeros((batch_size, self.num_speakers, max_len, feat_dim),
                                         dtype=np.float32)
                batch_seq_len = np.zeros(batch_size, dtype=np.int32)
                for j in range(batch_size):
                    this_len = one_batch[j][2]
                    batch_seq_len[j] = this_len
                    batch_seg_mix[j, 0:this_len, :] = one_batch[j][0]
                    batch_seg_src[j, :, 0:this_len, :] = one_batch[j][1]
                group_seg_mix.append(batch_seg_mix)
                group_seg_src.append(batch_seg_src)
                group_seq_len.append(batch_seq_len)
            group_list.append((group_seg_mix, group_seg_src, group_seq_len))
        return group_list

    def load_samples(self):
        load_file_num = self._config.load_file_num
        idx = self._next_load_idx
        for tuple_file in self.data_list[idx: idx+load_file_num]:
            self.sample_buffer.extend(self.load_one_mixture(tuple_file))
        self._next_load_idx += load_file_num
        if self._job_type == "train":
            random.shuffle(self.sample_buffer)
        group_list = self.patch_batch_data()
        return group_list

    def next_batch(self):
        while self._producer.exitcode == 0:
            try:
                batch_data = self._batch_queue.get(block=False)
                if len(batch_data) == 0:
                    return None
                else:
                    return batch_data
            except Exception as e:
                time.sleep(3)


def test():
    data_list = (cfg.train_spkr1_list, cfg.train_spkr2_list, cfg.train_mixsnr_list)
    start_time = time.time()
    reader = SpeechReader(cfg, data_list, max_sent_len=200, min_sent_len=10,
                          num_gpu=1, job_type="test")
    batch_data = reader.next_batch()
    seg_mix, seg_src, seq_len = batch_data
    print("seg_mix.shape: ", seg_mix[0].shape, seg_mix[0].dtype)
    print("seg_src.shape: ", seg_src[0].shape, seg_src[0].dtype)
    print("seq_len.shape: ", seq_len[0].shape, seq_len[0].dtype)
    for i in range(99):
        batch_data = reader.next_batch()
    duration = time.time() - start_time
    print("read 100 batches consume {:.2f} seconds".format(duration))
    reader._producer.stop()

def check_dev():
    data_list = (cfg.dev_spkr1_list, cfg.dev_spkr2_list, cfg.dev_mixsnr_list)
    reader = SpeechReader(cfg, data_list, max_sent_len=-1,
                          min_sent_len=cfg.min_sent_len, num_gpu=1,
                          job_type='dev')
    for i in range(39):
        batch_data = reader.next_batch()
    batch_input, batch_seg_src, seq_len = batch_data
    batch_input = batch_input[0]
    batch_seg_src = batch_seg_src[0]
    seq_len = seq_len[0]
    debug_dir = "debug"
    os.makedirs(debug_dir, exist_ok=True)
    s1_dir = os.path.join(debug_dir, 's1')
    s2_dir = os.path.join(debug_dir, 's2')
    mix_dir = os.path.join(debug_dir, 'mix')
    os.makedirs(s1_dir, exist_ok=True)
    os.makedirs(s2_dir, exist_ok=True)
    os.makedirs(mix_dir, exist_ok=True)
    for i in range(len(batch_input)):
        input_data = batch_input[i][0:seq_len[i]].reshape((-1))
        src_data = batch_seg_src[i][:, 0:seq_len[i], :].reshape((2, -1))
        name = "%03d.data" % i
        np.savetxt(os.path.join(mix_dir, name), input_data)
        np.savetxt(os.path.join(s1_dir, name), src_data[0])
        np.savetxt(os.path.join(s2_dir, name), src_data[1])
    np.savetxt(os.path.join(debug_dir, 'seq.len'), seq_len, fmt='%d')
    reader._producer.stop()

if __name__ == "__main__":
    test()
    # check_dev()

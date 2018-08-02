import os
import sys
project_path = os.path.abspath('..')
sys.path.append(project_path)

import logging
import time
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from utils.tools import *


class TASmodel(object):
    def __init__(self, sess, config, num_gpu, initializer=None):
        self.session = sess
        self.config = config
        self.num_gpu = num_gpu
        self.epoch_counter = 0
        self.num_speakers = 2
        self.initializer = initializer
        self.eps = 1e-7
        self.global_step = tf.get_variable("global_step", shape=[], trainable=False,
                                           initializer=tf.constant_initializer(0),
                                           dtype=tf.int32)
        # define placeholder
        self.lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.create_placeholder()
        self.training = tf.placeholder(tf.bool, shape=[])
        # init graph
        self.optimize()
        self.reset()
        # create job_env
        self.job_dir = config.job_dir
        create_folders(self.job_dir)
        self.best_snr_dir = os.path.join(config.job_dir, 'bset_snr')
        create_folders(self.best_snr_dir)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        self.best_snr_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        train_event_dir = os.path.join(config.job_dir, 'train_event')
        dev_event_dir = os.path.join(config.job_dir, 'dev_event')
        create_folders(train_event_dir)
        create_folders(dev_event_dir)
        self.train_writer = tf.summary.FileWriter(train_event_dir, sess.graph)
        self.dev_writer = tf.summary.FileWriter(dev_event_dir)

    def create_placeholder(self):
        self._input = []
        self._target = []
        self._seq_len = []
        feat_dim = self.config.frame_size
        for i in range(self.num_gpu):
            self._input.append(tf.placeholder(tf.float32, shape=[None, None, feat_dim]))
            self._target.append(tf.placeholder(tf.float32, shape=[None, self.num_speakers, None, feat_dim]))
            self._seq_len.append(tf.placeholder(tf.int32, shape=[None]))

    def reset(self):
        self.batch_counter = 0
        self.total_snr = 0
        self.latest_snr = 0
        self.latest_batch_counter = 0
        self.epoch_counter += 1

    def encode(self, inputs):
        # gate-conv1d style to encode segment inputs into mixture weights
        # return mixture weights and unit-norm coefs for reconstruction
        num_basis = self.config.num_basis
        feat_dim = self.config.frame_size
        batch_size = tf.shape(inputs)[0]
        # shape is [B, T, F]
        max_len = tf.shape(inputs)[1]
        with tf.variable_scope("encoder"):
            basis_kernel = tf.get_variable("encoder_basis", shape=[feat_dim, 1, num_basis], dtype=tf.float32)
            gate_kernel = tf.get_variable("encoder_gate", shape=[feat_dim, 1, num_basis], dtype=tf.float32)
            norm_coef = tf.sqrt(tf.reduce_sum(inputs ** 2, axis=2, keep_dims=True) + self.eps)
            norm_inputs = inputs / norm_coef
            reshape_inputs = tf.expand_dims(tf.reshape(norm_inputs, [-1, feat_dim]), axis=2)
            conv_inputs = tf.nn.relu(tf.nn.conv1d(reshape_inputs, basis_kernel, stride=1, padding='VALID'))
            gate = tf.nn.sigmoid(tf.nn.conv1d(reshape_inputs, gate_kernel, stride=1, padding='VALID'))
            mixture_w = tf.reshape(conv_inputs * gate, [batch_size, max_len, num_basis], name='mixture_w')
        return mixture_w, norm_coef

    def separate(self, mixture_w, seq_len):
        num_basis = self.config.num_basis
        num_layers = self.config.num_layers
        hidden_size = self.config.hidden_size
        rnn_cell = tf.contrib.rnn.LSTMCell
        batch_size = tf.shape(mixture_w)[0]
        max_len = tf.shape(mixture_w)[1]
        with tf.variable_scope("separator"):
            norm_mix_w = layer_normalization(mixture_w, num_basis, axis=2, eps=self.eps)
            lstm_input = norm_mix_w
            for i in range(num_layers):
                if self.config.bidirectional:
                    fw_cell = rnn_cell(hidden_size, use_peepholes=True, cell_clip=25, state_is_tuple=True)
                    bw_cell = rnn_cell(hidden_size, use_peepholes=True, cell_clip=25, state_is_tuple=True)
                    initial_fw = fw_cell.zero_state(tf.shape(lstm_input)[0], dtype=tf.float32)
                    initial_bw = bw_cell.zero_state(tf.shape(lstm_input)[0], dtype=tf.float32)
                    output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, lstm_input,
                                                                sequence_length=tf.to_int32(seq_len),
                                                                initial_state_fw=initial_fw,
                                                                initial_state_bw=initial_bw,
                                                                scope='lstm_%d'%(i+1), dtype=tf.float32,
                                                                time_major=False)
                    output = tf.concat(output, axis=2)
                    lstm_input = output
                else:
                    fw_cell = rnn_cell(hidden_size, use_peepholes=True, cell_clip=25, state_is_tuple=True)
                    initial_fw = fw_cell.zero_state(tf.shape(lstm_input)[0], dtype=tf.float32)
                    output, _ = tf.nn.dynamic_rnn(fw_cell, lstm_input,
                                                  sequence_length=tf.to_int32(seq_len),
                                                  initial_state=initial_fw, dtype=tf.float32,
                                                  scope='lstm_%d'%(i+1), time_major=False)
                    lstm_input = output
            out_dims = hidden_size * 2 if self.config.bidirectional else hidden_size
            output = tf.reshape(output, [-1, out_dims])
            output = tcl.fully_connected(inputs=output, num_outputs=self.num_speakers*num_basis, activation_fn=None)
            masks = tf.reshape(output, [batch_size, max_len, self.num_speakers, num_basis])
            masks = tf.nn.softmax(masks, axis=2)
        return masks

    def decode(self, mixture_w, masks):
        num_basis = self.config.num_basis
        feat_dim = self.config.frame_size
        with tf.variable_scope("deocder"):
            expand_mix_w = tf.expand_dims(mixture_w, axis=2)
            source_w = expand_mix_w * masks
            recon_basis = tf.get_variable("recon_basis", shape=[num_basis, feat_dim], dtype=tf.float32)
            source_sig = tf.einsum('btcw,wf->btcf', source_w, recon_basis)
        return source_sig

    def get_seq_mask(self, max_len, seq_len):
        r = tf.range(max_len)
        func = lambda x: tf.cast(tf.less(r, x), dtype=tf.float32)
        seq_mask = tf.map_fn(func, seq_len, dtype=tf.float32)
        return seq_mask

    def get_si_snr(self, targets, recon_sig, seq_len, name='pit_snr'):
        batch_size = tf.shape(targets)[0]
        max_len = tf.shape(targets)[2]
        nc = self.num_speakers
        feat_dim = self.config.frame_size
        # mask the padding part and flat the segmentation
        # zero-mean targets and recon in the real length
        seq_mask = self.get_seq_mask(max_len, seq_len)
        seq_mask = tf.reshape(seq_mask, [batch_size, 1, -1, 1])
        mask_targets = targets * seq_mask
        mask_recon = recon_sig * seq_mask
        sample_count = tf.cast(tf.reshape(seq_len * feat_dim, [batch_size, 1, 1, 1]), tf.float32)
        mean_targets = tf.reduce_sum(mask_targets, axis=[2, 3], keep_dims=True) / sample_count
        mean_recon = tf.reduce_sum(mask_recon, axis=[2, 3], keep_dims=True) / sample_count
        zero_mean_targets = mask_targets - mean_targets * seq_mask
        zero_mean_recon = mask_recon - mean_recon * seq_mask
        # shape is [B, nc, s]
        flat_targets = tf.reshape(zero_mean_targets, [batch_size, nc, -1])
        flat_recon = tf.reshape(zero_mean_recon, [batch_size, nc, -1])
        # calculate the SI-SNR, PIT is necessary
        with tf.variable_scope(name):
            v_perms = tf.constant(
                list(itertools.permutations(range(nc))),
                dtype=tf.int32)
            perms_one_hot = tf.one_hot(v_perms, depth=nc, dtype=tf.float32)
            # shape is [B, 1, nc, s]
            s_truth = tf.expand_dims(flat_targets, axis=1)
            # shape is [B, nc, 1, s]
            s_estimate = tf.expand_dims(flat_recon, axis=2)
            pair_wise_dot = tf.reduce_sum(s_estimate * s_truth, axis=3, keep_dims=True)
            s_truth_energy = tf.reduce_sum(s_truth ** 2, axis=3, keep_dims=True) + self.eps
            pair_wise_proj = pair_wise_dot * s_truth / s_truth_energy
            e_noise = s_estimate - pair_wise_proj
            # shape is [B, nc, nc]
            pair_wise_snr = tf.div(tf.reduce_sum(pair_wise_proj ** 2, axis=3),
                                   tf.reduce_sum(e_noise ** 2, axis=3) + self.eps)
            pair_wise_snr = 10 * tf.log(pair_wise_snr + self.eps) / tf.log(10.0)  # log operation use 10 as base
            snr_set = tf.einsum('bij,pij->bp', pair_wise_snr, perms_one_hot)
            max_snr_idx = tf.cast(tf.argmax(snr_set, axis=1), dtype=tf.int32)
            max_snr = tf.gather_nd(snr_set,
                tf.stack([tf.range(batch_size, dtype=tf.int32), max_snr_idx], axis=1))
            max_snr = max_snr / nc
        return max_snr, v_perms, max_snr_idx

    def tower_cost(self, inputs, targets, seq_len):
        mixture_w, norm_coef = self.encode(inputs)
        masks = self.separate(mixture_w, seq_len)
        # shape is [B, T, nc, F]
        recon_sig = self.decode(mixture_w, masks)
        expand_norm_coef = tf.expand_dims(norm_coef, axis=2)
        # shape is [B, nc, T, F]
        recon_sig = tf.transpose(recon_sig*expand_norm_coef, [0, 2, 1, 3])
        max_snr, v_perms, max_snr_idx = self.get_si_snr(targets, recon_sig, seq_len)
        loss = 0 - tf.reduce_mean(max_snr)
        # get reordered reconstruction signal
        nc = self.num_speakers
        batch_size = tf.shape(recon_sig)[0]
        feat_dim = self.config.frame_size
        tar_perm = tf.gather(v_perms, max_snr_idx)
        tar_perm = tf.transpose(tf.one_hot(tar_perm, nc), [0, 2, 1])
        tar_perm = tf.cast(tf.argmax(tar_perm, axis=2), tf.int32)
        outer_axis = tf.tile(tf.reshape(tf.range(batch_size), [-1, 1]), [1, nc])
        gather_idx = tf.stack([outer_axis, tar_perm], axis=2)
        gather_idx = tf.reshape(gather_idx, [-1, 2])
        reorder_recon = tf.reshape(tf.gather_nd(recon_sig, gather_idx),
                                   [batch_size, nc, -1, feat_dim])
        return loss, max_snr, recon_sig, reorder_recon

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        tower_grads = []
        tower_snr = []
        tower_cost = []
        tower_recon = []
        tower_reorder_recon = []
        for i in range(self.num_gpu):
            worker = '/gpu:%d' % i
            device_setter = tf.train.replica_device_setter(
                worker_device=worker, ps_device='/cpu:0', ps_tasks=1)
            with tf.variable_scope("Model", reuse=(i>0)):
                with tf.device(device_setter):
                    with tf.name_scope("tower_%d" % i) as scope:
                        cost, max_snr, recon_sig, reorder_recon = self.tower_cost(
                            self._input[i], self._target[i], self._seq_len[i])
                        grads = optimizer.compute_gradients(cost)
                        tower_grads.append(grads)
                        tower_snr.append(max_snr)
                        tower_cost.append(cost)
                        tower_recon.append(recon_sig)
                        tower_reorder_recon.append(reorder_recon)
        grads = average_gradients(tower_grads, self.config.max_grad_norm)
        self.apply_gradients_op = optimizer.apply_gradients(grads, global_step=self.global_step)
        self.avg_snr = tf.reduce_mean(tower_snr)
        self.avg_cost = tf.reduce_mean(tower_cost)
        self.tower_recon = tower_recon
        self.tower_snr = tower_snr
        self.tower_reorder_recon = tower_reorder_recon
        tf.summary.scalar('avg_snr', self.avg_snr)
        tf.summary.scalar('avg_cost', self.avg_cost)
        self.merged = tf.summary.merge_all()

    def run_batch(self, group_data, learning_rate):
        feed_dict = {self.training: True, self.lr: learning_rate}
        step_size = 0
        for i in range(self.num_gpu):
            feed_dict[self._input[i]] = group_data[0][i]
            feed_dict[self._target[i]] = group_data[1][i]
            feed_dict[self._seq_len[i]] = group_data[2][i]
            step_size += len(group_data[2][i])
        start_time = time.time()
        _, i_global, i_merge, snr = self.session.run(
            [self.apply_gradients_op, self.global_step, self.merged, self.avg_snr],
            feed_dict=feed_dict)
        self.total_snr += snr
        self.latest_snr += snr
        self.batch_counter += 1
        self.latest_batch_counter += 1
        duration = time.time() - start_time
        if i_global % self.config.log_period == 0:
            logging.info("Epoch {:d}, Average Train SI-SNR: {:.6f}={:.6f}/{:d}, "
                         "Latest SI-SNR: {:.6f}, Speed: {:.2f} sentence/sec".format(
                         self.epoch_counter, self.total_snr / self.batch_counter,
                         self.total_snr, self.batch_counter,
                         self.latest_snr / self.latest_batch_counter,
                         step_size / duration))
            self.latest_batch_counter = 0
            self.latest_snr = 0
            self.train_writer.add_summary(i_merge, i_global)
        if i_global % self.config.save_period == 0:
            self.save_model(i_global)
        return i_global

    def get_reorder_recon(self, group_data):
        feed_dict = {self.training: False}
        for i in range(self.num_gpu):
            feed_dict[self._input[i]] = group_data[0][i]
            feed_dict[self._target[i]] = group_data[1][i]
            feed_dict[self._seq_len[i]] = group_data[2][i]
        reorder_recon = self.session.run(self.tower_reorder_recon, feed_dict=feed_dict)
        return reorder_recon

    def valid(self, reader):
        total_snr, batch_counter = 0.0, 0
        num_sent = 0
        logging.info("Start to dev")
        start_time = time.time()
        while True:
            batch_data = reader.next_batch()
            if batch_data == None:
                break
            else:
                feed_dict = {self.training: False}
                for i in range(self.num_gpu):
                    feed_dict[self._input[i]] = batch_data[0][i]
                    feed_dict[self._target[i]] = batch_data[1][i]
                    feed_dict[self._seq_len[i]] = batch_data[2][i]
                    num_sent += len(batch_data[2][i])
                snr = self.session.run(self.avg_snr, feed_dict=feed_dict)
                total_snr += snr
                batch_counter += 1
                if batch_counter % 10 == 0:
                    logging.info("Dev Sentence {:d}, AVG Dev SI-SNR: {:.6f}={:.6f}/{:d}, "
                                 "Speed: {:.2f} sentence/sec".format(
                                 num_sent, total_snr / batch_counter, total_snr,
                                 batch_counter, num_sent / (time.time() - start_time)))
        duration = time.time() - start_time
        avg_snr = total_snr / batch_counter
        dev_summary = create_valid_summary(avg_snr)
        i_global = self.session.run(self.global_step)
        self.dev_writer.add_summary(dev_summary, i_global)
        logging.info("Finish dev {:d} sentences in {:.2f} seconds, "
                     "AVG SI-SNR: {:.6f}".format(num_sent, duration, avg_snr))
        return avg_snr

    def save_model(self, i_global):
        model_path = os.path.join(self.job_dir, 'model.ckpt')
        self.saver.save(self.session, model_path, global_step=i_global)
        logging.info("Saved model, global_step={}".format(i_global))

    def restore_model(self):
        load_option = self.config.load_option
        if load_option == 0:
            load_path = tf.train.latest_checkpoint(self.job_dir)
        elif load_option == 1:
            load_path = tf.train.latest_checkpoint(self.best_snr_dir)
        else:
            load_path = self.config.load_path
        try:
            self.saver.restore(self.session, load_path)
            logging.info("Loaded model from path {}".format(load_path))
        except Exception as e:
            logging.error("Failed to load model from {}".format(load_path))
            raise e

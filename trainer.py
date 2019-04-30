from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
from six.moves import xrange
import tensorflow as tf

from input_ops import create_input_ops
from config import argparser
from util import log


class Trainer(object):

    def __init__(self, config, model, dataset, dataset_test):
        self.config = config
        self.model = model
        hyper_parameter_str = 'bs_{}_lr_{}'.format(
            config.batch_size,
            config.learning_rate,
        )

        self.train_dir = './train_dir/%s-%s-%s-%s' % (
            config.prefix,
            config.dataset,
            hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )

        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        _, self.batch_train = create_input_ops(
            dataset, self.batch_size, is_training=True)
        _, self.batch_test = create_input_ops(
            dataset_test, self.batch_size, is_training=False)

        # --- optimizer ---
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.learning_rate = config.learning_rate

        self.check_op = tf.no_op()

        # --- checkpoint and monitoring ---
        all_var = tf.trainable_variables()
        tf.contrib.slim.model_analyzer.analyze_vars(all_var, print_info=True)

        self.optimizer = tf.train.AdamOptimizer(
            self.learning_rate
        ).minimize(self.model.loss, global_step=self.global_step,
                   var_list=all_var, name='optimizer_loss')

        self.train_summary_op = tf.summary.merge_all(key='train')
        self.test_summary_op = tf.summary.merge_all(key='test')

        self.saver = tf.train.Saver(max_to_keep=100)
        self.pretrain_saver = tf.train.Saver(var_list=all_var, max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.max_steps = self.config.max_steps
        self.ckpt_save_step = self.config.ckpt_save_step
        self.log_step = self.config.log_step
        self.test_sample_step = self.config.test_sample_step
        self.write_summary_step = self.config.write_summary_step

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=600,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.pretrain_saver.restore(self.session, self.ckpt_path, )
            log.info("Loaded the pretrain parameters from the provided checkpoint path")

    def train(self):
        log.infov("Training Starts!")
        print(self.batch_train)

        max_steps = self.max_steps
        ckpt_save_step = self.ckpt_save_step
        log_step = self.log_step
        test_sample_step = self.test_sample_step
        write_summary_step = self.write_summary_step

        for s in xrange(max_steps):
            # periodic inference
            if s % test_sample_step == 0:
                step, test_summary, loss, output, step_time = \
                    self.run_test(self.batch_test, step=s, is_train=False)
                self.log_step_message(step, loss, step_time, is_train=False)
                self.summary_writer.add_summary(test_summary, global_step=step)

            step, train_summary, loss, output, step_time = \
                self.run_single_step(self.batch_train, step=s, is_train=True)

            if s % log_step == 0:
                self.log_step_message(step, loss, step_time)

            if s % write_summary_step == 0:
                self.summary_writer.add_summary(train_summary, global_step=step)

            if s % ckpt_save_step == 0:
                log.infov("Saved checkpoint at %d", s)
                self.saver.save(
                    self.session, os.path.join(self.train_dir, 'model'),
                    global_step=step)

    def run_single_step(self, batch, step=None, opt_gan=False, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        fetch = [self.global_step, self.train_summary_op, self.model.output,
                 self.model.loss, self.check_op, self.optimizer]

        fetch_values = self.session.run(
            fetch,
            feed_dict=self.model.get_feed_dict(batch_chunk, step=step)
        )
        [step, summary, output, loss] = fetch_values[:4]

        _end_time = time.time()

        return step, summary, loss, output, (_end_time - _start_time)

    def run_test(self, batch, step, is_train=False):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        step, summary, loss, output = self.session.run(
            [self.global_step, self.test_summary_op,
             self.model.loss, self.model.output],
            feed_dict=self.model.get_feed_dict(batch_chunk, step=step, is_training=False)
        )

        _end_time = time.time()

        return step, summary, loss, output, (_end_time - _start_time)

    def log_step_message(self, step, loss, step_time, is_train=True):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Loss: {loss:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         loss=loss,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time
                         )
               )


def main():

    config, model, dataset_train, dataset_test = argparser(is_train=False)

    trainer = Trainer(config, model, dataset_train, dataset_test)

    log.warning("dataset: %s", config.dataset)
    trainer.train()

if __name__ == '__main__':
    main()

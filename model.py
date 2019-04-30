from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from util import log, train_test_summary
from synthesizer import Synthesizer
from ssim import tf_ssim as ssim


class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information
        self.config = config
        # dims
        self.batch_size = config.batch_size
        self.h, self.w, self.c, self.p_dim, self.num_input = config.data_info
        # architecture
        # dataset
        self.dataset_type = config.dataset_type

        # create placeholders for the input
        self.image = tf.placeholder(
            name='image', dtype=tf.float32,
            shape=[self.batch_size, self.h, self.w, self.c],
        )

        self.camera_pose = tf.placeholder(
            name='camera_pose', dtype=tf.float32,
            shape=[self.batch_size, self.p_dim, self.num_input],
        )

        self.step = tf.placeholder(
            name='step', dtype=tf.int32,
            shape=[],
        )

        self.is_train = tf.placeholder(
            name='is_train', dtype=tf.bool,
            shape=[],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=True):
        fd = {
            self.image: batch_chunk['image'],  # [B, h, w, c]
            self.camera_pose: batch_chunk['camera_pose'],  # [B, h, w, c]
            self.step: step
        }
        # if is_training is not None:
        fd[self.is_train] = is_training

        return fd

    def renormalize_image(self, image):
        # renormalize images from [-1, 1] to [0, 1]
        return (image+1)/2 if self.dataset_type == 'scene' else 1-(image+1)/2

    def build(self, is_train=True):

        c = int(self.c / self.num_input)
        rescale = 1 if self.dataset_type == 'scene' else 1.5

        # Input {{{
        # =========
        source_image, source_pose = \
            self.image[:, :, :, -c:], self.camera_pose[:, :, -1]
        target_image, target_pose = \
            self.image[:, :, :, :c], self.camera_pose[:, :, 0]
        # }}}

        # Graph {{{
        # =========
        synthesizer = Synthesizer(self.config, is_train=is_train)
        target_image_pred = synthesizer(source_image, source_pose, target_pose)
        # }}}

        # Build Losses {{{
        # ==============
        self.loss = tf.reduce_mean(tf.abs(target_image - target_image_pred)) * rescale
        # renormalize images from [-1, 1] to [0, 1] before computing ssim
        self.ssim = ssim(
            self.renormalize_image(target_image_pred),
            self.renormalize_image(target_image),
        )
        # }}}

        # Tensorboard Summary {{{
        # ==============
        # scalar
        train_test_summary("loss/loss", self.loss)
        train_test_summary("loss/ssim", self.ssim)

        # image
        dummy_grid = tf.ones_like(source_image)
        dummy_grid *= 1 if self.dataset_type == 'scene' else -1

        tb_image = tf.clip_by_value(tf.concat([
            source_image, target_image, target_image_pred
        ], axis=1), -1, 1)

        train_test_summary(
            "image",
            tb_image if self.dataset_type == 'scene' else 1-tb_image,
            summary_type='image')
        # }}}

        # Output {{{
        # ==============
        self.output = {}
        # }}}

        # Evaluation {{{
        # ==============
        self.eval_loss = {
            'l1_loss': self.loss,
            'ssim': self.ssim,
        }
        self.eval_img = {
            'display': self.renormalize_image(tb_image),
        }
        # }}}

        log.warn('Successfully loaded the model.')

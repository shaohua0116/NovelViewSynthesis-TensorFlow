import numpy as np
import tensorflow as tf

from ops import conv2d, fc, deconv2d
from util import log


class Synthesizer(object):

    def __init__(self, config, name='Synthesizer', is_train=True):
        self.name = name
        self._is_train = is_train
        self._reuse = False
        # architecture
        self._norm_type = config.norm_type
        self._skip_connection = config.skip_connection
        self._learn_pose_encoding = config.learn_pose_encoding

    def print_info(self, info, print_fn):
        if not self._reuse:
            print_fn(info)

    def __call__(self, source_image, source_pose, target_pose):
        with tf.variable_scope(self.name, reuse=self._reuse):
            self.print_info(self.name, log.warn)

            bs, h, w, c = source_image.get_shape().as_list()

            def ImageEncoder(source_image, pose, num_channel,
                             skip_connection=False, name='ImageEncoder'):
                with tf.variable_scope(name, reuse=self._reuse) as scope:
                    fm = source_image
                    self.print_info(scope.name, log.warn)
                    self.print_info(
                        'Input shape: {}'.format(fm.get_shape().as_list()), log.infov)

                    # encode image using convs
                    fms = []
                    for i, ch in enumerate(num_channel):
                        fm = conv2d(fm, ch, self._is_train, info=not self._reuse,
                                    norm=self._norm_type, name='conv{}'.format(i+1))
                        if skip_connection and i < len(num_channel)-1:
                            fms.append(fm)

                    # merge pose and feature maps
                    _, p_dim = pose.get_shape().as_list()
                    _, fh, fw, _ = fm.get_shape().as_list()
                    pose = tf.tile(tf.reshape(pose, [bs, 1, 1, p_dim]), [1, fh, fw, 1])
                    fm = tf.concat([fm, pose], axis=-1)
                    self.print_info(
                        'Output shape: {}'.format(fm.get_shape().as_list()), log.infov)
                return fm, fms

            def PoseEncoder(source_pose, target_pose, name='PoseEncoder'):
                with tf.variable_scope(name, reuse=self._reuse) as scope:
                    if self._learn_pose_encoding:
                        pose = tf.stack([source_pose, target_pose], axis=-1)
                        num_fc_channel = [64, 32]
                        self.print_info(scope.name, log.warn)
                        self.print_info(
                            'Input shape: {}'.format(pose.get_shape().as_list()), log.infov)
                        for i, ch in enumerate(num_fc_channel):
                            pose = fc(pose, ch, self._is_train, info=not self._reuse,
                                      norm=self._norm_type, name='token_emb_fc{}'.format(i+1))
                        pose = tf.reshape(pose, [bs, -1])
                        for i, ch in enumerate(num_fc_channel):
                            pose = fc(pose, ch, self._is_train, info=not self._reuse,
                                      norm=self._norm_type, name='pose_emb_fc{}'.format(i+1))
                    else:
                        pose = target_pose - source_pose
                    self.print_info(
                        'Output shape: {}'.format(pose.get_shape().as_list()), log.infov)
                return pose

            num_channel = [32, 64, 128, 256, 512]

            # encode target and source pose
            pose = PoseEncoder(source_pose, target_pose)
            # encode image
            # fm: final feature maps
            # fms: all intermediate feature maps (for skip connections)
            fm, fms = ImageEncoder(source_image, pose, num_channel,
                                   skip_connection=self._skip_connection)

            num_channel.reverse()
            fms.reverse()
            with tf.variable_scope('Decoder', reuse=self._reuse) as scope:
                self.print_info(scope.name, log.warn)
                self.print_info(
                    'Input shape: {}'.format(fm.get_shape().as_list()), log.infov)
                for i, ch in enumerate(num_channel):
                    fm = deconv2d(
                        fm, ch, self._is_train, info=not self._reuse,
                        norm=self._norm_type, name='deconv{}'.format(i+1))
                    if self._skip_connection and i < len(num_channel)-1:
                        fm = tf.concat([fm, fms[i]], axis=-1)
                        if not self._reuse:
                            log.info('add skip connection: {}'.format(
                                fm.get_shape().as_list()))

                # final output layer2
                fm = deconv2d(
                    fm, ch//4, self._is_train, info=not self._reuse,
                    s=1, k=3, norm='none', name='deconv{}'.format(i+2))

                target_image = deconv2d(
                    fm, c, self._is_train, info=not self._reuse,
                    s=1, k=3, activation_fn=tf.tanh,
                    norm='none', name='deconv{}'.format(i+3))

                # fix the size if needed
                target_image = tf.image.resize_bilinear(target_image, [h, w])
                self.print_info(
                    'Output shape: {}'.format(target_image.get_shape().as_list()), log.infov)

            assert source_image.get_shape() == target_image.get_shape()
            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return target_image

import argparse
import numpy as np

from model import Model


def argparser(is_train=True):

    def str2bool(v):
        return v.lower() == 'true'
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--dataset', type=str, default='car', choices=[
        'car', 'chair', 'kitti', 'synthia'])
    parser.add_argument('--num_input', type=int, default=1,
                        help='this is only used for multiview novel view synthesis')
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    # Log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--ckpt_save_step', type=int, default=5000)
    parser.add_argument('--test_sample_step', type=int, default=100)
    parser.add_argument('--write_summary_step', type=int, default=100)
    # Learning
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    # Architecture
    parser.add_argument('--norm_type', type=str, default='none',
                        choices=['batch', 'instance', 'none'])
    parser.add_argument('--skip_connection', type=str2bool, default=True)
    parser.add_argument('--learn_pose_encoding', type=str2bool, default=False)

    # Testing config {{{
    # ========
    # checkpoint
    parser.add_argument('--max_eval_steps', type=int, default=500)
    parser.add_argument('--data_id_list', type=str, default=None)
    # task type
    parser.add_argument('--loss', type=str2bool, default=True)
    parser.add_argument('--write_summary', type=str2bool, default=False)
    parser.add_argument('--plot_image', type=str2bool, default=False)
    # write summary file
    parser.add_argument('--quiet', type=str2bool, default=False)
    parser.add_argument('--summary_file', type=str, default='report.txt',
                        help='the path to the summary file')
    parser.add_argument('--output_dir', type=str,
                        help='the output directory of plotted images')
    # }}}

    config = parser.parse_args()

    if config.dataset in ['car', 'chair']:
        config.dataset_type = 'object'
        import datasets.object_loader as dataset
    elif config.dataset in ['kitti', 'synthia']:
        config.dataset_type = 'scene'
        import datasets.scene_loader as dataset

    dataset_train, dataset_test = \
        dataset.create_default_splits(config.num_input, config.dataset)
    image, pose = dataset_train.get_data(dataset_train.ids[0])

    config.data_info = np.concatenate([np.asarray(image.shape), np.asarray(pose.shape)])

    # --- create model ---
    model = Model(config, debug_information=config.debug, is_train=is_train)

    return config, model, dataset_train, dataset_test

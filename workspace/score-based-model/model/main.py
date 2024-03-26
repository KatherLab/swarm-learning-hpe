import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy
from runners import *
import sys
import os

def parse_args_and_config():

    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, default='abdomen2dCT.yml', required=True,
                        help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp',
                        help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, required=True, default='mri', help='A string for documentation purpose. '
                        'Will be the name of the log folder.')
    parser.add_argument('--comment', type=str, default='',
                        help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info',
                        help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--conditional', action='store_true',
                        help='Whether to train conditional model')
    parser.add_argument('--sample', action='store_true',
                        help='Whether to produce samples from the model')
    parser.add_argument('--resume_training', action='store_true',
                        help='Whether to resume training')
    parser.add_argument('-i', '--sampling_folder', type=str, default='image_samples', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true',
                        help="No interaction. Suitable for Slurm Job launcher")

    config_file = 'xrayVinDrConditional.yml'


    args = parser.parse_args()

    scratch_dir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    args.max_epochs = int(os.getenv('MAX_EPOCHS', 100))
    args.min_peers= int(os.getenv('MIN_PEERS', 2))
    args.max_peers = int(os.getenv('MAX_PEERS', 7))
    args.sync_frequency = int(os.getenv('SYNC_FREQUENCY', 1024))
    args.node_weightage = 100
    args.exp = config_file.split('.')[0]

    sys.argv = ['main.py', '--config', config_file, '--doc', 'test', '--conditional', '--exp', args.exp]

    args.log_path = os.path.join(
        args.exp, 'logs', args.doc, f'conditional' if args.conditional else 'unconditional')

    print("out dir path: ")
    # print absolute path
    print(os.path.abspath(args.log_path))
    print(os.path.abspath(args.exp))


    # parse config file
    with open(os.path.join('/tmp/test/model/configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

    if not args.sample:
        if not args.resume_training:
            os.makedirs(args.log_path, exist_ok=True)

            with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(
            os.path.join(args.log_path, 'stdout.txt'))
        formatter = logging.Formatter(
            '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)
    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            args.image_folder = os.path.join(
                args.log_path,  args.sampling_folder)
            os.makedirs(args.image_folder, exist_ok=True)

    # add device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Training conditional model: {} ".format(args.conditional))
    logging.info("Class list: {}".format(config.data.classes))
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 100)
    config_dict = copy.copy(vars(config))
    if not args.sample:
        del config_dict['tb_logger']
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 100)

    try:
        if args.conditional:
            runner = NCSNRunnerConditional(args, config)
        else:
            runner = NCSNRunner(args, config)

        if args.sample:
            runner.sample()
        else:
            runner.train()
    except:
        logging.error(traceback.format_exc())

    return 0



if __name__ == '__main__':
    sys.exit(main())

import logging
import os
import torch
import yaml
import argparse
from utils import create_log_folder
from experiment import EEGExperiment


def detr_args():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parametersy
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='maze')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def errp_args():

    parser = argparse.ArgumentParser('Set errp decoder', add_help=False)
    parser.add_argument('--pretrained_classifier', default='outputs/pretrained_classifier/pretrained_classifier_lidan.pth', type=str)
    return parser


def env_args():
    parser = argparse.ArgumentParser('Set environment', add_help=False)
    parser.add_argument('--baselog_dir', default='./logs/', type=str)
    parser.add_argument('--subject', default='', type=str)
    return parser


def game_args():

    parser = argparse.ArgumentParser(description='Snake AI replay client.', add_help=False)

    parser.add_argument('--mode', type=str, choices=['navigation', 'surveillance', 'calibration'], help='Choose mode to run',)
    parser.add_argument('--interface', type=str, choices=['cli', 'gui'], default='gui', help='Interface mode (command-line or GUI).',)
    parser.add_argument('--agent', required=False, type=str, choices=['human', 'dqn', 'random', 'val-itr', 'mixed', 'one-hot-dqn', 'tile-coding', 'reward-learning', 'a2c'], default='random', help='Player agent to use.',)
    parser.add_argument('--model', type=str, help='File containing a pre-trained agent model.',)
    parser.add_argument('--level', type=str, default='./robotaxi_integration/levels/8x8-blank.json', help='JSON file containing a level definition.',)
    parser.add_argument('--num-episodes', type=int, default=1, help='The number of episodes to run consecutively.',)
    parser.add_argument('--save_frames', action="store_true", default=False, help='save frames as jpg files in screenshots/ folder.')
    parser.add_argument('--stationary', action="store_true", default=False, help='determine whether the environment is stationary')
    parser.add_argument('--collaborating_agent', type=str, choices=['human', 'dqn', 'random', 'val-itr', 'mixed', 'one-hot-dqn', 'tile-coding'], help='Collaborator agent to use.',)
    parser.add_argument('--collaborator_model', type=str, help='File containing a pre-trained agent model.',)
    parser.add_argument('--participant', type=str, default='test', help='Participant ID.',)
    parser.add_argument('--test_run', action="store_true", default=False, help='determine whether the environment is stationary')
    parser.add_argument('--spatial_audio', action="store_true", help='Enable spatial sound',)
    return parser


def parse_args():

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[detr_args(), errp_args(), env_args(), game_args()])
    args = parser.parse_args()
    return args



def main():
    # Get arguments and configurations
    args = parse_args()
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create log directory and set up logging
    log_dir = create_log_folder(args.baselog_dir)
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        ],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    args.baselog_dir = log_dir
    logging.info(args)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Run EEG experiment
    experiment = EEGExperiment(args, config)
    experiment.run()


if __name__ == '__main__':
    main()

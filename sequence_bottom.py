from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from reid.datasets import get_sequence
from reid.dist_metric import DistanceMetric
from reid.loss.oim import OIMLoss
from reid.loss.triplet import TripletLoss
from reid.models import ResNet_btfu
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import seqtransforms
from reid.utils.data.seqpreprocessor import SeqPreprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint



def get_data(dataset_name, split_id, data_dir, batch_size, seq_len, seq_srd, workers,
             num_instances, combine_trainval=True):

    root = osp.join(data_dir, dataset_name)

    dataset = get_sequence(dataset_name, root, split_id=split_id,
                           seq_len= seq_len, seq_srd=seq_srd, num_val=1, download=True)

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    normalizer = seqtransforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])

    train_processor = SeqPreprocessor(train_set, dataset, transform=seqtransforms.Compose([seqtransforms.RandomSizedRectCrop(256, 128), seqtransforms.RandomHorizontalFlip(),
                                    seqtransforms.ToTensor(), normalizer]))

    val_processor = SeqPreprocessor(dataset.val, dataset, transform=seqtransforms.Compose([seqtransforms.RectScale(256, 128),
                                       seqtransforms.ToTensor(), normalizer]))

    test_processor = SeqPreprocessor(list(set(dataset.query) | set(dataset.gallery)), dataset, transform=seqtransforms.Compose([seqtransforms.RectScale(256, 128),
                                       seqtransforms.ToTensor(), normalizer]))
    train_loader = DataLoader(
        train_processor, batch_size=batch_size, num_workers=workers, shuffle=True,
        pin_memory=True)

    val_loader = DataLoader(
        val_processor, batch_size=batch_size, num_workers=workers, shuffle=False,
        pin_memory=True)

    test_loader = DataLoader(
        test_processor, batch_size=batch_size, num_workers=workers, shuffle= False,
        pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cudnn.benchmark = True

    # Redirect print to both console and log file
    # All the print infomration are stored in the logs_dir

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    if args.loss == 'triplet':
        assert args.num_instances > 1, 'TripletLoss requires num_instances > 1'
        assert args.batch_size % args.num_instances == 0, \
            'num_instances should divide batch_size'

    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir,
             args.batch_size, args.seq_len, args.seq_srd,
                 args.workers, args.num_instances,
                 combine_trainval=args.combine_trainval)

    # Create model
    # model =
    # TODO


    for i, input in enumerate(train_loader):
        a = 0








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ID Training ResNet Model")
    # DATA
    parser.add_argument('-d', '--dataset', type=str, default='ilidsvidsequence',
                        choices=['ilidsvidsequence'])
    parser.add_argument('-b', '--batch-size', type=int, default=256)

    parser.add_argument('-j', '--workers', type=int, default=4)

    parser.add_argument('--seq_len', type=int, default=12)

    parser.add_argument('--seq_srd', type=int, default=6)


    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--combine-trainval', action='store_true',
                        help="Use train and val sets together for training."
                             "Val set is still used for validation.")
    parser.add_argument('--num-instances', type=int, default=0,
                        help="If greater than zero, each minibatch will"
                             "consist of (batch_size // num_instances)"
                             "identities, and each identity will have"
                             "num_instances instances. Used in conjunction with"
                             "--loss triplet")
    parser.add_argument('--loss', type=str, default='xentropy',
                        choices=['xentropy', 'oim', 'triplet'])
    parser.add_argument('--seed', type=int, default=1)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
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
from reid.models import ResNetLSTM_btfu
from reid.trainers import SeqTrainer
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



    if num_instances >0:
        train_loader = DataLoader(
        train_processor, batch_size=batch_size, num_workers=workers,  sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True)
    else:
        train_loader = DataLoader(train_processor, batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True)


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
                 combine_trainval=True)

    # Create model
    if args.loss == 'xentropy':
        model = ResNetLSTM_btfu(args.depth, pretrained=True,  num_features=args.features, dropout=args.dropout)

    elif args.loss == 'oim':
        model = ResNetLSTM_btfu(args.depth, pretrained=True,  num_features=args.features,  norm=True, dropout=args.dropout)

    elif args.loss == 'triplet':
        model = ResNetLSTM_btfu(args.depth,  pretrained=True, num_features=args.features, dropout=args.dropout)

    else:
        raise ValueError("cannot recognize loss type:", args.loss)

    model = torch.nn.DataParallel(model).cuda()


    # Load from checkpoint
    # TODO is not necessary currently

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)


    # Evaluator
    evaluator = Evaluator(model)


    # Criterion
    if args.loss == 'xentropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'oim':
        criterion = OIMLoss(model.module.num_features, num_classes,
                            scalar=args.oim_scalar, momentum=args.oim_momentum)
    elif args.loss == 'triplet':
        criterion = TripletLoss(margin=args.triplet_margin)
    else:
        raise ValueError("Cannot recognize loss type:", args.loss)
    criterion.cuda()

    # Optimizer
    if args.optimizer == 'sgd':
        if args.loss == 'xentropy':
            base_param_ids = set(map(id, model.module.base.parameters()))
            new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
            param_groups = [
                {'params': model.module.base.parameters(), 'lr_mult': 0.1},
                {'params': new_params, 'lr_mult': 1.0}]
        else:
            param_groups = model.parameters()
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)

    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)

    else:
        raise ValueError("Cannot recognize optimizer type:", args.optimizer)

    # Trainer
    trainer = SeqTrainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        if args.optimizer == 'sgd':
            lr = args.lr * (0.1 ** (epoch // 40))
        elif args.optimizer == 'adam':
            lr = args.lr if epoch <= 100 else \
                args.lr * (0.001 ** (epoch - 100) / 50)
        else:
            raise ValueError("Cannot recognize optimizer type:", args.optimizer)
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Starting training
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)

        top1 = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, multi_shot=True)











if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ID Training ResNet Model")
    # DATA
    parser.add_argument('-d', '--dataset', type=str, default='ilidsvidsequence',
                        choices=['ilidsvidsequence'])
    parser.add_argument('-b', '--batch-size', type=int, default=256)

    parser.add_argument('-j', '--workers', type=int, default=4)

    parser.add_argument('--seq_len', type=int, default=6)

    parser.add_argument('--seq_srd', type=int, default=3)


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
    # model
    parser.add_argument('--depth', type=int, default=50,
                        choices=[18, 34, 50, 101, 152])
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)

    # loss
    parser.add_argument('--loss', type=str, default='xentropy',
                        choices=['xentropy', 'oim', 'triplet'])
    parser.add_argument('--oim-scalar', type=float, default=30)
    parser.add_argument('--oim-momentum', type=float, default=0.5)
    parser.add_argument('--triplet-margin', type=float, default=0.5)

    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)

    # training_configs
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)

    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))

    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
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
from reid.utils.data import transforms
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint



def get_sequence(dataset_name, split_id, data_dir, batch_size, seq_len, seq_srd, workers,
             num_instances, seqcombine_trainval=False):

    root = osp.join(data_dir, dataset_name)

    dataset = get_sequence(dataset_name, root, seq_len, seq_srd,
                           split_id=split_id, num_val=1, download=True)


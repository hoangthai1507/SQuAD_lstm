
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD


args = get_train_args()

train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
train_loader = data.DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                collate_fn=collate_fn)

print("data")
print(train_dataset)
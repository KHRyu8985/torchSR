
import sys
sys.path.append('/data/shkim/A100/SuperResolution/torchSR/torchsr/train')

from torchsr.train.helpers import report_model
from torchsr.train.options import args
from torchsr.train.trainer import Trainer
import time
from utils import logger

LOG = logger.LOG

import random
import os
import numpy as np
import torch

def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed = args.seed
seed_everything(seed)
#print('seed number is ', seed)
LOG.info(f'seed number is {seed}')
test = ['1','2','3','4','5','6','7','8','9','10']
random.shuffle(test)
#print(test)
LOG.info(test)

start = time.time()
trainer = Trainer()

if args.validation_only or args.images:
    if args.load_pretrained is None and args.load_checkpoint is None and not args.download_pretrained:
        raise ValueError("For validation, please use --load-pretrained CHECKPOINT or --download-pretrained")
    if args.images:
        if args.lre:
            trainer.run_lre_model()
        else:
            trainer.run_model()
    else:
        trainer.validation()
else:
    report_model(trainer.model)
    trainer.train()

#print("Time taken for train : ",time.time()-start)
LOG.info(f"Time taken for train : {time.time()-start}")



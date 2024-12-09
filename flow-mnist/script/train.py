import os
import sys

from datetime import datetime
from tqdm import tqdm as tqdm
import torch



os.environ['PYTHONPATH']=os.getcwd()
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
iprt_dirs=[]
iprt_dirs.append(os.path.join(current_dir, '..', 'model'))
iprt_dirs.append(os.path.join(current_dir, '..', 'data'))
for import_dir in iprt_dirs:
    if import_dir not in sys.path:
        sys.path.append(import_dir)

from model import *
from data import *

# enable hydra full error:
os.environ['HYDRA_FULL_ERROR'] = '1'



import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from helpers import visualize

@hydra.main(version_base=None,
            config_path=os.path.join(os.getcwd(),'cfg'),         # can be override with --config-dir=
            config_name='cfg01.yaml'                             # can be override with --config-name=
            )
def main(cfg:OmegaConf):
    OmegaConf.resolve(cfg)

    algo =hydra.utils.instantiate(cfg.algo)
    
    train_dataset = MNISTDataset(csv_file=os.path.join('data', 'mnist_train.csv'), use_top=cfg.dataset.use_first_train)
    test_dataset = MNISTDataset(os.path.join('data', 'mnist_test.csv'), use_top=cfg.dataset.use_first_eval)

    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=True)
    
    # visualize(train_loader)

    algo.run(train_loader, test_loader)

if __name__=="__main__":
    
    main()
    
    
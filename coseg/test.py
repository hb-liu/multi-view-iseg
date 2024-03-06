import yaml
import torch
import os, pickle
import torch.nn as nn
from easydict import EasyDict
from models.resunet import SEGNET
from dataset.dataset import get_validset
from core.function import inference
from utils.utils import create_logger, setup_seed

def main(config):
    setup_seed(config.misc.seed)

    model_src = SEGNET(inc=config.dataset.nmodal, outc=config.dataset.nclass+1, midc=config.model.midc, stages=config.model.stages)
    model_src = nn.DataParallel(model_src, config.misc.devices).cuda()
    model_src.load_state_dict(torch.load(config.inference.weights)['src'])

    model_syn = SEGNET(inc=config.dataset.nmodal, outc=config.dataset.nclass+1, midc=config.model.midc, stages=config.model.stages)
    model_syn = nn.DataParallel(model_syn, config.misc.devices).cuda()
    model_syn.load_state_dict(torch.load(config.inference.weights)['syn'])

    with open(os.path.join(config.dataset.src_dir, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)
    validset_src = get_validset(splits['test'], config.dataset.src_dir)
    validset_syn = get_validset(splits['test'], config.dataset.syn_dir)

    logger = create_logger('log', 'test.log')
    inference([model_src, model_syn], [validset_src, validset_syn], logger, config)

if __name__ == '__main__':
    with open('core/configs/finetune.yaml') as f:
        config = EasyDict(yaml.safe_load(f))
    main(config)
import yaml
import torch
import os, pickle
import torch.nn as nn
import torch.optim as optim
from easydict import EasyDict
from models.resunet import SEGNET
from core.scheduler import PolyScheduler
from dataset.dataset import get_validset
from core.function import train, inference
from dataset.dataloader import get_trainloader
from utils.utils import save_checkpoint, create_logger, setup_seed
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

def main(config):
    setup_seed(config.misc.seed)

    model_src = SEGNET(inc=config.dataset.nmodal, outc=config.dataset.nclass+1, midc=config.model.midc, stages=config.model.stages)
    model_src = nn.DataParallel(model_src, config.misc.devices).cuda()
    model_src.load_state_dict(torch.load(config.model.pretrain_weights)['syn'])

    model_syn = SEGNET(inc=config.dataset.nmodal, outc=config.dataset.nclass+1, midc=config.model.midc, stages=config.model.stages)
    model_syn = nn.DataParallel(model_syn, config.misc.devices).cuda()
    model_syn.load_state_dict(torch.load(config.model.pretrain_weights)['src'])

    optim_src = optim.SGD(model_src.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay, momentum=config.train.momentum, nesterov=True)
    sched_src = PolyScheduler(optim_src, t_total=config.train.nepoch)
    optim_syn = optim.SGD(model_syn.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay, momentum=config.train.momentum, nesterov=True)
    sched_syn = PolyScheduler(optim_syn, t_total=config.train.nepoch)
    
    criterion = nn.CrossEntropyLoss()

    with open(os.path.join(config.dataset.src_dir, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)
    trainloader = get_trainloader(splits['train'], config)
    train_generator = MultiThreadedAugmenter(
        data_loader=trainloader,
        transform=NumpyToTensor(keys=['data', 'label'], cast_to='float'),
        num_processes=config.misc.num_workers,
        pin_memory=True
    )
    validset_src = get_validset(splits['valid'], config.dataset.src_dir)
    validset_syn = get_validset(splits['valid'], config.dataset.syn_dir)

    is_best = False
    best_perf = 0.0
    logger = create_logger('log', 'finetune.log')
    for epoch in range(config.train.nepoch):
        train([model_src, model_syn], train_generator, [optim_src, optim_syn], criterion, logger, config, epoch)
        sched_src.step()
        sched_syn.step()
        
        perf = inference([model_src, model_syn], [validset_src, validset_syn], logger, config)
        
        if perf > best_perf:
            best_perf = perf
            is_best = True
        else:
            is_best = False
        save_checkpoint({
            'src': model_src.state_dict(),
            'syn': model_syn.state_dict()
        }, is_best, config.misc.model_dir, filename='checkpoint.pth')

if __name__ == '__main__':
    with open('core/configs/finetune.yaml') as f:
        config = EasyDict(yaml.safe_load(f))
    main(config)
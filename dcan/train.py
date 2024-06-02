import yaml, shutil
from utils.utils import *
import torch.optim as optim
from models.networks import *
from easydict import EasyDict
from core.function import train
from dataset.dataloader import get_trainloader
from dataset.augmenter import get_train_generator
from torch.optim.lr_scheduler import ExponentialLR

def main(config):
    setup_seed(config.misc.seed)

    # generator
    enc_s = Encoder(inc=config.dataset.nmodal, zdim=config.model.latent_dim).cuda()
    enc_s = nn.DataParallel(enc_s, device_ids=config.misc.devices).cuda()
    enc_c = Encoder(inc=config.dataset.nmodal, zdim=config.model.latent_dim).cuda()
    enc_c = nn.DataParallel(enc_c, device_ids=config.misc.devices).cuda()
    dec = Decoder(outc=config.dataset.nmodal, zdim=config.model.latent_dim).cuda()
    dec = nn.DataParallel(dec, device_ids=config.misc.devices).cuda()
    params = list(enc_s.parameters()) + list(enc_c.parameters()) + list(dec.parameters())
    optim_gen = optim.Adam(params, lr=config.train.lr_gen)
    sched_gen = ExponentialLR(optim_gen, 0.985)
    
    # discriminator
    dis_src = Discriminator(inc=config.dataset.nmodal).cuda()
    dis_src = nn.DataParallel(dis_src, device_ids=config.misc.devices).cuda()
    dis_dst = Discriminator(inc=config.dataset.nmodal).cuda()
    dis_dst = nn.DataParallel(dis_dst, device_ids=config.misc.devices).cuda()
    params = list(dis_src.parameters()) + list(dis_dst.parameters())
    optim_dis = optim.Adam(params, lr=config.train.lr_dis)
    sched_dis = ExponentialLR(optim_dis, 0.985)

    # dataset
    src_loader = get_trainloader(config.dataset.src_dir, config)
    dst_loader = get_trainloader(config.dataset.dst_dir, config)
    src_loader = get_train_generator(src_loader, config)
    dst_loader = get_train_generator(dst_loader, config)

    logger = create_logger('log', 'train.log')
    for epoch in range(config.train.epoch):
        train(enc_s, enc_c, dec, dis_src, dis_dst, src_loader, dst_loader, optim_gen, optim_dis, logger, config, epoch)
        sched_gen.step()
        sched_dis.step()
        save_checkpoint(
            {'enc_s': enc_s.state_dict(), 'enc_c': enc_c.state_dict(), 'dec': dec.state_dict()},
            is_best=False, output_dir=config.misc.model_dir, filename='checkpoint.pth'
        )
        # a snapshot
        shutil.copy('tmp/train.png', f'saves/{epoch}.png')

if __name__ == '__main__':
    with open('core/config.yaml') as f:
        config = EasyDict(yaml.safe_load(f))
    main(config)
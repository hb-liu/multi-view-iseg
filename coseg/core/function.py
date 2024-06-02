import torch
import numpy as np
import torchio as tio
import SimpleITK as sitk
from statistics import mean
from core.loss import ConsisLoss
from medpy.metric.binary import dc
from utils.utils import AverageMeter
from torch.utils.data import DataLoader
from torchvision.utils import save_image

def train(model, train_generator, optimizer, criterion, logger, config, epoch):
    model_src, model_syn = model
    optim_src, optim_syn = optimizer
    model_src.train()
    model_syn.train()
    losses = AverageMeter()
    cons_loss = ConsisLoss()
    scaler = torch.cuda.amp.GradScaler()
    num_iter = config.train.niter
    for i in range(num_iter):
        data_dict = next(train_generator)
        image = data_dict['data'].cuda()
        label = data_dict['label'].cuda()

        src, syn = torch.chunk(image, 2, 1)

        with torch.cuda.amp.autocast():
            out_src = model_src(src)
            out_syn = model_syn(syn)
            lsrc = criterion(out_src, label.squeeze(1).long())
            lsyn = criterion(out_syn, label.squeeze(1).long())
            lcons = cons_loss(out_src, out_syn, label)
            loss = lsrc + lsyn + 10. * lcons
        
        batch_size = image.shape[0]
        losses.update(loss.item(), batch_size)

        optim_src.zero_grad()
        optim_syn.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optim_src)
        torch.nn.utils.clip_grad_norm_(model_src.parameters(), 12)
        scaler.unscale_(optim_syn)
        torch.nn.utils.clip_grad_norm_(model_syn.parameters(), 12)
        scaler.step(optim_src)
        scaler.step(optim_syn)
        scaler.update()
        # log
        if i % config.misc.print_freq == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                    epoch, i, num_iter,
                    loss = losses,
                )
            logger.info(msg)
    # a snapshot
    src = torch.cat(torch.split(src, 1, 1))[:, :, src.shape[2]//2]
    syn = torch.cat(torch.split(syn, 1, 1))[:, :, syn.shape[2]//2]
    label = torch.cat(torch.split(label, 1, 1))[:, :, label.shape[2]//2]
    out = (out_src + out_syn) / 2
    out = torch.argmax(torch.softmax(out, 1), dim=1, keepdim=True)
    out = torch.cat(torch.split(out, 1, 1))[:, :, out.shape[2]//2]
    save_image(torch.cat([src, syn, label, out]).cpu(), 
                'tmp/train.png', nrow=batch_size, scale_each=True, normalize=True)

@ torch.no_grad()
def inference(model, dataset, logger, config):
    model_src, model_syn = model
    validset_src, validset_syn = dataset
    model_src.eval()
    model_syn.eval()
    num_classes = config.dataset.nclass
    scores = {i+1: [] for i in range(num_classes)}
    for src, syn in zip(validset_src, validset_syn):
        patch_size = config.model.patch_size
        patch_overlap = config.inference.patch_overlap
        # data size may be smaller than patch size
        target_shape = np.max([patch_size, src['data'].shape[1:]], 0)
        transform = tio.CropOrPad(target_shape)
        src, syn = transform(src), transform(syn)
        # patch sampler
        sampler_src = tio.inference.GridSampler(src, patch_size, patch_overlap)
        loader_src = DataLoader(sampler_src, config.inference.batch_size)
        sampler_syn = tio.inference.GridSampler(syn, patch_size, patch_overlap)
        loader_syn = DataLoader(sampler_syn, config.inference.batch_size)
        aggregator = tio.inference.GridAggregator(sampler_src, 'average')
        # run inference
        for data_src, data_syn in zip(loader_src, loader_syn):
            patch_src = data_src['data'][tio.DATA].cuda()
            patch_syn = data_syn['data'][tio.DATA].cuda()
            with torch.cuda.amp.autocast():
                out_src = model_src(patch_src)
                out_syn = model_syn(patch_syn)
                out = (out_src + out_syn) / 2
                out = torch.softmax(out, 1)
            loc = data_src[tio.LOCATION]
            aggregator.add_batch(out, loc)
        # form final prediction
        pred = aggregator.get_output_tensor()
        pred = torch.argmax(pred, dim=0).cpu().numpy()
        label = src['label'][tio.DATA][0].numpy()
        # compensate for the loss of csf in synthesized image
        mask = (pred == 0) * (label == 1)
        pred[mask] = label[mask]
        # save segmentation results
        name = src['name']
        sitk.WriteImage(sitk.GetImageFromArray(pred), f'preds/{name}.nii.gz')
        # quantitative analysis
        for i in range(num_classes):
            scores[i+1].append(dc(pred==i+1, label==i+1))
    # log
    logger.info('------------ ----------- ------------')
    for i in range(num_classes):
        logger.info(f'class {i+1} dice mean: {mean(scores[i+1])}')
    logger.info('------------ ----------- ------------')

    perf = mean([mean(scores[i+1]) for i in range(num_classes)])
    return perf
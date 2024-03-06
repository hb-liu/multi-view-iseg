import torch
import random
import torch.nn as nn
import torchvision.utils as vutils
from utils.utils import AverageMeter

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

def lossD_basic(criterionGAN, netD, real, fake):
    # real
    pred_real = netD(real)
    loss_D_real = criterionGAN(pred_real, True)
    # fake
    pred_fake = netD(fake.detach())
    loss_D_fake = criterionGAN(pred_fake, False)
    # loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    return loss_D

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def transfer(model, inA, inB):
    enc_S, enc_C, dec = model
    # style code
    S_A = enc_S(inA, with_embed=False)
    S_B = enc_S(inB, with_embed=False)
    # content code
    C_A, E_A = enc_C(inA, with_embed=True)
    C_B, E_B = enc_C(inB, with_embed=True)
    # synthesize
    synB = dec(C_A + S_B, E_A)
    synA = dec(C_B + S_A, E_B)
    return synB, synA

def train(enc_S, enc_C, dec, netD_A, netD_B, loader_A, loader_B, optimG, optimD, logger, config, epoch):
    # generator
    enc_S.train()
    enc_C.train()
    dec.train()
    # discriminator
    netD_A.train()
    netD_B.train()

    # criterion
    criterionGAN = GANLoss().cuda()
    criterionCycle = nn.L1Loss()

    # log
    lossesG = AverageMeter()
    lossesD = AverageMeter()

    # image pool
    synA_pool = ImagePool(50)
    synB_pool = ImagePool(50)

    # weights
    lambdaA = 10
    lambdaB = 10
    margin = 0.5

    niter = config.train.niter
    scaler = torch.cuda.amp.GradScaler()
    for idx in range(niter):
        inA = next(loader_A)['data'].cuda()
        inB = next(loader_B)['data'].cuda()
        batch_size = inA.size(0)
        # forward
        with torch.cuda.amp.autocast():
            # style code
            S_A = enc_S(inA)
            S_B = enc_S(inB)
            # content code
            C_A, E_A = enc_C(inA, True)
            C_B, E_B = enc_C(inB, True)
            # synthesize
            synB = dec(C_A + S_B, E_A)
            synA = dec(C_B + S_A, E_B)
            # style code
            S_synA = enc_S(synA)
            S_synB = enc_S(synB)
            # content code
            C_synA, E_synA = enc_C(synA, True)
            C_synB, E_synB = enc_C(synB, True)
            # synthesize
            recA = dec(C_synB + S_synA, E_synB)
            recB = dec(C_synA + S_synB, E_synA)
            # identity
            _, ideA = transfer([enc_S, enc_C, dec], inA, synB)
            _, ideB = transfer([enc_S, enc_C, dec], inB, synA)
        # generator training
        set_requires_grad([netD_A, netD_B], False)
        optimG.zero_grad()
        with torch.cuda.amp.autocast():
            lossG_A = criterionGAN(netD_A(synA), True)
            lossG_B = criterionGAN(netD_B(synB), True)
            # self reconstruction
            lossR_A = criterionCycle(dec(C_A + S_A, E_A), inA)
            lossR_B = criterionCycle(dec(C_B + S_B, E_B), inB)
            lossR_synA = criterionCycle(dec(C_synA + S_synA, E_synA), synA)
            lossR_synB = criterionCycle(dec(C_synB + S_synB, E_synB), synB)
            
            # style loss
            lossS = 0.0
            for i in range(batch_size):
                for j in range(batch_size):
                    lossS = lossS + max(0, margin - 2.0*criterionCycle(S_A[i], S_B[i]) + criterionCycle(S_A[i], S_A[j]) + criterionCycle(S_B[i], S_B[j]))
            lossS = lossS / (i*j)
            # content loss
            lossC_A = criterionCycle(C_synB, C_A)
            lossC_B = criterionCycle(C_synA, C_B)

            # cycle consistency
            lossCycle_A = criterionCycle(recA, inA)
            lossCycle_B = criterionCycle(recB, inB)

            # identity loss
            lossI_A = criterionCycle(ideA, inA)
            lossI_B = criterionCycle(ideB, inB)
            
            # generator loss
            lossG = lossG_A + lossG_B + lambdaA * (lossCycle_A + lossR_A + lossR_synA + lossI_A) + lambdaB * (lossCycle_B + lossR_B + lossR_synB + lossI_B) + 10*(lossC_A + lossC_B) + lossS
        # back-propogation
        scaler.scale(lossG).backward()
        scaler.unscale_(optimG)
        torch.nn.utils.clip_grad_norm_(list(enc_S.parameters()) + list(enc_C.parameters()) + list(dec.parameters()), 1)
        scaler.step(optimG)
        scaler.update()
        # log
        lossesG.update(lossG.item(), batch_size)
        # discriminator training
        set_requires_grad([netD_A, netD_B], True)
        optimD.zero_grad()
        with torch.cuda.amp.autocast():
            _synA = synA_pool.query(synA)
            lossD_A = lossD_basic(criterionGAN, netD_A, inA, _synA)
            _synB = synB_pool.query(synB)
            lossD_B = lossD_basic(criterionGAN, netD_B, inB, _synB)
        # back-propagation
        scaler.scale(lossD_A).backward()
        scaler.scale(lossD_B).backward()
        scaler.unscale_(optimD)
        torch.nn.utils.clip_grad_norm_(list(netD_A.parameters()) + list(netD_B.parameters()), 1)
        scaler.step(optimD)
        scaler.update()
        # Log
        lossesD.update(lossD_A.item() + lossD_B.item(), batch_size)

        if idx % config.misc.print_freq == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'lossG {loss1.val:.3f} ({loss1.avg:.3f})\t' \
                'lossD {loss2.val:.3f} ({loss2.avg:.3f})\t'.format(
                    epoch, idx, niter,
                    loss1 = lossesG, loss2 = lossesD
                )
            logger.info(msg)
            vutils.save_image(
                torch.cat([
                    inA[:1, 0:1, :, :, 32], inB[:1, 0:1, :, :, 32], synA[:1, 0:1, :, :, 32], synB[:1, 0:1, :, :, 32], recA[:1, 0:1, :, :, 32], recB[:1, 0:1, :, :, 32],
                    inA[:1, 1:2, :, :, 32], inB[:1, 1:2, :, :, 32], synA[:1, 1:2, :, :, 32], synB[:1, 1:2, :, :, 32], recA[:1, 1:2, :, :, 32], recB[:1, 1:2, :, :, 32]
                ]),
                'tmp/train.png', nrow=batch_size, normalize=True, scale_each=True
            )
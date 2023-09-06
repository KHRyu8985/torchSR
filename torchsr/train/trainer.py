import copy
import os

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import piq
from tqdm import tqdm
from PIL import Image

import sys

from torchsr.train.enums import *

from torchsr.train.helpers import AverageMeter
from torchsr.train.helpers import to_image
from torchsr.train.helpers import to_tensor
from torchsr.train.helpers import to_luminance
from torchsr.train.helpers import to_YCbCr
from torchsr.train.helpers import get_model, get_vnet, get_vnet2
from torchsr.train.helpers import get_optimizer
from torchsr.train.helpers import get_scheduler
from torchsr.train.helpers import get_loss
from torchsr.train.helpers import get_device
from torchsr.train.helpers import get_dtype
from torchsr.train.helpers import get_datasets
import numpy as np

from .options import args

import logging
from utils import logger
import lpips
import pyiqa

LOG = logger.LOG

from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from torch.autograd import Variable

### SHKIM higher
import higher


### SHKIM albumentations
import albumentations

### SHKIM GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class Trainer:
    def __init__(self):
        self.epoch = 0
        self.best_psnr = None
        self.best_ssim = None
        self.best_fsim = None
        self.best_lpips = None
        self.best_fid = None
        self.best_niqe = None
        self.best_musiq = None
        self.best_nrqm = None
        self.best_loss = None
        self.best_epoch = None
        
        self.setup_device()
        self.setup_datasets()
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_loss()
        if args.load_checkpoint is not None :
            if os.path.isfile(args.load_checkpoint):
                LOG.info(f'load checkpoint from {args.load_checkpoint}')
                self.load_checkpoint()

        self.setup_tensorboard()

        # self.psnr = piq.psnr
        # self.ssim = piq.ssim
        # self.fsim = piq.fsim
        self.lpips_model = lpips.LPIPS().to(self.device)
        # self.psnr = pyiqa.create_metric('psnr').to(self.device)
        self.psnr = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(self.device)
        self.ssim = pyiqa.create_metric('ssim', device=self.device)
        if args.metric > 0:
            self.fsim = pyiqa.create_metric('fsim', device=self.device)
            self.lpips = pyiqa.create_metric('lpips', device=self.device)
            if args.metric > 1:
                self.fid = pyiqa.create_metric('fid', device=self.device)
                self.niqe = pyiqa.create_metric('niqe', device=self.device)
                self.musiq = pyiqa.create_metric('musiq-spaq', device=self.device)
                self.nrqm = pyiqa.create_metric('nrqm', device=self.device)

    def setup_device(self):
        self.device = get_device()
        self.dtype = get_dtype()

    def setup_datasets(self):
        self.loader_train, self.loader_val, self.loader_traval = get_datasets()

    def setup_model(self):
        self.model = get_model().to(self.device).to(self.dtype)
        self.vnet = get_vnet().to(self.device).to(self.dtype)
        self.vnet2 = get_vnet2().to(self.device).to(self.dtype)

    def setup_optimizer(self):
        self.optimizer = get_optimizer(self.model)
        self.vnet_opt = torch.optim.SGD(self.vnet.params(), 1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
        self.vnet_opt2 = torch.optim.SGD(self.vnet2.params(), 1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)

    def setup_scheduler(self):
        self.scheduler = get_scheduler(self.optimizer)
        self.vnet_scheduler = get_scheduler(self.vnet_opt)

    def setup_loss(self):
        self.loss_fn = get_loss()

    def setup_tensorboard(self):
        self.writer = None
        if not args.validation_only:
            try:
                # Only if tensorboard is present
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(args.log_dir, purge_step=self.epoch)
            except ImportError:
                if args.log_dir is not None:
                    raise ImportError("tensorboard is required to use --log-dir")

    def train_iter(self):
        with torch.enable_grad():
            self.model.train()
            t = tqdm(range(len(self.loader_train) * args.dataset_repeat))
            t.set_description(f"Epoch {self.epoch} train ")
            loss_avg = AverageMeter(0.05)
            l1_avg = AverageMeter(0.05)
            l2_avg = AverageMeter(0.05)
#            for name, param in self.model.named_parameters():
#                if param.requires_grad is False:
#                    print(name, param.requires_grad)
#                    param.requires_grad = True

            for i in range(args.dataset_repeat):
                for hr, lr in self.loader_train:
                    self.model.train()
                    hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device)
                    lr = to_var(lr, requires_grad=False)
                    hr = to_var(hr, requires_grad=False)

#                    self.optimizer.zero_grad()
                    sr = self.model(lr)
                    sr = self.process_for_eval(sr)
                    hr = self.process_for_eval(hr)
                    loss = self.loss_fn(sr, hr)
                    self.optimizer.zero_grad()
                    loss.backward()
                    if args.gradient_clipping is not None:
#                        nn.utils.clip_grad_norm_(self.model.params(), args.gradient_clipping)
                        nn.utils.clip_grad_norm_(self.model.parameters(), args.gradient_clipping)
                    self.optimizer.step()
                    l1_loss = nn.functional.l1_loss(sr, hr).item()
                    l2_loss = torch.sqrt(nn.functional.mse_loss(sr, hr)).item()
                    l1_avg.update(l1_loss)
                    l2_avg.update(l2_loss)
                    args_dic = {
                        'L1': f'{l1_avg.get():.4f}',
                        'L2': f'{l2_avg.get():.4f}'
                    }
                    if args.loss not in [LossType.L1, LossType.L2]:
                        loss_avg.update(loss.item())
                        args_dic[args.loss.name] = f'{loss_avg.get():.4f}'
                    t.update()
                    t.set_postfix(**args_dic)
            
        LOG.info(str(t))
    

    def test_iter_lre_higher(self, lr_img, hr_img):
        dl = 1
        # deformer = albumentations.OpticalDistortion(distort_limit=(dl,dl), shift_limit=0, p=1)
        for nv2 in range(3):
            # nv = 0.01 * nv2
            nv = args.noise_value * 0.01
            # nv = args.noise_value
            shifter_array = []
                # translater = albumentations.Affine(translate_percent={"x" : nvx, "y": nvy})
            
            # shifter_array.append(albumentations.Affine(translate_percent={"x" : nv, "y": nv}, p=1))                
            # shifter_array.append(albumentations.Affine(translate_percent={"x" : 0, "y": nv}, p=1))
            # shifter_array.append(albumentations.Affine(translate_percent={"x" : nv, "y": 0}, p=1))
            # shifter_array.append(albumentations.Affine(translate_px={"x" : 0, "y": args.noise_value}, p=1))
            shifter_array.append(albumentations.Affine(translate_px={"x" : args.noise_value, "y": 0}, p=1))

            imagename=args.imagename
            PSR = False
            for k, shifter in enumerate(shifter_array):
                # if nv == 0:
                #     if k != 0:
                #         continue
                hr_ps = 96
                lr_ps = hr_ps//2
                with torch.enable_grad():
        #        with torch.no_grad():
                    self.model.train()
                    hr = np.array(hr_img)
                    lr = np.array(lr_img)

                    ### SHKIM defomer lr image
                    # noise_lr = deformer(image=lr)
                    # lr = noise_lr['image']
                    ### SHKIM shifter lr image
                    if nv == 0:
                        lr = lr
                    else:
                        noise_lr = shifter(image=lr)
                        lr = noise_lr['image']

                    hr_h, hr_w = hr.shape[0], hr.shape[1]
                    lr_h, lr_w = lr.shape[0], lr.shape[1]
                    resize_index =False
                    if hr_h == lr_h * args.scale:
                        resize_index =True
                    if hr_w == lr_w * args.scale:
                        resize_index =True
                    if resize_index == True:
                        resizer = albumentations.Resize(height=lr_h*args.scale, width=lr_w*args.scale)
                        hr = resizer(image=hr)['image']
                    tsr_hr = to_tensor(hr)
                    tsr_lr = to_tensor(lr)
                    np_tsr_hr = np.array(tsr_hr)
                    np_tsr_lr = np.array(tsr_lr)

                    hr_patches = [] 
                    lr_patches = []
                    sr_patches = []
                    grayscale_cams = []

                    ### patch superresolution
                    if PSR :
                        for j in range((hr_h-(hr_h-(hr_h//hr_ps)*hr_ps)) // hr_ps):
                            for i in range((hr_w-(hr_w-(hr_w//hr_ps)*hr_ps)) // hr_ps):
                            # for i in range((hr_w-hr_ps//2) // hr_ps):
                                hr_patch = np_tsr_hr[:, :, (hr_h-(hr_h//hr_ps)*hr_ps)//2 + j*hr_ps:(hr_h-(hr_h//hr_ps)*hr_ps)//2+(j+1)*hr_ps, (hr_h-(hr_h//hr_ps)*hr_ps)//2 + i*hr_ps:(hr_h-(hr_h//hr_ps)*hr_ps)//2+(i+1)*hr_ps]
                                lr_patch = np_tsr_lr[:, :, (hr_h-(hr_h//hr_ps)*hr_ps)//4 + j*lr_ps:(hr_h-(hr_h//hr_ps)*hr_ps)//4+(j+1)*lr_ps, (hr_w-(hr_w//hr_ps)*hr_ps)//4 + i*lr_ps:(hr_w-(hr_w//hr_ps)*hr_ps)//4+(i+1)*lr_ps]
                                # hr_patch = np_tsr_hr[:, :, hr_ps//2 + j*hr_ps:hr_ps//2+(j+1)*hr_ps, hr_ps//2 + i*hr_ps:hr_ps//2+(i+1)*hr_ps]
                                # lr_patch = np_tsr_lr[:, :, lr_ps//2 + j*lr_ps:lr_ps//2+(j+1)*lr_ps, lr_ps//2 + i*lr_ps:lr_ps//2+(i+1)*lr_ps]
            #                    hr_patch = hr[48 + j*96:48+(j+1)*96, 48 + i*96:48+(i+1)*96, :]
            #                    lr_patch = lr[24 + j*48:24+(j+1)*48, 24 + i*48:24+(i+1)*48, :]
                                hr_patches.append(hr_patch)
                                lr_patches.append(lr_patch)
                        ### patch superresolution
                    else:
                        ### image superresolution
                        hr_patches.append(np_tsr_hr)
                        lr_patches.append(np_tsr_lr)
                        ### image superresolution

                    t = tqdm(range(len(hr_patches)))
                    t.set_description(f"Epoch {self.epoch} test ")
                    loss_avg = AverageMeter(0.05)
                    l1_avg = AverageMeter(0.05)
                    l2_avg = AverageMeter(0.05)
                    for i, hr_patch in enumerate(hr_patches) :
        #            for hr, lr in self.loader_train:
                        lr_patch = lr_patches[i]
                        #lr_patch = to_tensor(lr_patch).to(self.device)
                        #hr_patch = to_tensor(hr_patch).to(self.device)
                        lr_patch = torch.Tensor(lr_patch).to(self.device)
                        hr_patch = torch.Tensor(hr_patch).to(self.device)

                        hr_patch, lr_patch = hr_patch.to(self.dtype).to(self.device), lr_patch.to(self.dtype).to(self.device)
                        hr_patch, lr_patch = hr_patch.to(device=self.device, non_blocking=True),\
                                lr_patch.to(device=self.device, non_blocking=True)
                        self.optimizer.zero_grad()
                        with higher.innerloop_ctx(self.model, self.optimizer) as (meta_model, meta_opt):
                            # 1. Update meta model on training data
                            meta_train_outputs = meta_model(lr_patch)
                            meta_train_outputs = self.process_for_eval(meta_train_outputs)
                            hr_patch = self.process_for_eval(hr_patch)
            
                            self.loss_fn.reduction = 'none'
                            meta_train_loss = self.loss_fn(meta_train_outputs, hr_patch)
                            if args.batch_reweight:
                                meta_train_loss = torch.sum(meta_train_loss, dim=(1,2,3))
                                eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=self.device)
                            else:
                                eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=self.device)
                            meta_train_loss = torch.sum(eps * meta_train_loss)
                            meta_opt.step(meta_train_loss)
            
                            # 2. Compute grads of eps on meta validation data
                            meta_labels, meta_inputs = np.array(hr_img), np.array(lr_img)
        
                            if resize_index == True:
                                meta_labels = resizer(image=meta_labels)['image']
                                meta_labels = to_tensor(meta_labels)
                                meta_inputs = to_tensor(meta_inputs)

                            # meta_labels, meta_inputs =  next(self.loader_traval)
                            meta_inputs, meta_labels =  meta_inputs.to(self.dtype).to(self.device),\
                                                        meta_labels.to(self.dtype).to(self.device)
                            meta_inputs, meta_labels = meta_inputs.to(device=self.device, non_blocking=True),\
                                                    meta_labels.to(device=self.device, non_blocking=True)
                            meta_val_outputs = meta_model(meta_inputs)
                            meta_labels = self.process_for_eval(meta_labels)
                            meta_val_outputs = self.process_for_eval(meta_val_outputs)
            
                            self.loss_fn.reduction = 'mean'
                            meta_val_loss = self.loss_fn(meta_val_outputs, meta_labels)
                            eps_grads = torch.autograd.grad(meta_val_loss, eps)[0].detach()
        
                        # 3. Compute weights for current training batch
                        w_tilde = torch.clamp(-eps_grads, min=0, max=300)
                        l1_norm = torch.sum(w_tilde)
                        if l1_norm != 0:
                            w = w_tilde / l1_norm
                        else:
                            w = w_tilde
                                           
                        ### SHKIM GradCAM
        #                print(args.batch_reweight)
                        if args.batch_reweight:
                            grayscale_cam = w_tilde.repeat(hr_ps,hr_ps)
                        else:
                            grayscale_cam = torch.sum(w_tilde, 1)
                            grayscale_cam = grayscale_cam[0,:]

                        grayscale_cams.append(grayscale_cam.cpu().detach().numpy())
                        # grayscale_cam = grayscale_cam / torch.max(grayscale_cam)
                        # cam_img = show_cam_on_image(hr_patch.cpu().detach().numpy().squeeze(0).transpose(1,2,0), grayscale_cam.cpu().detach().numpy(), use_rgb=True, image_weight=0.8)
                        # cam_img = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
            
                        # cv2.imwrite(f'./results/test-cam-{i}-shift.jpg', cam_img)
                        # breakpoint()

                        # 4. Train model on weighted batch
                        sr_patch = self.model(lr_patch)

                        sr_patch = self.process_for_eval(sr_patch)
                        self.loss_fn.reduction = 'none'
                        minibatch_loss = self.loss_fn(sr_patch, hr_patch)
                        if args.batch_reweight:
                            minibatch_loss = torch.sum(minibatch_loss, dim=(1,2,3))
                        loss = torch.sum(minibatch_loss * w)
                        loss.backward()
                        if args.gradient_clipping is not None:
                            nn.utils.clip_grad_norm_(self.model.parameters(), args.gradient_clipping)
                        self.optimizer.step()
            
                        # keep track of epoch loss/accuracy
                        l1_loss = nn.functional.l1_loss(sr_patch, hr_patch).item()
                        l2_loss = torch.sqrt(nn.functional.mse_loss(sr_patch, hr_patch)).item()
                        l1_avg.update(l1_loss)
                        l2_avg.update(l2_loss)
                        args_dic = {
                            'L1': f'{l1_avg.get():.4f}',
                            'L2': f'{l2_avg.get():.4f}'
                        }
                        if args.loss not in [LossType.L1, LossType.L2]:
                            loss_avg.update(loss.item())
                            args_dic[args.loss.name] = f'{loss_avg.get():.4f}'
                        sr_patches.append(sr_patch.cpu().detach().numpy())
                        t.update()
                        t.set_postfix(**args_dic)

                    sr = np.zeros(np_tsr_hr.shape)
                    grayscale = np.zeros((np_tsr_hr.shape[2], np_tsr_hr.shape[3]))
                    if PSR :
                        ### patch superresolution
                        sr[:, :, 0:(hr_h-(hr_h//hr_ps)*hr_ps)//2, :] = np_tsr_hr[:, :, 0:(hr_h-(hr_h//hr_ps)*hr_ps)//2, :]
                        sr[:, :, :, 0:(hr_w-(hr_w//hr_ps)*hr_ps)//2] = np_tsr_hr[:, :, :, 0:(hr_w-(hr_w//hr_ps)*hr_ps)//2]
                        sr[:, :, hr_h-(hr_h-(hr_h//hr_ps)*hr_ps)//2:hr_h, :] = np_tsr_hr[:, :, hr_h-(hr_h-(hr_h//hr_ps)*hr_ps)//2:hr_h, :]
                        sr[:, :, :, hr_w-(hr_w-(hr_w//hr_ps)*hr_ps)//2:hr_w] = np_tsr_hr[:, :, :, hr_w-(hr_w-(hr_w//hr_ps)*hr_ps)//2:hr_w]

                        for j in range((hr_h-(hr_h-(hr_h//hr_ps)*hr_ps)) // hr_ps):
                            for i in range((hr_w-(hr_w-(hr_w//hr_ps)*hr_ps)) // hr_ps):
                                sr[0, :, (hr_h-(hr_h//hr_ps)*hr_ps)//2 + j*hr_ps:(hr_h-(hr_h//hr_ps)*hr_ps)//2+(j+1)*hr_ps, (hr_w-(hr_w//hr_ps)*hr_ps)//2 + i*hr_ps:(hr_w-(hr_w//hr_ps)*hr_ps)//2+(i+1)*hr_ps] = sr_patches[j*((hr_w-(hr_w-(hr_w//hr_ps)*hr_ps))//hr_ps) + i]
                                grayscale[(hr_h-(hr_h//hr_ps)*hr_ps)//2 + j*hr_ps:(hr_h-(hr_h//hr_ps)*hr_ps)//2+(j+1)*hr_ps, (hr_w-(hr_w//hr_ps)*hr_ps)//2 + i*hr_ps:(hr_w-(hr_w//hr_ps)*hr_ps)//2+(i+1)*hr_ps] = grayscale_cams[j*((hr_w-(hr_w-(hr_w//hr_ps)*hr_ps))//hr_ps) + i]
                        # for j in range((hr_h-hr_ps//2) // hr_ps):
                        #     for i in range((hr_w-hr_ps//2) // hr_ps):
                        #         sr[0, :, hr_ps//2 + j*hr_ps:hr_ps//2+(j+1)*hr_ps, hr_ps//2 + i*hr_ps:hr_ps//2+(i+1)*hr_ps] = sr_patches[j*((hr_w-hr_ps)//hr_ps) + i]
                        #         grayscale[hr_ps//2 + j*hr_ps:hr_ps//2+(j+1)*hr_ps, hr_ps//2 + i*hr_ps:hr_ps//2+(i+1)*hr_ps] = grayscale_cams[j*((hr_w-hr_ps)//hr_ps) + i]
                        ### patch superresolution
                    else:
                        ### image superresolution
                        sr = sr_patches[0]
                        grayscale = grayscale_cams[0]
                        ### image superresolution

                LOG.info(str(t))
                version='e150'
                grayscale = torch.Tensor(grayscale).to(self.device)
        #        grayscale = torch.sigmoid(grayscale)
                # cam_img = show_cam_on_image(np_tsr_hr.squeeze(0).transpose(1,2,0), grayscale.cpu().detach().numpy(), use_rgb=True, image_weight=0.3)
                # cam_img = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f'./results_230809/CAM/{version}_{k}_patch_{imagename}_y_{nv}px_cam.jpg', cam_img)
                grayscale = torch.tanh(grayscale)
                # cam_img = show_cam_on_image(np_tsr_hr.squeeze(0).transpose(1,2,0), grayscale.cpu().detach().numpy(), use_rgb=True, image_weight=0.3)
                # cam_img = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f'./results_230809/CAM/{version}_{k}_patch_{imagename}_y_{nv}px_tanh_cam.jpg', cam_img)


                grayscale = grayscale / torch.max(grayscale)
                # 
                # grayscale = grayscale / torch.max(grayscale)
                # breakpoint()
                # breakpoint()
                # sr = sr.clip(0,1)
                # cam_img = show_cam_on_image(sr.squeeze(0).transpose(1,2,0), grayscale.cpu().detach().numpy(), use_rgb=True, image_weight=0.3)
                cam_img = show_cam_on_image(np_tsr_hr.squeeze(0).transpose(1,2,0), grayscale.cpu().detach().numpy(), use_rgb=True, image_weight=0.3)
                cam_img = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
        #        cv2.imwrite(f'./results/0801-img-tanh-cam.jpg', cam_img)
                # if k==0:
                #     cv2.imwrite(f'./results/{imagename}sr{nv}%-img-tanh-cam.jpg', cam_img)
                # if k==0:
                #     cv2.imwrite(f'./results_230809/CAM/150_patch_{imagename}sry%{nv}-img-tanh-cam.jpg', cam_img)
                # elif k==1:
                #     cv2.imwrite(f'./results_230809/CAM/150_patch_{imagename}srx%{nv}-img-tanh-cam.jpg', cam_img)
                if PSR:
                    cv2.imwrite(f'./results_230809/CAM/{version}_{nv2}_patch_{imagename}_y_{nv}px_tanh_norm_cam.jpg', cam_img)
                else:
                    if nv2==2:
                        cv2.imwrite(f'./results_230809/CAM/{version}_{nv2}_image_{imagename}_y_{nv}px_tanh_norm_cam.jpg', cam_img)
                # elif k==1:
                    # cv2.imwrite(f'./results_230809/CAM/150_patch_{imagename}srxpx{nv}-img-tanh-cam.jpg', cam_img)




        sr = torch.Tensor(sr).to(self.device)
#        sr = to_tensor(np.array(sr, np.uint8)).to(self.device)

        return sr


    def train_iter_lre_higher(self):
        with torch.enable_grad():
            self.model.train()
            t = tqdm(range(len(self.loader_train) * args.dataset_repeat))
            t.set_description(f"Epoch {self.epoch} train ")
            loss_avg = AverageMeter(0.05)
            l1_avg = AverageMeter(0.05)
            l2_avg = AverageMeter(0.05)
            leakfinder = LeakFinder()
            self.iter = 0
            for i in range(args.dataset_repeat):
                for hr, lr in self.loader_train:
                    hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device)
                    hr, lr = hr.to(device=self.device, non_blocking=True),\
                             lr.to(device=self.device, non_blocking=True)
                    
                    self.optimizer.zero_grad()
                    with higher.innerloop_ctx(self.model, self.optimizer) as (meta_model, meta_opt):
                        # 1. Update meta model on training data
                        meta_train_outputs = meta_model(lr)
                        meta_train_outputs = self.process_for_eval(meta_train_outputs)
                        hr = self.process_for_eval(hr)

                        self.loss_fn.reduction = 'none'
                        #cost = nn.L1Loss(reduce=False)(y_f_hat, hr)
                        meta_train_loss = self.loss_fn(meta_train_outputs, hr)
#                        meta_train_loss = criterion(meta_train_outputs, hr.type_as(outputs))

                        if args.reweight == 0:
                            meta_train_loss = torch.sum(meta_train_loss, dim=(1,2,3))
                            eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=self.device)
                        elif args.reweight == 1:
                            patch_size = 32
                            meta_train_loss_tmp = torch.zeros([meta_train_loss.shape[0],meta_train_loss.shape[1], args.patch_size_train//patch_size, args.patch_size_train//patch_size], device=self.device)
                            for i in range(int(args.patch_size_train//patch_size)):
                                for j in range(int(args.patch_size_train//patch_size)):
                                    meta_train_loss_tmp[:,:,i,j] = torch.sum(meta_train_loss[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size], dim=(2,3))
                            meta_train_loss = meta_train_loss_tmp
                            eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=self.device)

                        else:
                            eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=self.device)
                        
                        meta_train_loss = torch.sum(eps * meta_train_loss)
                        meta_opt.step(meta_train_loss)

                        # 2. Compute grads of eps on meta validation data
                        meta_labels, meta_inputs =  next(self.loader_traval)
                        meta_inputs, meta_labels =  meta_inputs.to(self.dtype).to(self.device),\
                                                    meta_labels.to(self.dtype).to(self.device)
                        meta_inputs, meta_labels = meta_inputs.to(device=self.device, non_blocking=True),\
                                                   meta_labels.to(device=self.device, non_blocking=True)
                        meta_val_outputs = meta_model(meta_inputs)
                        meta_labels = self.process_for_eval(meta_labels)
                        meta_val_outputs = self.process_for_eval(meta_val_outputs)
    
                        self.loss_fn.reduction = 'mean'
                        meta_val_loss = self.loss_fn(meta_val_outputs, meta_labels)
#                        meta_val_loss = criterion(meta_val_outputs, meta_labels.type_as(outputs))
                        eps_grads = torch.autograd.grad(meta_val_loss, eps)[0].detach()

                    # 3. Compute weights for current training batch
                    w_tilde = torch.clamp(-eps_grads, min=0)
                    l1_norm = torch.sum(w_tilde)
                    if l1_norm != 0:
                        w = w_tilde / l1_norm
                    else:
                        w = w_tilde
                    
                    # 4. Train model on weighted batch
                    sr = self.model(lr)
                    sr = self.process_for_eval(sr)
                    self.loss_fn.reduction = 'none'
                    minibatch_loss = self.loss_fn(sr, hr)
                    if args.reweight == 0:
                        minibatch_loss = torch.sum(minibatch_loss, dim=(1,2,3))
                    elif args.reweight == 1:
                        minibatch_loss_tmp = torch.zeros([minibatch_loss.shape[0],minibatch_loss.shape[1], args.patch_size_train//patch_size, args.patch_size_train//patch_size], device=self.device)
                        for i in range(int(args.patch_size_train//patch_size)):
                            for j in range(int(args.patch_size_train//patch_size)):
                                minibatch_loss_tmp[:,:,i,j] = torch.sum(minibatch_loss[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size], dim=(2,3))
                        minibatch_loss = minibatch_loss_tmp
                    loss = torch.sum(minibatch_loss * w)
                    loss.backward()
                    if args.gradient_clipping is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), args.gradient_clipping)
                    self.optimizer.step()

                    # keep track of epoch loss/accuracy
                    l1_loss = nn.functional.l1_loss(sr, hr).item()
                    l2_loss = torch.sqrt(nn.functional.mse_loss(sr, hr)).item()
                    l1_avg.update(l1_loss)
                    l2_avg.update(l2_loss)
                    args_dic = {
                        'L1': f'{l1_avg.get():.4f}',
                        'L2': f'{l2_avg.get():.4f}'
                    }
                    if args.loss not in [LossType.L1, LossType.L2]:
                        loss_avg.update(loss.item())
                        args_dic[args.loss.name] = f'{loss_avg.get():.4f}'
                    t.update()
                    t.set_postfix(**args_dic)
            
        LOG.info(str(t))


    def train_iter_mwn(self):
        with torch.enable_grad():
            self.model.train()
            t = tqdm(range(len(self.loader_train) * args.dataset_repeat))
            t.set_description(f"Epoch {self.epoch} train ")
            loss_avg = AverageMeter(0.05)
            l1_avg = AverageMeter(0.05)
            l2_avg = AverageMeter(0.05)
            self.iter = 0
            for i in range(args.dataset_repeat):
                for hr, lr in self.loader_train:
                    hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device)
                    hr, lr = hr.to(device=self.device, non_blocking=True),\
                             lr.to(device=self.device, non_blocking=True)
                    
                    self.optimizer.zero_grad()
                    with higher.innerloop_ctx(self.model, self.optimizer) as (meta_model, meta_opt):
                        # 1. Update meta model on training data
                        meta_train_outputs = meta_model(lr)
                        meta_train_outputs = self.process_for_eval(meta_train_outputs)
                        hr = self.process_for_eval(hr)

                        self.loss_fn.reduction = 'none'
                        meta_train_loss = self.loss_fn(meta_train_outputs, hr)

                        if args.reweight == 0:
                            meta_train_loss = torch.sum(meta_train_loss, dim=(1,2,3))
                            eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=self.device)
                        elif args.reweight == 1:
                            patch_size = 32
                            meta_train_loss_tmp = torch.zeros([meta_train_loss.shape[0],meta_train_loss.shape[1], args.patch_size_train//patch_size, args.patch_size_train//patch_size], device=self.device)
                            for i in range(int(args.patch_size_train//patch_size)):
                                for j in range(int(args.patch_size_train//patch_size)):
                                    meta_train_loss_tmp[:,:,i,j] = torch.sum(meta_train_loss[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size], dim=(2,3))
                            meta_train_loss = meta_train_loss_tmp
                            eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=self.device)
                        else:
                            # cost_v = torch.reshape(cost, (len(cost), 1))
                            # meta_train_loss = torch.mean(meta_train_loss, dim=(1,2,3))
                            # meta_train_loss = torch.reshape(meta_train_loss, (len(meta_train_loss), 1))
                            # breakpoint()
                            # breakpoint()
                            eps = self.vnet(meta_train_loss.data)
                        
        
                        # v_lambda = torch.clamp(-v_lambda, min=0)
                        norm_c = torch.sum(eps)
                        # l_f_meta = torch.sum(cost * v_lambda)/len(cost)
                        if norm_c != 0:
                            eps_norm = eps / norm_c
                        else:
                            eps_norm = eps

                        meta_train_loss = torch.sum(meta_train_loss * eps_norm)
                        meta_opt.step(meta_train_loss)

                        # 2. Compute grads of eps on meta validation data
                        meta_labels, meta_inputs =  next(self.loader_traval)
                        meta_inputs, meta_labels =  meta_inputs.to(self.dtype).to(self.device),\
                                                    meta_labels.to(self.dtype).to(self.device)
                        meta_inputs, meta_labels = meta_inputs.to(device=self.device, non_blocking=True),\
                                                   meta_labels.to(device=self.device, non_blocking=True)
                        meta_val_outputs = meta_model(meta_inputs)
                        meta_labels = self.process_for_eval(meta_labels)
                        meta_val_outputs = self.process_for_eval(meta_val_outputs)
                        
                        self.loss_fn.reduction = 'mean'
                        meta_val_loss = self.loss_fn(meta_val_outputs, meta_labels)

                        meta_val_loss = torch.mean(meta_val_loss)

                        # self.vnet_opt.zero_grad()
                        # meta_val_loss.backward()
                        # # # print(self.vnet.conv1.weight.grad)
                        # self.vnet_opt.step()
                        
                        # self.vnet_opt.step(meta_val_loss)
                        eps_grads = torch.autograd.grad(meta_val_loss, (self.vnet.params()), create_graph=True)
                        self.vnet.update_params(1e-5, source_params=eps_grads)
                        del eps_grads


                    # 4. Train model on weighted batch
                    sr = self.model(lr)
                    sr = self.process_for_eval(sr)
                    self.loss_fn.reduction = 'none'
                    minibatch_loss = self.loss_fn(sr, hr)

                    if args.reweight == 0:
                        minibatch_loss = torch.sum(minibatch_loss, dim=(1,2,3))
                    elif args.reweight == 1:
                        minibatch_loss_tmp = torch.zeros([minibatch_loss.shape[0],minibatch_loss.shape[1], args.patch_size_train//patch_size, args.patch_size_train//patch_size], device=self.device)
                        for i in range(int(args.patch_size_train//patch_size)):
                            for j in range(int(args.patch_size_train//patch_size)):
                                minibatch_loss_tmp[:,:,i,j] = torch.sum(minibatch_loss[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size], dim=(2,3))
                        minibatch_loss = minibatch_loss_tmp
                    
                    # breakpoint()

                    # minibatch_loss = torch.mean(minibatch_loss, dim=(1,2,3))
                    # minibatch_loss = torch.reshape(minibatch_loss, (len(minibatch_loss), 1))
                    
                    with torch.no_grad():
                        w_tilde = self.vnet(minibatch_loss)

                    # w_tilde = torch.clamp(w_tilde, min=0)
                    l1_norm = torch.sum(w_tilde)
                    if norm_c != 0:
                        w = w_tilde / l1_norm
                    else:
                        w = w_tilde

                    loss = torch.sum(minibatch_loss * w)
                    loss.backward()
                    if args.gradient_clipping is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), args.gradient_clipping)
                    self.optimizer.step()

                    # keep track of epoch loss/accuracy
                    l1_loss = nn.functional.l1_loss(sr, hr).item()
                    l2_loss = torch.sqrt(nn.functional.mse_loss(sr, hr)).item()
                    l1_avg.update(l1_loss)
                    l2_avg.update(l2_loss)
                    args_dic = {
                        'L1': f'{l1_avg.get():.4f}',
                        'L2': f'{l2_avg.get():.4f}'
                    }
                    if args.loss not in [LossType.L1, LossType.L2]:
                        loss_avg.update(loss.item())
                        args_dic[args.loss.name] = f'{loss_avg.get():.4f}'
                    t.update()
                    t.set_postfix(**args_dic)
            
        LOG.info(str(t))


    def train_iter_warpi(self):
        with torch.enable_grad():
            self.model.train()
            t = tqdm(range(len(self.loader_train) * args.dataset_repeat))
            t.set_description(f"Epoch {self.epoch} train ")
            loss_avg = AverageMeter(0.05)
            l1_avg = AverageMeter(0.05)
            l2_avg = AverageMeter(0.05)
            self.iter = 0
            for i in range(args.dataset_repeat):
                for hr, lr in self.loader_train:
                    hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device)
                    hr, lr = hr.to(device=self.device, non_blocking=True),\
                             lr.to(device=self.device, non_blocking=True)
                    
                    self.optimizer.zero_grad()
                    with higher.innerloop_ctx(self.model, self.optimizer) as (meta_model, meta_opt):
                        # 1. Update meta model on training data
                        meta_train_outputs = meta_model(lr)
                        meta_train_outputs = self.process_for_eval(meta_train_outputs)
                        hr = self.process_for_eval(hr)

                        # self.loss_fn.reduction = 'none'
                        # cost = self.loss_fn(meta_train_outputs, hr)
                        if args.reweight == 0:
                            meta_train_loss = torch.sum(meta_train_loss, dim=(1,2,3))
                            eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=self.device)
                        elif args.reweight == 1:
                            patch_size = 32
                            meta_train_loss_tmp = torch.zeros([meta_train_loss.shape[0],meta_train_loss.shape[1], args.patch_size_train//patch_size, args.patch_size_train//patch_size], device=self.device)
                            for i in range(int(args.patch_size_train//patch_size)):
                                for j in range(int(args.patch_size_train//patch_size)):
                                    meta_train_loss_tmp[:,:,i,j] = torch.sum(meta_train_loss[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size], dim=(2,3))
                            meta_train_loss = meta_train_loss_tmp
                            eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=self.device)
                        else:
                            # cost_v = torch.reshape(cost, (len(cost), 1))
                            meta_train_outputs_v = torch.sum(meta_train_outputs, dim=(1,2,3))
                            meta_train_outputs_v = torch.reshape(meta_train_outputs_v, (len(meta_train_outputs), 1))                            
                            v_lambda = self.vnet2(meta_train_outputs_v.detach(), hr, 3)
                            self.loss_fn.reduction = 'none'
                            l_f_meta = self.loss_fn(v_lambda * meta_train_outputs, hr)

                        
        
                        # v_lambda = torch.clamp(-v_lambda, min=0)
                        # norm_c = torch.sum(l_f_meta)
                        # # l_f_meta = torch.sum(cost * v_lambda)/len(cost)
                        # if norm_c != 0:
                        #     l_f_meta_norm = l_f_meta / norm_c
                        # else:
                        #     l_f_meta_norm = l_f_meta




                        # l_f_meta = torch.sum(cost * l_f_meta_norm)
                        meta_opt.step(l_f_meta)

                        # 2. Compute grads of eps on meta validation data
                        meta_labels, meta_inputs =  next(self.loader_traval)
                        meta_inputs, meta_labels =  meta_inputs.to(self.dtype).to(self.device),\
                                                    meta_labels.to(self.dtype).to(self.device)
                        meta_inputs, meta_labels = meta_inputs.to(device=self.device, non_blocking=True),\
                                                   meta_labels.to(device=self.device, non_blocking=True)
                        meta_val_outputs = meta_model(meta_inputs)
                        meta_labels = self.process_for_eval(meta_labels)
                        meta_val_outputs = self.process_for_eval(meta_val_outputs)
                        
                        self.loss_fn.reduction = 'mean'
                        meta_val_loss = self.loss_fn(meta_val_outputs, meta_labels)
                        self.vnet_opt.zero_grad()
                        meta_val_loss.backward()
                        # print(self.vnet.conv1.weight.grad)
                        # breakpoint()
                        self.vnet_opt.step()
                        # eps_grads = torch.autograd.grad(meta_val_loss)[0].detach()


                    # 4. Train model on weighted batch
                    sr = self.model(lr)
                    sr = self.process_for_eval(sr)
                    # self.loss_fn.reduction = 'none'
                    # minibatch_loss = self.loss_fn(sr, hr)
                    # minibatch_loss_v = torch.reshape(minibatch_loss, (len(minibatch_loss), 1))

                    if args.reweight == 0:
                        minibatch_loss = torch.sum(minibatch_loss, dim=(1,2,3))
                    elif args.reweight == 1:
                        minibatch_loss_tmp = torch.zeros([minibatch_loss.shape[0],minibatch_loss.shape[1], args.patch_size_train//patch_size, args.patch_size_train//patch_size], device=self.device)
                        for i in range(int(args.patch_size_train//patch_size)):
                            for j in range(int(args.patch_size_train//patch_size)):
                                minibatch_loss_tmp[:,:,i,j] = torch.sum(minibatch_loss[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size], dim=(2,3))
                        minibatch_loss = minibatch_loss_tmp

                    # minibatch_loss = torch.sum(minibatch_loss, dim=(1,2,3))
                    # minibatch_loss = torch.reshape(minibatch_loss, (len(minibatch_loss), 1))
                    sr_v = torch.sum(sr, dim=(1,2,3))
                    sr_v = torch.reshape(sr_v, (len(sr), 1))                            

                    with torch.no_grad():
                        w_new = self.vnet2(sr_v.detach(), hr, 3)

                    self.loss_fn.reduction = 'none'
                    loss = self.loss_fn(w_new * sr, hr)

                    # w_new = torch.clamp(-w_new, min=0)
                    # norm_v = torch.sum(w_new)
                    # if norm_c != 0:
                    #     w_v = w_new / norm_v
                    # else:
                    #     w_v = w_new

                    # loss = torch.sum(minibatch_loss * w_v)
                    loss.backward()
                    if args.gradient_clipping is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), args.gradient_clipping)
                    self.optimizer.step()

                    # keep track of epoch loss/accuracy
                    l1_loss = nn.functional.l1_loss(sr, hr).item()
                    l2_loss = torch.sqrt(nn.functional.mse_loss(sr, hr)).item()
                    l1_avg.update(l1_loss)
                    l2_avg.update(l2_loss)
                    args_dic = {
                        'L1': f'{l1_avg.get():.4f}',
                        'L2': f'{l2_avg.get():.4f}'
                    }
                    if args.loss not in [LossType.L1, LossType.L2]:
                        loss_avg.update(loss.item())
                        args_dic[args.loss.name] = f'{loss_avg.get():.4f}'
                    t.update()
                    t.set_postfix(**args_dic)
            
        LOG.info(str(t))


#     def train_iter_lre(self):
#         with torch.enable_grad():
#             self.model.train()
#             t = tqdm(range(len(self.loader_train) * args.dataset_repeat))
#             t.set_description(f"Epoch {self.epoch} train ")
#             loss_avg = AverageMeter(0.05)
#             l1_avg = AverageMeter(0.05)
#             l2_avg = AverageMeter(0.05)
#             leakfinder = LeakFinder()
#             self.iter = 0
#             for i in range(args.dataset_repeat):
#                 for hr, lr in self.loader_train:
#                     print('step 0')
# #                    leakfinder.set_batch(self.iter)
#                     hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device)
#                     for val_hr, val_lr in self.loader_traval:
#                         val_hr, val_lr = val_hr.to(self.dtype).to(self.device), val_lr.to(self.dtype).to(self.device)
#                         break
                   
#                     ### SHKIM Reweight
#                     self.meta_net.load_state_dict(self.model.state_dict())
#                     self.meta_net.cuda()
#                     lr = to_var(lr, requires_grad=False)
#                     hr = to_var(hr, requires_grad=False)
#                     val_lr = to_var(val_lr, requires_grad=False)
#                     var_hr = to_var(val_lr, requires_grad=False)
#                     y_f_hat = self.meta_net(lr)
#                     y_f_hat = self.process_for_eval(y_f_hat)
#                     hr = self.process_for_eval(hr)
#                     cost = nn.L1Loss(reduce=False)(y_f_hat, hr)
# #                    eps = nn.Parameter(torch.zeros(cost.size()).cuda())
#                     eps = to_var(torch.zeros(cost.size()), requires_grad=True)

#                     l_f_meta = torch.sum(cost*eps)
#                     self.meta_net.zero_grad()
#                     self.optimizer.zero_grad()
# #                    for name, param in self.meta_net.named_parameters():
# #                        if param.requires_grad is False:
# #                            param.requires_grad = True

#                     grads = torch.autograd.grad(l_f_meta, (self.meta_net.params()), create_graph=True)
# #                    self.meta_net.update_params(0.1, source_params=grads)
#                     current_lr = self.optimizer.param_groups[0]['lr']
# #                    print(self.optimizer)
# #                    print(self.optimizer.param_groups[0])

#                     self.meta_net.update_params(current_lr, source_params=grads)

#                     print('step 1')
#                     breakpoint()
# #                    leakfinder.get_cuda_perc() 
# #                    leakfinder.find_leaks() 
# #                    self.iter += 1

#                     self.meta_net.eval()
#                     y_g_hat = self.meta_net(val_lr)
#                     y_g_hat = self.process_for_eval(y_g_hat)
#                     val_hr = self.process_for_eval(val_hr)
#                     l_g_meta = self.loss_fn(y_g_hat, val_hr)
# #                    for param in eps:
# #                        if param.requires_grad is False:
# #                            print(param)
# #                            param.requires_grad = True
# #                        else:
# #                            print('True')
#                     grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, allow_unused=True)[0]
#                     w_tilde = torch.clamp(-grad_eps, min=0)
#                     norm_c = torch.sum(w_tilde)
#                     if norm_c != 0:
#                         w = w_tilde / norm_c
#                     else:
#                         w = w_tilde
                    
#                     print('step 2')
#                     breakpoint()

#                     self.optimizer.zero_grad()
#                     y_f_hat = self.model(lr)
#                     y_f_hat = self.process_for_eval(y_f_hat)
#                     cost = nn.L1Loss(reduce=False)(y_f_hat, hr)
#                     loss = torch.sum(cost * w)
#                     loss.backward()

# #                    self.optimizer.zero_grad()
# #                    sr = self.model(lr)
# #                    sr = self.process_for_eval(sr)
# #                    hr = self.process_for_eval(hr)
# #                    loss = self.loss_fn(sr, hr)
# #                    loss.backward()
#                     if args.gradient_clipping is not None:
#                         nn.utils.clip_grad_norm_(self.model.parameters(), args.gradient_clipping)
#                     self.optimizer.step()
#                     l1_loss = nn.functional.l1_loss(y_f_hat, hr).item()
#                     l2_loss = torch.sqrt(nn.functional.mse_loss(y_f_hat, hr)).item()
#                     l1_avg.update(l1_loss)
#                     l2_avg.update(l2_loss)
#                     args_dic = {
#                         'L1': f'{l1_avg.get():.4f}',
#                         'L2': f'{l2_avg.get():.4f}'
#                     }
#                     if args.loss not in [LossType.L1, LossType.L2]:
#                         loss_avg.update(loss.item())
#                         args_dic[args.loss.name] = f'{loss_avg.get():.4f}'
#                     t.update()
#                     t.set_postfix(**args_dic)
#                     print('step 3')
#                     breakpoint()

            
#         LOG.info(str(t))


    def val_iter(self, final=True):
        with torch.no_grad():
            self.model.eval()
            t = tqdm(self.loader_val)
            if final:
                t.set_description("Validation")
            else:
                t.set_description(f"Epoch {self.epoch} val   ")
            psnr_avg = AverageMeter()
            ssim_avg = AverageMeter()
            if args.metric > 0:
                ### SHKIM fsim / lpips
                fsim_avg = AverageMeter()
                lpips_avg = AverageMeter()
                if args.metric > 1:
                    niqe_avg = AverageMeter()
                    musiq_avg = AverageMeter()
                    nrqm_avg = AverageMeter()
                ### SHKIM fsim / lpips
            l1_avg = AverageMeter()
            l2_avg = AverageMeter()
            loss_avg = AverageMeter()
            for hr, lr in t:
                hr, lr = hr.to(self.dtype).to(self.device), lr.to(self.dtype).to(self.device)
                sr = self.model(lr).clamp(0, 1)
                if final:
                    # Round to pixel values
                    sr = sr.mul(255).round().div(255)
                sr = self.process_for_eval(sr)
                hr = self.process_for_eval(hr)
                self.loss_fn.reduction = 'mean'
                loss = self.loss_fn(sr, hr)
                l1_loss = nn.functional.l1_loss(sr, hr).item()
                l2_loss = torch.sqrt(nn.functional.mse_loss(sr, hr)).item()
                psnr = self.psnr(hr, sr)
                ssim = self.ssim(hr, sr)
                if args.metric > 0:
                    ### SHKIM fsim / lpips
                    fsim = self.fsim(hr, sr)
                    lpips_score = self.lpips_model.forward(hr, sr)
                    if args.metric > 1:
                        niqe = self.niqe(sr)
                        musiq = self.musiq(sr)
                        nrqm = self.nrqm(sr)
                    ### SHKIM fsim / lpips
                loss_avg.update(loss.item())
                l1_avg.update(l1_loss) #.item()
                l2_avg.update(l2_loss) #.item()
                psnr_avg.update(psnr.item()) #.item()
                ssim_avg.update(ssim.item()) #.item()
                if args.metric > 0:
                    ### SHKIM fsim / lpips
                    fsim_avg.update(fsim.item()) #.item()
                    lpips_avg.update(lpips_score.item())
                    if args.metric > 1:
                        niqe_avg.update(niqe.item())
                        musiq_avg.update(musiq.item())
                        nrqm_avg.update(nrqm.item())
                    ### SHKIM fsim / lpips
                        args_dic = {
                            'PSNR': f'{psnr_avg.get():.4f}',
                            'SSIM': f'{ssim_avg.get():.4f}',
                            'FSIM': f'{fsim_avg.get():.4f}',
                            'LPIPS': f'{lpips_avg.get():.4f}',
                            'NIQE': f'{niqe_avg.get():.4f}',
                            'MUSIQ': f'{musiq_avg.get():.4f}',
                            'NRQM': f'{nrqm_avg.get():.4f}',
                            'L1': f'{l1_avg.get():.4f}',
                            'L2': f'{l2_avg.get():.4f}',
                        }
                    else:
                        args_dic = {
                        'PSNR': f'{psnr_avg.get():.4f}',
                        'SSIM': f'{ssim_avg.get():.4f}',
                        'FSIM': f'{fsim_avg.get():.4f}',
                        'LPIPS': f'{lpips_avg.get():.4f}',
                        'L1': f'{l1_avg.get():.4f}',
                        'L2': f'{l2_avg.get():.4f}',
                    }
                else:
                    args_dic = {
                        'PSNR': f'{psnr_avg.get():.4f}',
                        'SSIM': f'{ssim_avg.get():.4f}',
                        'L1': f'{l1_avg.get():.4f}',
                        'L2': f'{l2_avg.get():.4f}',
                    }

                if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM]:
                    args_dic[args.loss.name] = f'{loss_avg.get():.4f}'
                t.set_postfix(**args_dic)
            
            if self.writer is not None:
                self.writer.add_scalar('PSNR', psnr_avg.get(), self.epoch)
                self.writer.add_scalar('SSIM', ssim_avg.get(), self.epoch)

                if args.metric > 0:
                    self.writer.add_scalar('FSIM', fsim_avg.get(), self.epoch)
                    self.writer.add_scalar('LPIPS', lpips_avg.get(), self.epoch)
                    if args.metric > 1:
                        self.writer.add_scalar('NIQE', niqe_avg.get(), self.epoch)
                        self.writer.add_scalar('MUSIQ', musiq_avg.get(), self.epoch)
                        self.writer.add_scalar('NRQM', nrqm_avg.get(), self.epoch)
                self.writer.add_scalar('L1', l1_avg.get(), self.epoch)
                self.writer.add_scalar('L2', l2_avg.get(), self.epoch)
                if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM]:
                    self.writer.add_scalar(args.loss.name, loss_avg.get(), self.epoch)
            LOG.info(str(t))
            if args.metric > 0:
                if args.metric > 1:
                    return loss_avg.get(), psnr_avg.get(), ssim_avg.get(), fsim_avg.get(), lpips_avg.get(), niqe_avg.get(), musiq_avg.get(), nrqm_avg.get()
                else:
                    return loss_avg.get(), psnr_avg.get(), ssim_avg.get(), fsim_avg.get(), lpips_avg.get()
            else:
                return loss_avg.get(), psnr_avg.get(), ssim_avg.get()

    def validation(self):
        if args.metric > 0:
            if args.metric > 1:
                loss, psnr, ssim, fsim, lpips, niqe, musiq, nrqm = self.val_iter()
                if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM]:
                    print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, FSIM: {fsim:.4f}, LPIPS: {lpips:.4f}, NIQE: {niqe:.4f}, MUSIQ: {musiq:.4f}, NRQM: {nrqm:.4f}, loss: {loss:.4f}")
                else:
                    print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, FSIM: {fsim:.4f}, LPIPS: {lpips:.4f}, NIQE: {niqe:.4f}, MUSIQ: {musiq:.4f}, NRQM: {nrqm:.4f}, ")

            else:
                loss, psnr, ssim, fsim, lpips = self.val_iter()
            # if args.lre and args.noise_value != 0:
            #     fid = self.fid('results_230809/LRE', 'results_230809/HR')
            # elif args.noise_value == 0:
            #     fid = self.fid('results_230809/Align', 'results_230809/HR')
            # else:
            #     fid = self.fid('results_230809/unalign', 'results_230809/HR')
            # print(f"FID:{fid:.4f}")
                if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM]:
                    print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, FSIM: {fsim:.4f}, LPIPS: {lpips:.4f}, loss: {loss:.4f}")
                else:
                    print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, FSIM: {fsim:.4f}, LPIPS: {lpips:.4f} ")

        else:
            loss, psnr, ssim = self.val_iter()
            fsim, lpips, fid, niqe, musiq, nrqm = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            if args.loss not in [LossType.L1, LossType.L2, LossType.SSIM]:
                print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, loss: {loss:.4f}")
            else:
                print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f},")
            
        
        
    def run_model(self):
        scale = args.scale
        with torch.no_grad():
            self.model.eval()
            input_images = []
            high_input_images = []
            for f in args.images:
                if os.path.isdir(f):
                    for g in os.listdir(f):
                        n = os.path.join(f, g)
                        if os.path.isfile(n):
                            if '.png' in n:
                                input_images.append(n)
                else:
                    input_images.append(f)

            for f in args.hr:
                if os.path.isdir(f):
                    for g in os.listdir(f):
                        n = os.path.join(f, g)
                        if os.path.isfile(n):
                            if '.png' in n:
                                high_input_images.append(n)
                else:
                    high_input_images.append(f)

            input_images.sort()
            high_input_images.sort()
            if args.destination is None:
                raise ValueError("You should specify a destination directory")
            os.makedirs(args.destination, exist_ok=True)

            t = tqdm(input_images)
            psnr_avg = AverageMeter()
            ssim_avg = AverageMeter()
            ### SHKIM fsim / lpips
            fsim_avg = AverageMeter()
            lpips_avg = AverageMeter()

            t.set_description("Run")
            for i, filename in enumerate(t):
                try:
                    hr = Image.open(high_input_images[i])
                    hr.load()
                    img = Image.open(filename)
                    img.load()
                except:
                    print(f"Could not open {filename}")
                    continue
                if hr.size[0] != img.size[0]*args.scale or hr.size[1] != img.size[1]*args.scale:
                    hr = hr.resize((img.size[0]*2, img.size[1]*2))
                    
                img = to_tensor(img).to(self.device)
                hr = to_tensor(hr).to(self.device)
                sr_img = self.model(img).clamp(0, 1)
                    

                psnr = self.psnr(hr, sr_img)
                ssim = self.ssim(hr, sr_img)

                psnr_avg.update(psnr.item())
                ssim_avg.update(ssim.item())
                ### SHKIM fsim / lpips

                sr_img = to_image(sr_img)

                destname = os.path.splitext(os.path.basename(filename))[0] + f"_x{scale}.png"
                sr_img.save(os.path.join(args.destination, destname))

            print(f"PSNR: {psnr_avg.get():.2f}, SSIM: {ssim_avg.get():.4f}")


    def run_lre_model(self):
        scale =args.scale
#        with torch.enable_grad():
#            self.model.eval()
        input_images = []
        high_input_images = []
        for f in args.images:
            if os.path.isdir(f):
                for g in os.listdir(f):
                    n = os.path.join(f, g)
                    if os.path.isfile(n):
                        if '.png' in n:
                            input_images.append(n)
            else:
                input_images.append(f)
        for f in args.hr:
            if os.path.isdir(f):
                for g in os.listdir(f):
                    n = os.path.join(f, g)
                    if os.path.isfile(n):
                        if '.png' in n:
                            high_input_images.append(n)
            else:
                high_input_images.append(f)
        if args.destination is None:
            raise ValueError("You should specify a destination directory")
        os.makedirs(args.destination, exist_ok=True)

        t = tqdm(input_images)
        t.set_description("Run")
        for i, filename in enumerate(t):
            try:
                lr = Image.open(filename)
                hr = Image.open(high_input_images[i])
                lr.load()
                hr.load()
            except:
                print(f"Could not open {filename}")
                continue

            
            

            if hr.size[0] != lr.size[0]*args.scale or hr.size[1] != lr.size[1]*args.scale:
                hr = hr.resize((lr.size[0]*2, lr.size[1]*2))
#            lr = to_tensor(lr).to(self.device)
#            hr = to_tensor(hr).to(self.device)

            # if hr.size[0] > 1000 or hr.size[1] >1000:
            #     from ..transforms import transforms
            #     hr = transforms.Crop(1024, allow_smaller=True, scales=[1, args.scale])(hr)
            #     lr = transforms.Crop(512, allow_smaller=True, scales=[1, args.scale])(lr)

            sr_img = self.test_iter_lre_higher(lr, hr).clamp(0, 1)
            
            # sr_img = self.model(img)
            # sr_img = to_image(sr_img)
            hr = to_tensor(hr).to(self.device)
            sr_img = to_image(sr_img)
            sr_img = to_tensor(sr_img).to(self.device)
            
            psnr = self.psnr(hr, sr_img)
            ssim = self.ssim(hr, sr_img)
            ### SHKIM fsim / lpips
            t.set_postfix(PSNR=f'{psnr.item():.2f}', SSIM=f'{ssim.item():.4f}')
            
            sr_img = to_image(sr_img)
            hr = to_image(hr)

            destname = os.path.splitext(os.path.basename(filename))[0] + f"_x{scale}_3.png"
            sr_img.save(os.path.join(args.destination, destname))
            destname = os.path.splitext(os.path.basename(filename))[0] + f"_3_HR.png"
            hr.save(os.path.join(args.destination, destname))
            
            
            # destname = os.path.splitext(os.path.basename(filename))[0] + f"_x{scale}.png"
            # sr_img.save(os.path.join(args.destination, destname))


    def train(self):
        t = tqdm(total=args.epochs, initial=self.epoch)
        t.set_description("Epochs")
        if self.best_epoch is not None:
            args_dic = {'best': self.best_epoch}
            if self.best_psnr is not None:
                args_dic['PSNR'] = f'{self.best_psnr:.2f}'
            if self.best_ssim is not None:
                args_dic['SSIM'] = f'{self.best_ssim:.2f}'
            if self.best_fsim is not None:
                args_dic['FSIM'] = f'{self.best_fsim:.2f}'
            if self.best_lpips is not None:
                args_dic['LPIPS'] = f'{self.best_lpips:.2f}'
            if self.best_fid is not None:
                args_dic['FID'] = f'{self.best_fid:.2f}'
            if self.best_niqe is not None:
                args_dic['NIQE'] = f'{self.best_niqe:.2f}'
            if self.best_musiq is not None:
                args_dic['MUSIQ'] = f'{self.best_musiq:.2f}'
            if self.best_nrqm is not None:
                args_dic['NRQM'] = f'{self.best_nrqm:.2f}'
            if self.best_loss is not None:
                args_dic['loss'] = f'{self.best_loss:.2f}'
            t.set_postfix(**args_dic)

        ### SHKIM To check loader
        if args.load_checkpoint is not None :
            if os.path.isfile(args.load_checkpoint):
                if args.metric > 0 :
                    if args.metric > 1 :
                        loss, psnr, ssim, fsim, lpips, niqe, musiq, nrqm = self.val_iter(final=False)
                    else:
                        loss, psnr, ssim, fsim, lpips = self.val_iter(final=False)

                else:
                    loss, psnr, ssim = self.val_iter(final=False)


        ### SHKIM lre
        if args.lre:
            train_epoch = args.epochs/2
            # train_epoch = 0
        else:
            train_epoch = args.epochs

        while self.epoch < train_epoch:
            self.epoch += 1
            self.train_iter()
            if args.metric > 0 :
                if args.metric > 1 :
                    loss, psnr, ssim, fsim, lpips, niqe, musiq, nrqm = self.val_iter(final=False)
                else:
                    loss, psnr, ssim, fsim, lpips = self.val_iter(final=False)
                    fid, niqe, musiq, nrqm = 0.0, 0.0, 0.0, 0.0
            else:
                loss, psnr, ssim = self.val_iter(final=False)
                fsim, lpips, fid, niqe, musiq, nrqm = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            is_best = self.best_loss is None or loss < self.best_loss          
            if is_best:
                self.best_loss = loss
                self.best_psnr = psnr
                self.best_ssim = ssim
                self.best_fsim = fsim
                self.best_lpips = lpips
                self.best_fid = fid
                self.best_niqe = niqe
                self.best_musiq = musiq
                self.best_nrqm = nrqm
                self.best_epoch = self.epoch
                t.set_postfix(best=self.epoch, PSNR=f'{psnr:.2f}',
                              SSIM=f'{ssim:.4f}', FSIM=f'{fsim:.4f}',
                              LPIPS=f'{lpips:.4f}', FID=f'{fid:.4f}',
                              NIQE=f'{niqe:.4f}', MUSIQ=f'{musiq:.4f}',
                              NRQM=f'{nrqm:.4f}', loss=f'{loss:.4f}')
            self.save_checkpoint(best=is_best)
            t.update(1)          
            self.scheduler.step()

        ### SHKIM higher
        while self.epoch < args.epochs:
            self.epoch += 1
            self.train_iter_mwn()
            # self.train_iter_warpi()
            # self.train_iter_lre_higher()
            # self.train_iter_lre()
            if args.metric > 0 :
                if args.metric > 1 :
                    loss, psnr, ssim, fsim, lpips, niqe, musiq, nrqm = self.val_iter(final=False)
                else:
                    loss, psnr, ssim, fsim, lpips = self.val_iter(final=False)
                    fid, niqe, musiq, nrqm = 0.0, 0.0, 0.0, 0.0
            else:
                loss, psnr, ssim = self.val_iter(final=False)
                fsim, lpips, fid, niqe, musiq, nrqm = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0    
            
            is_best = self.best_loss is None or loss < self.best_loss          
            if is_best:
                self.best_loss = loss
                self.best_psnr = psnr
                self.best_ssim = ssim
                self.best_fsim = fsim
                self.best_lpips = lpips
                self.best_fid = fid
                self.best_niqe = niqe
                self.best_musiq = musiq
                self.best_nrqm = nrqm
                self.best_epoch = self.epoch
                t.set_postfix(best=self.epoch, PSNR=f'{psnr:.2f}',
                              SSIM=f'{ssim:.4f}', FSIM=f'{fsim:.4f}',
                              LPIPS=f'{lpips:.4f}', FID=f'{fid:.4f}',
                              NIQE=f'{niqe:.4f}', MUSIQ=f'{musiq:.4f}',
                              NRQM=f'{nrqm:.4f}', loss=f'{loss:.4f}')
            self.meta_save_checkpoint(best=is_best)
            t.update(1)          
            self.scheduler.step()
        
        LOG.info(str(t))
        

    def get_model_state_dict(self):
        # Ensures that the state_dict is on the CPU and reverse model transformations
        self.model.to('cpu')
        model = copy.deepcopy(self.model)
        self.model.to(self.device)
        if args.weight_norm:
            for m in model.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    m = nn.utils.remove_weight_norm(m)
        return model.state_dict()

    def load_checkpoint(self):
        if args.load_checkpoint is None:
            return
        ckp = torch.load(args.load_checkpoint)
        self.model.load_state_dict(ckp['state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(ckp['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckp['scheduler'])
        self.epoch = ckp['epoch']
        if 'best_epoch' in ckp:
            self.best_epoch = ckp['best_epoch']
        if 'best_psnr' in ckp:
            self.best_psnr = ckp['best_psnr']
        if 'best_ssim' in ckp:
            self.best_ssim = ckp['best_ssim']
        if 'best_fsim' in ckp:
            self.best_fsim = ckp['best_fsim']
        if 'best_lpips' in ckp:
            self.best_lpips = ckp['best_lpips']
        if 'best_fid' in ckp:
            self.best_fid = ckp['best_fid']
        if 'best_niqe' in ckp:
            self.best_niqe = ckp['best_niqe']
        if 'best_musiq' in ckp:
            self.best_musiq = ckp['best_musiq']
        if 'best_nrqm' in ckp:
            self.best_nrqm = ckp['best_nrqm']
        if 'best_loss' in ckp:
            self.best_loss = ckp['best_loss']

    def save_checkpoint(self, best=False):
        if args.save_checkpoint is None:
            return
        path = args.save_checkpoint
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_epoch': self.best_epoch,
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'best_fsim': self.best_fsim,
            'best_lpips': self.best_lpips,
            'best_fid': self.best_fid,
            'best_niqe': self.best_niqe,
            'best_musiq': self.best_musiq,
            'best_nrqm': self.best_nrqm,
            'best_loss': self.best_loss,
        }
        torch.save(state, path)
        base, ext = os.path.splitext(path)
        if args.save_every is not None and self.epoch % args.save_every == 0:
            torch.save(state, base + f"_e{self.epoch}" + ext)
        if best:
            torch.save(state, base + "_best" + ext)
            torch.save(self.get_model_state_dict(), base + "_best_model" + ext)

    def meta_save_checkpoint(self, best=False):
        if args.save_checkpoint is None:
            return
        path = args.save_checkpoint
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_epoch': self.best_epoch,
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'best_fsim': self.best_fsim,
            'best_lpips': self.best_lpips,
            'best_fid': self.best_fid,
            'best_niqe': self.best_niqe,
            'best_musiq': self.best_musiq,
            'best_nrqm': self.best_nrqm,
            'best_loss': self.best_loss,
        }
        torch.save(state, path)
        base, ext = os.path.splitext(path)
        if args.save_every is not None and self.epoch % args.save_every == 0:
            torch.save(state, base + f"_meta_e{self.epoch}" + ext)
        if best:
            torch.save(state, base + "_meta_best" + ext)
            torch.save(self.get_model_state_dict(), base + "_meta_best_model" + ext)

    def process_for_eval(self, img):
        if args.shave_border != 0:
            shave = args.shave_border
            img = img[..., shave:-shave, shave:-shave]
        if args.eval_luminance:
            img = to_luminance(img)
        elif args.scale_chroma is not None:
            img = to_YCbCr(img)
            chroma_scaling = torch.tensor([1.0, args.scale_chroma, args.scale_chroma])
            img = img * chroma_scaling.reshape(1, 3, 1, 1).to(img.device)
        return img


class LeakFinder:
    def __init__(self):
        self.step = 0
        self.batch = 0
        self.values = {}
        self.predict_every = 10
        self.verbose = True
    
    def set_batch(self, epoch):
        self.batch = epoch
        self.step = 0
        self.values[epoch] = {}

    def get_cuda_perc(self):
        perc = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        self.values[self.batch][self.step] = perc * 100

        self.step += 1

    def predict_leak_position(self, diffs, per_epoch_remainder):
    # train a tree regressor to predict the per epoch increase
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        from sklearn.preprocessing import MinMaxScaler

        # insert a zero at the start of  per_epoch_remainder
        per_epoch_remainder = torch.cat([torch.tensor([0]), per_epoch_remainder])

        # scale the data to be between 0 and 1
        x_scaler = MinMaxScaler()
        diffs = x_scaler.fit_transform(diffs)

        y_scaler = MinMaxScaler()
        per_epoch_remainder = y_scaler.fit_transform(per_epoch_remainder.reshape(-1, 1))

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(diffs, per_epoch_remainder, test_size=0.1, random_state=42)

        # train regressor
        regressor = DecisionTreeRegressor(random_state=0)
        regressor.fit(X_train, y_train)

        # predict
        y_pred = regressor.predict(X_test)

        # calculate error
        mse = mean_squared_error(y_test, y_pred)
        mag = mse / per_epoch_remainder.mean() * 100
        print(f"\nMSE: {mse} ({mag:.2f}%)")
#        LOG.info(f"\nMSE: {mse} ({mag:.2f}%)")

        # find the most important feature
        feature_importance = regressor.feature_importances_
        most_important_feature = torch.argmax(torch.tensor(feature_importance))
        print(f"\nLikely leak position between step {most_important_feature} and step {most_important_feature + 1}")
#        LOG.info(f"\nLikely leak position between step {most_important_feature} and step {most_important_feature + 1}")


    def find_leaks(self):
        if self.batch <2:
            return
        if self.verbose and self.batch % self.predict_every != 0:
            return

        diffs = []
        for epoch, values in self.values.items():
            dif = []
            for step in range(1, len(values)):
                dif += [values[step] - values[step - 1]]
            diffs.append(dif)

        lens = [len(x) for x in diffs]
        min_lens = min(lens)

        per_epoch_increase = [self.values[epoch][min_lens - 1] - self.values[epoch][0] for epoch in self.values.keys() if epoch > 0]
        between_epoch_decrease = [self.values[epoch][0] - self.values[epoch - 1][min_lens - 1] for epoch in self.values.keys() if epoch > 0]
        per_epoch_increase = torch.tensor(per_epoch_increase)
        between_epoch_decrease = torch.tensor(between_epoch_decrease)

        per_epoch_remainder = per_epoch_increase + between_epoch_decrease

        per_epoch_increase_mean = per_epoch_remainder.mean()
        per_epoch_increase_sum = per_epoch_remainder.sum()

        diffs = torch.tensor(diffs)

        print(
            f"\nPer epoch increase: {per_epoch_increase_mean:.2f}% cuda memory "
            f"(total increase of {per_epoch_increase_sum:.2f}%) currently at "
            f"{self.values[self.batch][min_lens - 1]:.2f}% cuda memory")
#        LOG.info(
#            f"\nPer epoch increase: {per_epoch_increase_mean:.2f}% cuda memory "
#            f"(total increase of {per_epoch_increase_sum:.2f}%) currently at "
#            f"{self.values[self.batch][min_lens - 1]:.2f}% cuda memory")

        if self.batch % self.predict_every == 0:
            self.predict_leak_position(diffs, per_epoch_remainder)



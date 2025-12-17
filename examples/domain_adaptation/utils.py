"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import os.path as osp
import time
import random
import numpy as np
from PIL import Image

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Subset


from collections import defaultdict
from torch.nn.modules.batchnorm import _BatchNorm
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform

sys.path.append('../../..')
import tllib.vision.datasets as datasets
import tllib.vision.models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.vision.datasets.imagelist import MultipleDomainsDataset
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_robust_model(model_name, norm='linf', eps=1.0):
    if model_name in models.__dict__:
        backbone = models.__dict__[model_name]()
        # load robust pretrained model
        checkpoint = torch.load(f'/home/ydy0415/data/pretrained/robust/{model_name}_{norm}_eps{eps}.ckpt', map_location='cpu')
        sd= checkpoint["model"]
        sd = {k[len('module.model.'):]:v for k,v in sd.items()}
        backbone.load_state_dict(sd, strict=False)
        
        #print(backbone)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone



def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + ['Digits']


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name == "Digits":
        train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), download=True,
                                                            transform=train_source_transform)
        train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                            transform=train_target_transform)
        val_dataset = test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                                  download=True, transform=val_transform)
        class_names = datasets.MNIST.get_classes()
        num_classes = len(class_names)
    elif dataset_name in datasets.__dict__:
        # load datasets from tllib.vision.datasets
        dataset = datasets.__dict__[dataset_name]

        def concat_dataset(tasks, start_idx, **kwargs):
            # return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])
            return MultipleDomainsDataset([dataset(task=task, **kwargs) for task in tasks], tasks,
                                          domain_ids=list(range(start_idx, start_idx + len(tasks))))

        train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform,
                                              start_idx=0)
        train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform,
                                              start_idx=len(source))
        val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform,
                                     start_idx=len(source))

        if dataset_name == 'DomainNet':
            test_dataset = concat_dataset(root=root, tasks=target, split='test', download=True, transform=val_transform,
                                          start_idx=len(source))
        else:
            test_dataset = val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)
    else:
        raise NotImplementedError(dataset_name)
    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    normalizer = InputNormalize(torch.tensor([0.485, 0.456, 0.406]).to(device),
                                torch.tensor([0.229, 0.224, 0.225]).to(device))
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')
    
    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)
            
            images = normalizer(images)
                
            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

    return top1.avg



def validate_rob(val_loader, model, attack, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    normalizer = InputNormalize(torch.tensor([0.485, 0.456, 0.406]).cuda(),
                                torch.tensor([0.229, 0.224, 0.225]).cuda())
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')
    

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    
    end = time.time()
    for i, data in enumerate(val_loader):
        images, target = data[:2]
        images = images.to(device)
        target = target.to(device)
        
        images_adv, _ = attack.perturb(images, target)
        images_adv = normalizer(images_adv)
        # compute output
        output = model(images_adv)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, = accuracy(output, target, topk=(1,))
        if confmat:
            confmat.update(target, output.argmax(1))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    print(' *Rob-Acc@1 {top1.avg:.3f}'.format(top1=top1))
    if confmat:
        print(confmat.format(args.class_names))

    return top1.avg


def validate_aa(val_loader, model, autoattack, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    normalizer = InputNormalize(torch.tensor([0.485, 0.456, 0.406]).cuda(),
                                torch.tensor([0.229, 0.224, 0.225]).cuda())
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='AA Test: ')
    

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    
    end = time.time()
    for i, data in enumerate(val_loader):
        images, target = data[:2]
        images = images.to(device)
        target = target.to(device)
        
        #images_adv, _ = attack.perturb(images, target)
        images_adv = autoattack.run_standard_evaluation(images, target, bs=args.batch_size)
        #print(images_adv==images)# all true
        #print((images_adv - images)[0].max()) = 0
        #print((images_adv-images).max())
        images_adv = normalizer(images_adv)
        
        # compute output
        output = model(images_adv)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, = accuracy(output, target, topk=(1,))
        if confmat:
            confmat.update(target, output.argmax(1))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    print(' *AA-Acc@1 {top1.avg:.3f}'.format(top1=top1))
    if confmat:
        print(confmat.format(args.class_names))

    return top1.avg


def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    transformed_img_size = 224
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224, scale=scale, ratio=ratio)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    elif random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        #T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        #T.Normalize(mean=norm_mean, std=norm_std)
    ])


def empirical_risk_minimization(train_source_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)



class PGD_Linf():

    def __init__(self, model, epsilon=8/255, step_size=2/255, num_steps=10,
                 random_start=True, target_mode= False, criterion= 'ce', normalized=False, train=True):

        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.target_mode = target_mode
        self.train = train
        self.criterion = criterion
        self.normalized = normalized
        self.normalizer = InputNormalize(torch.tensor([0.485, 0.456, 0.406]).cuda(), torch.tensor([0.229, 0.224, 0.225]).cuda())
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCELoss()
        self.criterion_kl = nn.KLDivLoss(reduction='sum')

    def perturb(self, x_nat, targets=None):
        
        self.model.eval()
            
        if self.random_start:
            x_adv = x_nat.detach() + torch.empty_like(x_nat).uniform_(-self.epsilon, self.epsilon).cuda().detach()
            x_adv = torch.clamp(x_adv, min=0, max=1)
        else:
            x_adv = x_nat.clone().detach()

        for _ in range(self.num_steps):
            x_adv.requires_grad_()
            #self.model.zero_grad()
            if self.normalized:
                outputs_adv = self.model(self.normalizer(x_adv))
                if self.criterion == "kl":
                    outputs_nat = self.model(self.normalizer(x_nat))
            else:
                outputs_adv = self.model(x_adv)
            if self.criterion == "ce":
                loss = self.criterion_ce(outputs_adv, targets)
                grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "kl":
                loss = self.criterion_kl(F.log_softmax(outputs_adv, dim=1), F.softmax(outputs_nat), dim = 1)
                grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "bce":
                loss = self.criterion_bce(outputs_adv, targets)
                grad = torch.autograd.grad(loss, [x_adv])[0]
                
            if self.target_mode:
                x_adv = x_adv - self.step_size * grad.sign()
            else:
                x_adv = x_adv + self.step_size * grad.sign()
            
            x_adv = torch.min(torch.max(x_adv, x_nat - self.epsilon), x_nat + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            d_adv = x_adv - x_nat
            
        if self.train:
            self.model.train()
        
        return x_adv, d_adv
    
class PGD_Linf_Div():

    def __init__(self, model, epsilon=8/255, step_size=2/255, num_steps=10,
                 random_start=True, target_mode= False, normalized=False):

        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.target_mode = target_mode
        self.normalized = normalized
        self.normalizer = InputNormalize(torch.tensor([0.485, 0.456, 0.406]).cuda(), torch.tensor([0.229, 0.224, 0.225]).cuda())
        self.criterion_ce = nn.CrossEntropyLoss()
        self.domain_discri = DomainDiscriminator(in_feature=self.model.features_dim, hidden_size=1024).cuda()
        self.domain_adv_loss = DomainAdversarialLoss(self.domain_discri).cuda()

    def perturb(self, x_s, x_t, targets=None):
        
        self.model.train()
            
        if self.random_start:
            #x_s_adv = x_source.detach() + torch.empty_like(x_source).uniform_(-self.epsilon, self.epsilon).cuda().detach()
            #x_s_adv = torch.clamp(x_s_adv, min=0, max=1)

            x_t_adv = x_t.detach() + torch.empty_like(x_t).uniform_(-self.epsilon, self.epsilon).cuda().detach()
            x_t_adv = torch.clamp(x_t_adv, min=0, max=1)
        else:
            #x_s_adv = x_source.clone().detach()
            x_t_adv = x_t.clone().detach()

        for _ in range(self.num_steps):
            x_t_adv.requires_grad_()
            #self.model.zero_grad()
            if self.normalized:
                x = torch.cat((x_s, x_t_adv), dim=0)
                y, f = self.model(self.normalizer(x))

                _, y_t = y.chunk(2, dim=0)
                f_s, f_t = f.chunk(2, dim=0)

            else:
                x = torch.cat((x_s, x_t_adv), dim=0)
                y, f = self.model(x)

                _, y_t = y.chunk(2, dim=0)
                f_s, f_t = f.chunk(2, dim=0)

            transfer_loss = self.domain_adv_loss(f_s, f_t)

            loss = self.criterion_ce(y_t, targets) + transfer_loss
            grad = torch.autograd.grad(loss, [x_t_adv])[0]

            if self.target_mode:
                x_t_adv = x_t_adv - self.step_size * grad.sign()
            else:
                x_t_adv = x_t_adv + self.step_size * grad.sign()
            
            x_t_adv = torch.min(torch.max(x_t_adv, x_t - self.epsilon), x_t + self.epsilon)
            x_t_adv = torch.clamp(x_t_adv, 0.0, 1.0)
            d_adv = x_t_adv - x_t
        
        return x_t_adv, d_adv

class InputNormalize(torch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
        

def disable_running_stats(model):
    r"""Disable running stats (momentum) of BatchNorm."""

    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)
    
def enable_running_stats(model):
    r"""Enable running stats (momentum) of BatchNorm."""

    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, 'backup_momentum'):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class LipschitzEstimator:
    def __init__(self, model, epsilon=0.01, n_steps=10, step_size_factor=4, device=None):
        """
        :param model: PyTorch model whose Lipschitz constant we want to estimate
        :param epsilon: Perturbation radius (ε)
        :param n_steps: Number of steps for the PGD-like procedure
        :param step_size_factor: Factor to divide epsilon for step size
        """
        self.model = model.eval()
        self.epsilon = epsilon
        self.n_steps = n_steps
        self.step_size = epsilon / step_size_factor
        self.device = device
        self.normalizer = InputNormalize(torch.tensor([0.485, 0.456, 0.406]).to(self.device),
                                torch.tensor([0.229, 0.224, 0.225]).to(self.device))


    def estimate_local_lipschitz(self, x):
        """
        Estimate the local Lipschitz constant for a batch of inputs x.
        
        :param x: Input tensor (batch of samples)


        :return: Estimated local Lipschitz constant for each sample in the batch
        """
        batch_size = x.size(0)  # Number of samples in the batch
        
        # Ensure the input requires gradients for perturbation
        x = x.clone().detach().requires_grad_(True)
        
        # Get the original output for input x
        original_output = self.model(self.normalizer(x))

        perturbed_x = x.detach() + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon).cuda().detach()
        # Perform PGD-like iterative updates to find worst-case perturbation
        for _ in range(self.n_steps):
            # Compute the gradient of the loss w.r.t. perturbed_x
            #self.model.zero_grad()
            # Initialize perturbed input as a copy of the original input
            perturbed_x = perturbed_x.requires_grad_(True)
            perturbed_output = self.model(self.normalizer(perturbed_x))
            
            # Use L1 norm for the difference in outputs (as per the formula)
            loss = torch.norm(perturbed_output - original_output, p=1)
            grad = torch.autograd.grad(loss, [perturbed_x])[0]

            # Update perturbed_x in the direction of the gradient (gradient ascent)
            
            perturbed_x = perturbed_x+ self.step_size * grad.sign()
                
            # Project back into the ε-ball (L∞ norm constraint)
            perturbed_x = torch.min(torch.max(perturbed_x, x - self.epsilon), x + self.epsilon).detach()
            
            # Zero out gradients for next iteration
            #perturbed_x.grad.zero_()

        # Compute final output difference and input difference after perturbation
        final_output_diff = torch.norm(self.model(self.normalizer(perturbed_x)) - original_output, p=1, dim=1) # L1 norm across batch
        #input_diff = torch.norm(perturbed_x - x, p=float('inf'), dim=1)  # L∞ norm across batch
        input_diff = torch.norm(perturbed_x - x, p=float('inf'), dim=(1,2,3))  # L∞ norm across batch
        
        #breakpoint()
        # Compute and return the estimated local Lipschitz constants for each sample in the batch
        lipschitz_constants = final_output_diff / input_diff

        return lipschitz_constants

    def estimate_batch_lipschitz(self, dataloader):
        """
        Estimate the average local Lipschitz constant over a dataset.
        
        :param dataloader: Dataloader providing batches of samples
        :return: Average estimated Lipschitz constant across all batches and samples
        """
        lipschitz_values = []
        
        for batch_idx, data in enumerate(dataloader):
            images, _ = data[:2]
            images = images.to(self.device)
            lipschitz_constants = self.estimate_local_lipschitz(images)
            lipschitz_values.extend(lipschitz_constants.detach().cpu().numpy())  # Collect results from each batch
        
        return sum(lipschitz_values) / len(lipschitz_values)
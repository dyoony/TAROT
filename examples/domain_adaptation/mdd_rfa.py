"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import time
import warnings
import argparse
import os.path as osp
import shutil
import copy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
from tllib.alignment.mdd import ClassificationMarginDisparityDiscrepancy \
    as MarginDisparityDiscrepancy, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

from autoattack import AutoAttack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=25, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    
    backbone = utils.get_robust_model(args.arch, norm=args.pre_norm, eps=args.pre_eps)
        
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 width=args.bottleneck_dim, pool_layer=pool_layer).to(device)    

    mdd = MarginDisparityDiscrepancy(args.margin).to(device)

    attack_val = utils.PGD_Linf(classifier, epsilon=args.eps, step_size=args.step_size, num_steps=args.val_numsteps, random_start=True,
                                target_mode=False, criterion='ce', normalized=True, train=False)
    # define optimizer and lr_scheduler
    # The learning rate of the classiﬁers are set 10 times to that of the feature extractor by default.
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        #rob_acc1 = utils.validate_rob(test_loader, classifier, attack_val, args, device)
        
        auto_attack = AutoAttack(classifier, norm='Linf', eps=args.eps, version='standard', verbose=False)
        #auto_attack.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab', 'square']
        auto_attack.attacks_to_run = ['apgd-ce', 'apgd-t']
        auto_attack.apgd.n_restarts = 1
        auto_attack.fab.n_restarts = 1
        
        aa_acc1 = utils.validate_aa(test_loader, classifier, auto_attack, args, device)
        
        print("test_acc1 = {:3.1f}".format(acc1))
        print("aa_acc1 = {:3.1f}".format(aa_acc1))
        
        return

    # start training
    best_rob_acc1 = 0.
    rob_acc1 = 0.
    # Teacher #################################################################################################
    classifier_teacher = copy.deepcopy(classifier)
    ###########################################################################################################

    # Frozen backbone teacher #################################################################################
    for param in classifier_teacher.parameters():
        param.requires_grad = False
    ###########################################################################################################

    for epoch in range(args.epochs):
        start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()  # epoch 시작 시 peak memory 초기화
        print("lr:", lr_scheduler.get_last_lr()[0])

        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, classifier_teacher, mdd, optimizer,
              lr_scheduler, epoch, args)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # GPU 메모리 사용량 확인
        if torch.cuda.is_available():
            current_mem = torch.cuda.memory_allocated() / (1024**2)   # MB 단위
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB 단위
            print(f"[Epoch {epoch}] Time: {elapsed_time:.2f}s | GPU Memory: {current_mem:.2f}MB (Peak: {peak_mem:.2f}MB)")
        else:
            print(f"[Epoch {epoch}] Time: {elapsed_time:.2f}s (CPU only)")



        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)
        if (epoch+1) % args.rob_eval == 0 or (epoch+1) == args.epochs:
            rob_acc1 = utils.validate_rob(val_loader, classifier, attack_val, args, device)
        
        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if rob_acc1 > best_rob_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_rob_acc1 = max(rob_acc1, best_rob_acc1)
            best_acc1 = acc1

    print("best_acc1 = {:3.1f}".format(best_acc1))
    print("best_rob_acc1 = {:3.1f}".format(best_rob_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    rob_acc1 = utils.validate_rob(val_loader, classifier, attack_val, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))
    print("test_rob_acc1 = {:3.1f}".format(rob_acc1))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          classifier: ImageClassifier, classifier_teacher: ImageClassifier, mdd: MarginDisparityDiscrepancy, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    normalizer = utils.InputNormalize(torch.tensor(args.norm_mean).to(device),
                                      torch.tensor(args.norm_std).to(device))

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    classifier.train()
    mdd.train()

    classifier_teacher.eval()

    # Hook Registeration ######################################################################################
    layer_activation = {}
    layer_activation_t = {}
    
    def getActivation(name):
        def hook(module, input, output):
            layer_activation[name] = output
        return hook
    
    def getActivation_t(name):
        def hook(module, input, output):
            layer_activation_t[name] = output.detach()
        return hook

    l1 = classifier.backbone.layer1.register_forward_hook(getActivation('layer1'))
    l2 = classifier.backbone.layer2.register_forward_hook(getActivation('layer2'))
    l3 = classifier.backbone.layer3.register_forward_hook(getActivation('layer3'))
    l4 = classifier.backbone.layer4.register_forward_hook(getActivation('layer4'))

    l1_t = classifier_teacher.backbone.layer1.register_forward_hook(getActivation_t('layer1'))
    l2_t = classifier_teacher.backbone.layer2.register_forward_hook(getActivation_t('layer2'))
    l3_t = classifier_teacher.backbone.layer3.register_forward_hook(getActivation_t('layer3'))
    l4_t = classifier_teacher.backbone.layer4.register_forward_hook(getActivation_t('layer4'))   
    ###########################################################################################################

    end = time.time()
    for i in range(args.iters_per_epoch):
        optimizer.zero_grad()

        
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        
        normalizer = normalizer.to(device)
        x_s = normalizer(x_s)
        x_t = normalizer(x_t)
        
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        outputs, outputs_adv = classifier(x)
        y_s, y_t = outputs.chunk(2, dim=0)
        y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

        # teacher forward ####################################################################################
        y_teacher = classifier_teacher(x)
        ###################################################################################################### 

        # compute cross entropy loss on source domain
        cls_loss = F.cross_entropy(y_s, labels_s)
        # compute margin disparity discrepancy between domains
        # for adversarial classifier, minimize negative mdd is equal to maximize mdd
        transfer_loss = -mdd(y_s, y_s_adv, y_t, y_t_adv)

         # RFA loss ###########################################################################################
        rfa_loss = 0.
        for layer_name, _ in layer_activation.items():
            q_teacher = torch.reshape(layer_activation_t[layer_name], (args.batch_size*2,-1))
            q_student = torch.reshape(layer_activation[layer_name], (args.batch_size*2,-1))

            qq_teacher = torch.matmul(q_teacher, q_teacher.transpose(0,1))
            qq_teacher_norm_2 = qq_teacher / torch.norm(qq_teacher, 2, dim = 1)[:, None]
            qq_student = torch.matmul(q_student, q_student.transpose(0,1))
            qq_student_norm_2 = qq_student / torch.norm(qq_student, 2, dim = 1)[:, None]

            rfa_loss += (torch.norm(qq_teacher_norm_2 - qq_student_norm_2, "fro")**2)/(args.batch_size**2)
        #######################################################################################################

        alpha = 2000
        loss = cls_loss + transfer_loss * args.trade_off + rfa_loss*alpha
        classifier.step()

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            #print('rfa_loss : {}'.format(rfa_loss))

    l1.remove()
    l2.remove()
    l3.remove()
    l4.remove()
    l1_t.remove()
    l2_t.remove()
    l3_t.remove()
    l4_t.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDD for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true', help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+', default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+', default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=512, type=int)
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--robust', action='store_false', help='whether train from adversarially robust model.')
    parser.add_argument('--pre_eps', default=None, type=float, help='perturbation budget of pretrained adversarially robust model (pre-epsilon)')
    parser.add_argument('--pre_norm', default='linf', type=str, help='norm of epsilon of pretrained adversarially robust model')
    parser.add_argument('--margin', type=float, default=4., help="margin gamma")
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    
    # robustness parameter
    parser.add_argument('--eps', default=8/255, type=float,
                        help='the budget of perturbation')
    parser.add_argument('--step_size', default=2/255, type=float,
                        help='the step size for PGD')
    parser.add_argument('--val_numsteps', default=20, type=int,
                        help='the number of steps for PGD')
    
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.004, type=float,
                        
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.002, type=float)
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--rob-eval', default=5, type=int,
                        help='Number of warm-up epoch')
    parser.add_argument('-p', '--print-freq', default=333, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='gpu id')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='mdd',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)

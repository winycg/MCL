import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np


import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds, DistillKL, correct_num, adjust_lr, AverageMeter


from bisect import bisect_right
import time
import math

from losses.imagenet_sup_mcl_loss import Sup_MCL_Loss
from dataset.imagenet import get_imagenet_dataloader_sample


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='./data/ImageNet/', type=str, help='dataset directory')
parser.add_argument('--dataset', default='ImageNet', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet18', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[30, 60, 90], type=int, nargs='+', help='milestones for lr-multistep')
parser.add_argument('--epochs', type=int, default=90, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--num-workers', type=int, default=16, help='number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resume-checkpoint', default='./checkpoint/resnet32.pth.tar', type=str, help='resume checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--evaluate-checkpoint', default='./checkpoint/resnet32_best.pth.tar', type=str, help='evaluate checkpoint')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')

parser.add_argument('--number-net', type=int, default=2, help='number of networks')
parser.add_argument('--logit-distill', action='store_true', help='combine with logit distillation')

parser.add_argument('--kd_T', type=float, default=3, help='temperature of KL-divergence')
parser.add_argument('--alpha', type=float, default=0., help='weight balance for VCL')
parser.add_argument('--gamma', type=float, default=0., help='weight balance for Soft VCL')
parser.add_argument('--beta', type=float, default=0., help='weight balance for ICL')
parser.add_argument('--lam', type=float, default=0., help='weight balance for Soft ICL')

parser.add_argument('--rep-dim', default=1024, type=int, help='penultimate dimension')
parser.add_argument('--feat-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--pos_k', default=1, type=int, help='number of positive samples for NCE')
parser.add_argument('--neg_k', default=8192, type=int, help='number of negative samples for NCE')
parser.add_argument('--tau', default=0.07, type=float, help='temperature parameter for softmax')
parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
if not os.path.isdir('./result/'):
    os.makedirs('./result/')

log_txt = 'result/'+ str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'arch'+args.arch + '_'+\
          'dataset' + '_' +  args.dataset + '_'+\
          'seed'+ str(args.manual_seed) +'.txt'


np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# -----------------------------------------------------------------------------------------


num_classes = 1000

if args.evaluate is False:
    n_data, trainloader, testloader = get_imagenet_dataloader_sample(data_folder=args.data,
                                                            args=args,
                                                            is_sample=True)
else:
    testloader = get_imagenet_dataloader_sample(data_folder=args.data,
                                                            args=args,
                                                            is_sample=True)

# --------------------------------------------------------------------------------------------

print('==> Building model..')
model = getattr(models, args.arch)
net = model(num_classes=num_classes, number_net=args.number_net)
net.eval()
print('Params: %.2fM, Multi-adds: %.2fG'
      % (cal_param_size(net)/1e6, cal_multi_adds(net, (1, 3, 224, 224))/1e9))
del(net)

net = model(num_classes=num_classes, number_net=args.number_net).cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = True


# Training
def train(epoch, criterion_list, optimizer):
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_logit_kd = AverageMeter('train_loss_logit_kd', ':.4e')
    train_loss_vcl = AverageMeter('train_loss_vcl', ':.4e')
    train_loss_icl = AverageMeter('train_loss_icl', ':.4e')
    train_loss_soft_vcl = AverageMeter('train_loss_soft_vcl', ':.4e')
    train_loss_soft_icl = AverageMeter('train_loss_soft_icl', ':.4e')

    correct_1 = [0] * (args.number_net + 1)
    correct_5 = [0] * (args.number_net + 1)
    total = [0] * (args.number_net + 1)

    lr = adjust_lr(optimizer, epoch, args)
    start_time = time.time()
    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_mcl = criterion_list[2]

    net.train()
    batch_start_time = time.time()
    for batch_idx, (inputs, targets, pos_idx, neg_idx) in enumerate(trainloader):
        inputs = inputs.float().cuda()
        targets = targets.cuda()
        pos_idx = pos_idx.cuda().long()
        neg_idx = neg_idx.cuda().long()

        optimizer.zero_grad()
        logits, embeddings = net(inputs)

        loss_cls = torch.tensor(0.).cuda()
        loss_logit_kd = torch.tensor(0.).cuda()

        ensemble_logits = 0.
        for i in range(len(logits)):
            loss_cls = loss_cls + criterion_ce(logits[i], targets)
        for i in range(len(logits)):
            ensemble_logits = ensemble_logits + logits[i]
        ensemble_logits = ensemble_logits / (len(logits))

        if args.logit_distill:
            loss_logit_kd = loss_logit_kd + criterion_div(logits[-1], ensemble_logits)

        loss_vcl, loss_soft_vcl, loss_icl, loss_soft_icl = criterion_mcl(embeddings, pos_idx, neg_idx)
        loss_mcl = args.alpha * loss_vcl + args.gamma * loss_soft_vcl \
                    + args.beta * loss_icl + args.lam * loss_soft_icl

        loss = loss_cls + loss_logit_kd + loss_mcl
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
        train_loss_cls.update(loss_cls.item(), inputs.size(0))
        train_loss_logit_kd.update(loss_logit_kd.item(), inputs.size(0))
        train_loss_vcl.update(args.alpha * loss_vcl.item(), inputs.size(0))
        train_loss_soft_vcl.update(args.gamma * loss_soft_vcl.item(), inputs.size(0))
        train_loss_icl.update(args.beta * loss_icl.item(), inputs.size(0))
        train_loss_soft_icl.update(args.lam * loss_soft_icl.item(), inputs.size(0))

        for i in range(args.number_net + 1):
            if i == args.number_net:
                prec1, prec5 = correct_num(ensemble_logits, targets, topk=(1, 5))
            else:
                prec1, prec5 = correct_num(logits[i], targets, topk=(1, 5))
            correct_1[i] += prec1.item()
            correct_5[i] += prec5.item()
            total[i] += targets.size(0)

        print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
            epoch, batch_idx, len(trainloader), lr, time.time()-batch_start_time, (correct_1[0]/total[0])))

        batch_start_time = time.time()


    acc1 = [round((correct_1[i]/total[i]), 4) for i in range(args.number_net+1)]
    acc5 = [round((correct_5[i]/total[i]), 4) for i in range(args.number_net+1)]

    with open(log_txt, 'a+') as f:
        f.write('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
                '\n Train_loss:{:.5f}'
                '\t Train_loss_cls:{:.5f}'
                '\t Train_loss_logit_kd:{:.5f}'
                '\t Train_loss_vcl:{:.5f}'
                '\t Train_loss_soft_vcl:{:.5f}'
                '\t Train_loss_icl:{:.5f}'
                '\t Train_loss_soft_icl:{:.5f}'
                '\nTrain top-1 accuracy: {}\nTrain top-5 accuracy: {}\n'
                .format(epoch, lr, time.time() - start_time,
                        train_loss.avg,
                        train_loss_cls.avg,
                        train_loss_logit_kd.avg,
                        train_loss_vcl.avg,
                        train_loss_soft_vcl.avg,
                        train_loss_icl.avg,
                        train_loss_soft_icl.avg,
                        str(acc1), str(acc5)))


def test(epoch, criterion_ce):
    net.eval()
    global best_acc
    test_loss_cls = AverageMeter('test_loss_cls', ':.4e')

    correct_1 = [0] * (args.number_net + 1)
    correct_5 = [0] * (args.number_net + 1)
    total = [0] * (args.number_net + 1)

    with torch.no_grad():
        batch_start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, embedding = net(inputs)

            loss_cls = 0.
            ensemble_logits = 0.
            for i in range(len(logits)):
                loss_cls = loss_cls + criterion_ce(logits[i], targets)
            for i in range(len(logits)):
                ensemble_logits = ensemble_logits + logits[i]
            ensemble_logits = ensemble_logits / (len(logits))
            ensemble_logits = ensemble_logits.detach()

            test_loss_cls.update(loss_cls, inputs.size(0))

            print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}'.format(epoch, batch_idx, len(testloader), time.time()-batch_start_time))
            batch_start_time = time.time()

            for i in range(args.number_net + 1):
                if i == args.number_net:
                    prec1, prec5 = correct_num(ensemble_logits, targets, topk=(1, 5))
                else:
                    prec1, prec5 = correct_num(logits[i], targets, topk=(1, 5))
                correct_1[i] += prec1.item()
                correct_5[i] += prec5.item()
                total[i] += targets.size(0)

        acc1 = [round((correct_1[i]/total[i]), 4) for i in range(args.number_net+1)]
        acc5 = [round((correct_5[i]/total[i]), 4) for i in range(args.number_net+1)]

        with open(log_txt, 'a+') as f:
            f.write('Test epoch:{}\t Test_loss_cls:{:.5f}\t Test top-1 accuracy:{}\nTest top-5 accuracy:{}\n'
                    .format(epoch, test_loss_cls.avg, str(acc1), str(acc5)))

        print('Test epoch:{}\t Test top-1 accuracy:{}\n'.format(epoch, str(acc1)))

    return max(acc1[:-1])


if __name__ == '__main__':
    criterion_ce = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)

    if args.evaluate: 
        print('Evaluate pre-trained weights from: {}'.format(args.evaluate_checkpoint))
        checkpoint = torch.load(args.evaluate_checkpoint, map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        test(start_epoch, criterion_ce)
    else:
        trainable_list = nn.ModuleList([])
        trainable_list.append(net)

        data = torch.randn(1, 3, 224, 224)
        net.eval()
        logits, embeddings = net(data)

        args.rep_dim = embeddings[0].shape[1]
        args.n_data = n_data

        criterion_mcl = Sup_MCL_Loss(args)
        trainable_list.append(criterion_mcl.embed_list)
        optimizer = optim.SGD(trainable_list.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_ce)  # classification loss
        criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
        criterion_list.append(criterion_mcl)  # other knowledge distillation loss
        criterion_list.cuda()

        if args.resume: 
            print('Resume pre-trained weights from: {}'.format(args.resume_checkpoint))
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_ce)

            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))
            best_acc = -1
            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

        print('Evaluate the best model:')
        print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_ce)

        with open(log_txt, 'a+') as f:
            f.write('Test top-1 best_accuracy: {} \n'.format(top1_acc))
        print('Test top-1 best_accuracy: {} \n'.format(top1_acc))

import argparse
import os
import shutil
import time
import sys
import sklearn
import sklearn.metrics
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


from AlexNet import *
from voc_dataset import *
from utils import *

import wandb
USE_WANDB = True # use flags, wandb is not convenient for debugging


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    args.pretrained=True
    args.batch_size = 32
    args.epochs = 2
    args.lr = 0.01

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    # you can also use PlateauLR scheduler, which usually works well
    
    step_size = 30
    gamma = 0.1

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # Data loading code
    
    #TODO: Create Datasets and Dataloaders using VOCDataset - Ensure that the sizes are as required
    # Also ensure that data directories are correct - the ones use for testing by TAs might be different
    # Resize the images to 512x512
    inp_size = 512
    top_n = 30
    train_dataset = VOCDataset('trainval', image_size=inp_size, top_n=top_n)
    val_dataset = VOCDataset('test', image_size=inp_size, top_n=top_n)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # shuffle=(train_sampler is None),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    
    
    # TODO: Create loggers for wandb - ideally, use flags since wandb makes it harder to debug code.
    # import wandb
    if USE_WANDB:
        wandb.init(project="vlr-hw2")

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        
        if scheduler is not None:
            scheduler.step()
        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)




#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):   # i is batch
        # measure data loading time
        data_time.update(time.time() - end)

        # TODO: Get inputs from the data dict
        img = data['image'].to('cuda')
        _, _, height, width = img.shape
        target = data['label'].to('cuda')
        # wgt = data['wgt']
        # rois = data['rois']
        # gt_boxes = data['gt_boxes']
        # gt_classes = data['gt_classes']
        
        optimizer.zero_grad()

        # TODO: Get output from model
        imoutput = model(img)
        n, c, h, w = imoutput.shape
        # TODO: Perform any necessary functions on the output such as clamping
        
        output = nn.MaxPool2d(kernel_size=(h,w))(imoutput)
        output = torch.reshape(output, (n,c))
        output = torch.sigmoid(output)
        
        # TODO: Compute loss using ``criterion``
        loss = criterion(output, target)
        # measure metrics and record loss
        m1 = metric1(output, target)
        m2 = metric2(output, target)
        losses.update(loss.item(), img.size(0))
        # print(m1, m2)
        avg_m1.update(m1)
        avg_m2.update(m2)


        # TODO:
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        #TODO: Visualize/log things as mentioned in handout
            # logging the loss
        if USE_WANDB:
            wandb.log({'epoch': epoch, 'loss': loss.item(), 'mAP': avg_m1.avg, 'Recall': avg_m2.avg})

        #TODO: Visualize at appropriate intervals
        # for Q1.5
            if i==50 or i==100:
                idx = 10
                label = target[idx].nonzero()[0]
                heatmap = imoutput[idx,label,:,:]
                heatmap = nn.Upsample(size=(height,width),mode='nearest')(heatmap.view(1,1,h,w)).view(height,width)
                
                original_image = tensor_to_PIL(img[idx,:].cpu().detach())
                hm = heatmap.cpu().detach().numpy()
                img_hm = plt.imshow(hm, cmap='viridis')
                img_orig = wandb.Image(original_image)
                wandb.log({"image": img_orig})
                wandb.log({"image with heatmap": img_hm})
        # End of train()


def validate(val_loader, model, criterion, epoch = 0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    end = time.time()
    for i, (data) in enumerate(val_loader):

        # TODO: Get inputs from the data dict
        img = data['image'].to('cuda')
        _, _, height, width = img.shape
        target = data['label'].to('cuda')
        # wgt = data['wgt']
        # rois = data['rois']
        # gt_boxes = data['gt_boxes']
        # gt_classes = data['gt_classes']
        # img, target, wgt = img.to('cuda'), target.to('cuda'), wgt.to('cuda')


        # TODO: Get output from model
        imoutput = model(img)
        n, c, h, w = imoutput.shape
        
        # TODO: Perform any necessary functions on the output
        n, c, h, w = imoutput.shape
        output = nn.MaxPool2d(kernel_size=(h,w))(imoutput)
        output = torch.reshape(output, (n,c))

        output = torch.sigmoid(output)
        # TODO: Compute loss using ``criterion``
        loss = criterion(output, target)


        # measure metrics and record loss
        # m1 = metric1(imoutput.data, target)
        # m2 = metric2(imoutput.data, target)
        m1 = metric1(output, target)
        m2 = metric2(output, target)
        losses.update(loss.item(), img.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        #TODO: Visualize/log things as mentioned in handout
            # logging the loss
        if USE_WANDB:
            wandb.log({'validate/epoch': epoch, 'validate/mAP': avg_m1.avg, 'validate/Recall': avg_m2.avg})

        #TODO: Visualize at appropriate intervals
        # if i==50 or i==100:
        #     idx = 20
        #     label = target[idx].nonzero()[0]
        #     heatmap = imoutput[idx,label,:,:]
        #     heatmap = nn.Upsample(size=(height,width),mode='nearest')(heatmap.view(1,1,h,w)).view(height,width)
            
        #     original_image = tensor_to_PIL(img[idx,:].cpu().detach())
        #     hm = heatmap.cpu().detach().numpy()
        #     img_hm = plt.imshow(hm, cmap='jet')
        #     # B = A.unsqueeze(1).repeat(1, K, 1)
        #     # hm = tensor_to_PIL(hm.cpu().detach())
        #     img_orig = wandb.Image(original_image)
        #     # img_hm = wandb.Image(hm)
        #     wandb.log({"validate/image": img_orig})
        #     wandb.log({"validate/image with heatmap": img_hm})


    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    target = target.cpu().detach().numpy()
    output = output.cpu().detach().numpy()
    nclasses = target.shape[1]    
    AP = []
    for cid in range(nclasses):
        gt_cls = target[:, cid].astype('float32')
        pred_cls = output[:, cid].astype('float32')
        pred_cls -= 1e-5 * gt_cls
        is_all_zero = np.all((gt_cls == 0))
        if not is_all_zero:
            ap = sklearn.metrics.average_precision_score(
                gt_cls, pred_cls)
            AP.append(ap)
    mAP = np.mean(AP)

    return mAP


def metric2(output, target):
    #TODO: Ignore for now - proceed till instructed
    target = np.argmax(target.cpu().detach().numpy(), axis=1)
    output = np.argmax(output.cpu().detach().numpy(), axis=1)
    recall = sklearn.metrics.recall_score(target, output, average='micro')

    return recall


if __name__ == '__main__':
    main()

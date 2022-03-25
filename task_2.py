from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL, iou
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
import matplotlib.patches as patches

USE_WANDB = True # use flags, wandb is not convenient for debugging

def drawBoxes(plot_boxes, plot_labels, plot_scores, img_box):    # boxes: N x 4 (x1, y1, x2, y2)
    color = (1,0,0)
    # img_box = plt.figure()
    for z in range(len(plot_boxes)):
        box = plot_boxes[z]
        cls_label = plot_labels[z]
        scores = plot_scores[z]
        for i, (_, x1, y1, x2, y2) in enumerate(box):
            x = x1*512
            y = y1*512
            w = (x2-x1)*512
            h = (y2-y1)*512
            # print((x, y), w, h)
            rect = patches.Rectangle((x, y), w, h, facecolor='none', edgecolor=color, linewidth=2)
            plt.gca().add_patch(rect)
            img_box = plt.gca().text(x, y - 2, '%s %.2f' % (cls_label, scores[i]), bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    return img_box

# hyper-parameters
# ------------
start_step = 0
end_step = 20000
lr_decay_steps = {150000}
lr_decay = 1. / 10
rand_seed = 1024

lr = 0.0001
momentum = 0.9
weight_decay = 0.0005
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load datasets and create dataloaders
inp_size = 512
top_n = 300
train_dataset = VOCDataset('trainval', image_size=inp_size, top_n=top_n)
val_dataset = VOCDataset('test', image_size=inp_size, top_n=top_n)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,   # batchsize is one for this implementation
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    sampler=None,
    drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True)


# Create network and initialize
net = WSDDN(classes=train_dataset.CLASS_NAMES)
print(net)

if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
else:
    pret_net = model_zoo.load_url(
        'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
             pkl.HIGHEST_PROTOCOL)
own_state = net.state_dict()

for name, param in pret_net.items():
    print(name)
    if name not in own_state:
        continue
    if isinstance(param, Parameter):
        param = param.data
    try:
        own_state[name].copy_(param)
        print('Copied {}'.format(name))
    except:
        print('Did not find {}'.format(name))
        continue


# Move model to GPU and set train mode
net.load_state_dict(own_state)
net.cuda()
net.train()

# TODO: Create optimizer for network parameters from conv2 onwards
# (do not optimize conv1)
for i,child in enumerate(net.children()):
    # print(child)
    for k,child_0 in enumerate(child.children()):
        if k<1 and i<1:
            for params in child_0.parameters():
                # print(params)
                params.requires_grad=False
            print("Frozen {} layer {}".format(k,child_0))
            break

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=end_step, gamma=lr_decay)

if USE_WANDB:
    wandb.init(project="vlr-hw2")


output_dir = "../task_2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
disp_interval = 10
val_interval = 500
epochs = 5
print_freq = 500


def test_net(model, val_loader=None, thresh=0.05, plot=False):
    """
    tests the networks and visualize the detections
    thresh is the confidence threshold
    """
    model.eval()
    mAP = 0
    img_count = 0
    iou_threshold = 0.3
    ap = np.zeros(20)
    for iter, data in enumerate(val_loader):
        img_count += 1

        fp = 0
        tp = 0
        plot_boxes = []
        plot_scores = []
        plot_labels = []
        # one batch = data for one image
        image           = data['image']
        target          = data['label']
        wgt             = data['wgt']
        rois            = data['rois']
        gt_boxes        = data['gt_boxes']
        gt_class_list   = data['gt_classes']

        gt_detected = np.zeros(len(gt_boxes), dtype=np.int8)

        #TODO: perform forward pass, compute cls_probs
        rois = rois.view(rois.size()[1:])
        rois = torch.cat((torch.zeros(rois.size()[0],1), rois),dim=1)
        imoutput = net(image.to('cuda'), rois.type('torch.cuda.FloatTensor'), target.type('torch.cuda.FloatTensor'))

        # TODO: Iterate over each class (follow comments)
        for class_num in range(20):            
            # get valid rois and cls_scores based on thresh
            class_scores = imoutput[:, class_num].cpu().detach().numpy()
            filtered_idxes = np.where(class_scores > thresh)[0]
            if len(filtered_idxes) <= 0:
                continue
            class_scores = class_scores[filtered_idxes]
            selected_rois = rois[filtered_idxes, :]
            # use NMS to get boxes and scores
            boxes, scores = nms(selected_rois, class_scores, threshold=0.4)

            #TODO: visualize bounding box predictions when required
            if plot:
                plot_boxes.append(boxes)
                plot_scores.append(scores)
                plot_labels.append(VOCDataset.CLASS_NAMES[class_num])
            

            #TODO: Calculate mAP on test set
            for j in range(len(scores)):
                    selected_roi = boxes[j]
                    max_iou = 0
                    detected_gt_idx = -1

                    # Looking for its best matched ground_truth
                    for k, gt_bbox in enumerate(gt_boxes):
                        if gt_class_list[k] == class_num:
                            gt_roi = gt_bbox
                            IoU = iou(selected_roi, gt_roi)
                            if IoU > max_iou:
                                max_iou = IoU
                                detected_gt_idx = k
                    if max_iou > iou_threshold:
                        tp += 1
                        gt_detected[detected_gt_idx] = 1    # Mark matched ground truth to calculate FN
                    else:
                        fp += 1

            fn = len(gt_detected) - np.sum(gt_detected)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            avg_p = recall * precision
            # mAP += ap
            ap[class_num] += avg_p
            # print('img: %s ap: %f' % (iter, ap))
        # print('img: ', iter, 'ap: ', ap)
        if ((plot) and (iter == 0 or iter == 12 or iter == 20 or iter == 200 or iter == 480 or iter == 1000 or iter == 2017 or iter == 2200 or iter == 2750 or iter == 1200 or iter == 1029 or iter == 2990 or iter == 3110 or iter == 4800 or iter == 3000 or iter == 3952 or iter == 4000 or iter == 4333 or iter == 4500 or iter == 4400)):
                img_box = plt.figure()
                img_box = plt.imshow(image.permute(2,3,1,0).view(512,512,3).cpu().detach().numpy(), vmin=0, vmax=1)              
                img_box = drawBoxes(plot_boxes, plot_labels, plot_scores, img_box)
                # plt.show()
                if USE_WANDB:
                    wandb.log({"Image with Bounding Box {}".format(iter): img_box})
    ap /= img_count
    if USE_WANDB:
        wandb.log({'ap[0] {}'.format(VOCDataset.CLASS_NAMES[0]): ap[0], 'ap[1] {}'.format(VOCDataset.CLASS_NAMES[1]): ap[1], 'ap[2] {}'.format(VOCDataset.CLASS_NAMES[2]): ap[2], 'ap[3]{}'.format(VOCDataset.CLASS_NAMES[3]): ap[3], 'ap[4]{}'.format(VOCDataset.CLASS_NAMES[4]): ap[4], 
        'ap[5] {}'.format(VOCDataset.CLASS_NAMES[5]): ap[5], 'ap[6]{}'.format(VOCDataset.CLASS_NAMES[6]): ap[6], 'ap[7]{}'.format(VOCDataset.CLASS_NAMES[7]): ap[7], 'ap[8] {}'.format(VOCDataset.CLASS_NAMES[8]): ap[8], 'ap[9] {}'.format(VOCDataset.CLASS_NAMES[9]): ap[9],
        'ap[10] {}'.format(VOCDataset.CLASS_NAMES[10]): ap[10], 'ap[11] {}'.format(VOCDataset.CLASS_NAMES[0]): ap[11], 'ap[12] {}'.format(VOCDataset.CLASS_NAMES[0]): ap[12], 'ap[13] {}'.format(VOCDataset.CLASS_NAMES[13]): ap[13], 'ap[14] {}'.format(VOCDataset.CLASS_NAMES[14]): ap[14], 'ap[15] {}'.format(VOCDataset.CLASS_NAMES[0]): ap[15], 
        'ap[16] {}'.format(VOCDataset.CLASS_NAMES[16]): ap[16], 'ap[17] {}'.format(VOCDataset.CLASS_NAMES[17]): ap[17], 'ap[18] {}'.format(VOCDataset.CLASS_NAMES[18]): ap[18], 'ap[19] {}'.format(VOCDataset.CLASS_NAMES[19]): ap[19]})
    mAP = np.mean(ap)
    # print('img: ', img_count, 'ap: ', ap)
    return mAP



for epoch in range(epochs):
    net.train()

    for iter, data in enumerate(train_loader):

        #TODO: get one batch and perform forward pass
        # one batch = data for one image
        image           = data['image']
        target          = data['label']
        wgt             = data['wgt']
        rois            = data['rois']
        gt_boxes        = data['gt_boxes']
        gt_class_list   = data['gt_classes']
        

        #TODO: perform forward pass - take care that proposal values should be in pixels for the fwd pass
        # also convert inputs to cuda if training on GPU
        optimizer.zero_grad()
        rois = rois.view(rois.size()[1:])
        rois = torch.cat((torch.zeros(rois.size()[0],1), rois),dim=1)
        imoutput = net(image.to('cuda'), rois.type('torch.cuda.FloatTensor'), target.type('torch.cuda.FloatTensor'))

        # backward pass and update
        loss = net.loss    
        train_loss += loss.item()
        step_cnt += 1

        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #TODO: evaluate the model every N iterations (N defined in handout)
        
        if iter%val_interval == 0 and iter != 0:
            net.eval()
            # ap = test_net(net, val_loader, plot=True)
            if(iter==5000 and (epoch == 0 or epoch == 4)):
                mAP = test_net(net, val_loader, plot=True)
            else:
                mAP = test_net(net, val_loader)
            print("mAP ", mAP)
            if USE_WANDB:
                wandb.log({'Epoch': epoch, 'mAP': mAP})
            net.train()

        if iter % print_freq == 0 and iter != 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                 'Loss {loss:.4f}'.format(
                      epoch,
                      iter,
                      len(train_loader),
                      loss=train_loss/print_freq))
            
            if USE_WANDB:
                wandb.log({'train/Epoch': epoch, 'train/Loss': train_loss/print_freq})
            train_loss=0


        #TODO: Perform all visualizations here
        #The intervals for different things are defined in the handout

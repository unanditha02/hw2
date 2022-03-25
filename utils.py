import os
import random
import time
import copy

import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np


#TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bounding_boxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    boxes = []
    scores = []
    while (len(bounding_boxes)!= 0):
        
        max_conf_i = np.argmax(confidence_score)
        best_conf_score = confidence_score[max_conf_i]
        best_conf_box = bounding_boxes[max_conf_i]
        if(type(bounding_boxes)==torch.Tensor):
            bounding_boxes = bounding_boxes.cpu().detach().numpy()

        bounding_boxes = np.delete(bounding_boxes, max_conf_i, 0)
        confidence_score = np.delete(confidence_score, max_conf_i, 0)
        boxes.append(best_conf_box)
        scores.append(best_conf_score)

        for i, box in enumerate(bounding_boxes):
            if(iou(best_conf_box, box) > 0.3):
                if(type(bounding_boxes)==torch.Tensor):
                    bounding_boxes = bounding_boxes.cpu().detach().numpy()
                i_remove = np.argwhere((bounding_boxes==box).all(axis=1))[0][0]
                bounding_boxes = np.delete(bounding_boxes, i_remove, 0)
                confidence_score = np.delete(confidence_score, i_remove, 0)

    return boxes, scores

#TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    xx = max(box1[0], box2[0])
    yy = max(box1[1], box2[1])
    aa = min(box1[2], box2[2])
    bb = min(box1[3], box2[3])

    w = max(0, aa - xx)
    h = max(0, bb - yy)

    intersection_area = w*h
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / union_area
    return iou


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
            "class_id" : classes[i],
        } for i in range(len(classes))
        ]

    return box_list





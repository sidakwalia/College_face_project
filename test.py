import os
import os
import random
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# xml library for parsing xml files
from xml.etree import ElementTree as et

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch



if __name__=='__main__':

    annot_file_path = 'WIDER_train_annotations/61_Street_Battle_streetfight_61_793.xml'
    classes =  ['_ ','face']
    boxes = []
    labels = []
    tree = et.parse(annot_file_path)
    root = tree.getroot()
    print(root)
    for member in root.findall('object'):
        xmin = int(member.find('bndbox').find('xmin').text)
        xmax = int(member.find('bndbox').find('xmax').text)
        ymin = int(member.find('bndbox').find('ymin').text)
        ymax = int(member.find('bndbox').find('ymax').text)


    # cv2 image gives size as height x width
    # wt = img.shape[1]
    # ht = img.shape[0]

    # box coordinates for xml files are extracted and corrected for image size given
    # for member in root.findall('object'):
    #     labels.append(classes.index(member.find('name').text))
    #
    #     # bounding box
    #     xmin = int(member.find('bndbox').find('xmin').text)
    #     xmax = int(member.find('bndbox').find('xmax').text)
    #
    #     ymin = int(member.find('bndbox').find('ymin').text)
    #     ymax = int(member.find('bndbox').find('ymax').text)
    #
    #     xmin_corr = (xmin / wt) * self.width
    #     xmax_corr = (xmax / wt) * self.width
    #     ymin_corr = (ymin / ht) * self.height
    #     ymax_corr = (ymax / ht) * self.height
    #
    #     boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
    #
    # # convert boxes into a torch.Tensor
    # boxes = torch.as_tensor(boxes, dtype=torch.float32)
    #
    # # getting the areas of the boxes
    # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    #
    # # suppose all instances are not crowd
    # iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
    #
    # labels = torch.as_tensor(labels, dtype=torch.int64)

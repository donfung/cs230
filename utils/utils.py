import numpy as np
from shapely.geometry import Polygon
import time
import scipy
import math
import cv2

def load_bbox(input_file='../milestone_progress/sample_results/Venice-2/Venice-2.txt', dataset='otb'):
  data_input = []
  
  if dataset == 'otb':
    n_values = 4
  elif dataset == 'mot':
    n_values = 6
  try:
    with open(input_file) as f:
        for line in f:
            values = line.split(',')
            #data_input.append(a[:])
            data_input.append(list(map(float, values[0:n_values])))
    bbox = np.asarray(data_input).T  
    return bbox
  except:
    print('Could not load file: {}'.format(input_file))



def create_poly(box):
    x1, y1, x3, y3 = box
    return [(x1, y1),(x1, y3), (x3, y3), (x3, y1), (x1, y1)]
    
def calculate_iou(polygon1,polygon2):
    polygon1 = Polygon(polygon1)
    polygon2 = Polygon(polygon2)
    polygon1 = polygon1.buffer(0)
    polygon2 = polygon2.buffer(0)
    intersection = polygon1.intersection(polygon2)
    union = polygon1.area + polygon2.area - intersection.area
    return (intersection.area / union)


def iou_calc(bbox, gtruth):
    iou = calculate_iou(create_poly(bbox), create_poly(gtruth))
    return iou

def normalize_bbox(bbox, img_dir):
    image = cv2.imread(img_dir)
    height, width = image.shape[:2]
    norm_bbox = bbox
    norm_bbox[0,:] = bbox[0,:]/width
    norm_bbox[1,:] = bbox[1,:]/height
    norm_bbox[2,:] = bbox[2,:]/width
    norm_bbox[3,:] = bbox[3,:]/height

    return norm_bbox




import numpy as np
import scipy
import math
from shapely.geometry import Polygon
import time

# Loads input data
data_input = []
with open("/Users/prerna/Documents/MastersQ2/CS230/cs230/milestone_progress/sample_results/Venice-2/Venice-2.txt") as f:
    for line in f:
        a = line.split(',')
        data_input.append(list(map(float, a[0:6])))

#Loads the ground truth file
data_gt = []        
with open("/Users/prerna/Documents/MastersQ2/CS230/cs230/milestone_progress/sample_results/Venice-2/gt.txt") as f:
    for line in f:
        a = line.split(',')
        data_gt.append(list(map(float, a[0:6])))
        
# Helper function for IOU calculation
def create_poly(box):
    x1, y1, x3, y3 = box
    return [(x1, y1),(x1, y3), (x3, y3), (x3, y1), (x1, y1)]
    
# Helper function for IOU calculation
def calculate_iou(polygon1,polygon2):
    polygon1 = Polygon(polygon1)
    polygon2 = Polygon(polygon2)
    polygon1 = polygon1.buffer(0)
    polygon2 = polygon2.buffer(0)
    intersection = polygon1.intersection(polygon2)
    union = polygon1.area + polygon2.area - intersection.area
    return (intersection.area / union)

# Calculates IOU for two bounding boxes
def iou_calc(bbox, gtruth):
#     Returns iou value 
    iou = calculate_iou(create_poly(bbox), create_poly(gtruth))
    return (iou)
       
# Converting list to array       
gt_np = np.array(data_gt)
bb = np.array(data_input)

# Removing the object ID column since its just -1s
bb = np.delete(bb,1,1)
frames = np.unique(gt_np[:,0])

start=time.time()
mix = np.zeros(11)
for i in bb:
    poss = gt_np[gt_np[:,0] == i[0]]
    max = float("-inf")
    k = []
    for j in poss:
        iouV= iou_calc(i[1:5],j[2:6])
        if iouV>=0.5 and iouV>=max:
            max = iouV 
            k = j
    if k != []:
        jo = np.hstack((k,i[1:5]))
        mix=np.vstack((mix,jo))
elapsed=time.time()-start       
idx = np.argsort(mix[:,1], kind = "stable")
gt = mix[idx]



        
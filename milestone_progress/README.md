This folder has our preliminary codes (trials and functional) for the project milestone.

Project : 
Use a combination of a CNN and an RNN network to improve bounding box stability in object tracking. 

Our primary tasks involve :
1) Detection - Identify various object classes, and tag them accurately. This is the first step in object tracking and our initial idea was to simply use the best available classifiers to do this for us. In this milestone, we did this with YOLO. THe classifier provides per frame, the coordinates and position of a bounding box that identifies an object in that frame. 

2) Tracker - Our eventual goal is to try and come up with an RNN network that uses information from prior frames to predict the location of the next bounding box for a tracked object on a frame. Since all three of us have only recently been exposed to this new type of network, we first decided to approach the problem by trying to see what exisiting literature is there regarding this topic. We found a very simple and novel approach (SORT) based on Kalman Filters and decided to use this as a starting point in our project. For our milestone, we tried reproducing the results observed in this approach by learning and building up code to do this. 

Benchmarks: 
We used the 2015 MOT benchmark for our purpose, and this code takes a video input and throws out a video output which has the detected bounding boxes for the objects in the video & a table of values in text form that mimics the output requirements of the MOT benchmark. 
The detection algorithm was YOLO trained on the COCO data set (we plan to try and train this later with the MOT set itself, but for now this works with some post-processing in the code to limit our results to human detection (as implemented in the original SORT paper) 


Scripts

milestone.ipynb is the main python notebook for the script. It may not be strightforward to run this right away because the path locations are all set based on a local version. but we have highlighted where paths need to be changed and if this is done, the code will produce the results listed below. 

models.py, sort.py, utils_.py all implement helper functions and the sort alogrithm to run the entire project.

The model folder has the pretrained YOLO weights, configuration etc. stored in it (again, you'll need to refer correctly to this directory in the code to execute the project). The utils folder has utility functions to help YOLO run successfully.  

Results: 

1. We have a folder (results) which has a sample output on the venice video set (both the ground truths & the text file our code generates) 

2. We also have a videos which show bounding boxes for the processed data (not included in the github repository) 

3. We have roughly emulated the performance of the SORT algorithm paper (in the references) to help us have an idea of what is a really good metric to try and beat (the SORT algorithm was one of the best open source algorithms for object tracking in the MOT benchmark) 

Our Results: 
                   Rcll   Prcn    GT   MT  PT    ML   FP    FN  IDs   FM   MOTA   MOTP  
TUD-Campus        73.5%   82.2%    8   4    4    0    57    95   0    26   57.7%  0.269  
ETH-Sunnyday      84.1%   71.8%   30  16   12    2   613   295   0   107   51.1%  0.273  
Venice-2          44.1%   58.7%    1   0    1    0  2258  4069   0   599   13.1%  0.320  
ETH-Pedcross2     58.2%   79.2%  133  33   56   44   958  2621   0   281   42.9%  0.265  
ADL-Rundle-8      50.2%   65.5%   28   9   16    3  1796  3376   0   531   23.8%  0.307  
KITTI-17          73.4%   84.6%    9   2    7    0    91   182   0    39   60.0%  0.290  

Note: MOTP should be calculated as (1-MOTP)*100 because of the implementation of the benchmark script in python  

These are very close to the benchmarks reported in the SORT paper:  

Sequence	    Rcll	Prcn	FAR	   GT  MT   PT  ML	FP    FN     IDs  FM	MOTA MOTP    
TUD-Campus	  68.5	94.3	0.21	  8   6    2   0	15    113     6    9	62.7 73.7   
ETH-Sunnyday	77.5	81.9	0.90	 30  11   16   3	319   418    22   54	59.1 74.4   
ETH-Pedcross2	51.9	90.8	0.39	133  17   60  56	330  3014    77   103	45.4 74.8   
ADL-Rundle-8	44.3	75.8	1.47	 28   6   16   6	959  3781   103   211	28.6 71.1   
Venice-2	    42.5	64.8	2.75	 26   7    9  10	1650 4109    57   106	18.6 73.4   
KITTI-17	    67.1	92.3	0.26	  9   1    8   0	38    225     9    16	60.2 72.3   




Issues: 

1. Many bugs in the YOLO implementation which are being currently worked on -> Tasked with handing over the CNN output (bounding box information, classes & ids) which will serve as the input to the RNN network (classifier) 

2. Have a good understanding of RNNs and working to replace the sort() algorithm section with an RNN network. 
    a. The data set for this will again be the MOT2015 data -> and the set of ground truths from the current frame will serve as the input to help predict the next frame.
    b. Have to decide on the depth and layers for this network and are currently looking at vanilla RNN architectures to figure out what would be the best approach
    
3. Our tasks going forward 
  1. Work towards retraining the CNN half on the MOT data set 
  2. Implement an RNN network to replace sort() and train it on the MOT dataset 
  3. Try to improve performance on the MOTA and MOTP benchmark 
 

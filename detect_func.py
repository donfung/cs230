import argparse
import shutil
import time
from pathlib import Path
from sys import platform

from models import *
from utils.datasets import *
from utils.utils2 import *


def detect(
        cfg,
        weights,
        images,
        output='test',  # output folder
        img_size=416,
        conf_thres=0.3,
        nms_thres=0.5,
        save_txt=True,
        save_images=False,
        webcam=False
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()

    # Set Dataloader
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg('cfg/coco_orig.data')['names'])
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]
    
    bbox_array = [0,0,0,0,0,0]
    for i, (path, img, im0) in enumerate(dataloader):
        t = time.time()
        if webcam:
            print('webcam frame %g: ' % (i + 1), end='')
        else:
            print('image %g/%g %s: ' % (i + 1, len(dataloader), path), end='')
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        #pred = model(img)
        pred, a_val = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold
    
        if len(pred) > 0:
            # Run NMS on predictions
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]
            # print(detections)
            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()
            # print(detections)
            # Print results to screen
            unique_classes = detections[:, -1].cpu().unique()
            #print (unique_classes.data[0])
            for c in unique_classes:
                #print(detections[:,-1].cpu().data[0])
                n = (detections[:, -1].cpu() == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                if (cls==0):
                    new_bbox_array = np.asarray([x1.cpu().detach().numpy(),y1.cpu().detach().numpy(),
                                                x2.cpu().detach().numpy(),y2.cpu().detach().numpy(),
                                                cls.cpu().detach().numpy(),
                                                cls_conf.cpu().detach().numpy()*conf.cpu().detach().numpy()])
                    bbox_array = np.vstack([bbox_array,new_bbox_array])
                    #bbox_array = np.vstack([bbox_array,[x1,y1,x2,y2,cls,cls_conf*conf]])
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write('%g %g %g %g %g %g\n' %
                                       (x1, y1, x2, y2, cls, cls_conf * conf))
                # Add bbox to the image
                if (cls==0):
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[int(cls)])
        
        bbox_array = np.delete(bbox_array, 0, 0)
        
        dt = time.time() - t
        print('Done. (%.3fs)' % dt)

        if save_images:  # Save generated image with detections
            cv2.imwrite(save_path, im0)

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)
        
    if save_images and (platform == 'darwin'):  # linux/macos
        os.system('open ' + output + ' ' + save_path)
    
    return a_val,bbox_array

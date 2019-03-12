import numpy as np

# Reference: https://arxiv.org/pdf/1611.06467.pdf

def center_pos_error(bbox_pred_traj, bbox_gt_traj):
    # EQUATION (3) https://arxiv.org/pdf/1611.06467.pdf

    assert bbox_pred_traj.shape == bbox_gt_traj.shape, "Predicted and ground truth size mismatch: {} =/ {}".format(bbox_pred_traj.shape, bbox_gt_traj.shape)
    
    x_pred = bbox[0,:] # OR bbox[:,0] depending on format
    y_pred = bbox[1,:] 
    b_pred = bbox_pred[2,:]
    h_pred = bbox_pred[3,:]

    x_gt = bbox_gt[0,:]
    y_gt = bbox_gt[1,:]
    b_gt = bbox_gt[2,:]
    h_gt = bbox_gt[3,:]
    x_cg_pred = (x_pred + b_pred)/2
    y_cg_pred = (y_pred + h_pred)/2
    x_cg_gt = (x_gt + b_gt)/2
    y_cg_gt = (y_gt + h_gt)/2

    x_error = (x_cg_pred - x_cg_gt)/b_pred
    y_error = (y_cg_pred - y_cg_gt)/h_pred

    std_x = np.std(x_error)
    std_y = np.std(y_error)

    cp_error = std_x + std_y 
  
    return cp_error


def scale_ratio_error(bbox_pred_traj, bbox_gt_traj):
    # EQUATION (4) https://arxiv.org/pdf/1611.06467.pdf

    assert bbox_pred_traj.shape == bbox_gt_traj.shape, "Predicted and ground truth size mismatch: {} =/ {}".format(bbox_pred_traj.shape, bbox_gt_traj.shape)
    x_pred = bbox_pred[0,:]
    y_pred = bbox_pred[1,:]
    b_pred = bbox_pred[2,:]
    h_pred = bbox_pred[3,:]

    x_gt = bbox_gt[0,:]
    y_gt = bbox_gt[1,:]
    b_gt = bbox_gt[2,:]
    h_gt = bbox_gt[3,:]
    
    scale_error = np.sqrt(b_pred*h_pred/(b_gt*h_gt))
    ratio_error = (b_pred/h_pred)/(b_gt/h_gt)
    
    
    sigma_s = np.std(scale_error)
    sigma_r = np.std(ratio_error)
    
    sr_error = (sigma_s + sigma_r) 
    
    return sr_error

def fragment_error(bbox_pred_traj):

    ## WE MIGHT NOT USE THIS. BUT INCLUDING FOR COMPLETENESS 

    # EQUATION (2) https://arxiv.org/pdf/1611.06467.pdf
    traj_length = bbox_pred_traj.shape[0] # OR [1] depending on format. This is the length of the trajectory 
    num_drops = traj_length - bbox_pred_traj.count_nonzero() # THIS ONE ASSUMES THAT BBOX 'DROPS' ARE IDENTIFIED WHEN BBOX VALUES ARE ALL 0... NEEDS TO BE THOUGHT OF BETTER

    frag_error = num_drops/(traj_length-1)
    return frag_error

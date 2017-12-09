import torch
import os

global config
config = {}

# GPU to run on
config['device'] = 0 # 0,1,2 or 3

# IMAGES
config['image1_path'] = os.path.join('Images', "felix2.png")
config['image2_path'] = os.path.join('Images', "david2.png")

# SAVING
config['save_NNFs'] = True
config['save_FeatureMaps'] = True

# VGG
config['img_size'] = 224  # 56, 112, 224, 448
config['pool_mode'] = 'avg' # 'avg' or 'max'

# PATCH-MATCH
config['patch_size'] = {1:5, 2:5, 3:3, 4:3, 5:3}
config['n_iter'] = 4
config['distance_mode'] = 'bidirectional' # 'unidirectional' or 'bidirectional'
config['random_search_max_step'] = {1:4, 2:4, 3:6, 4:6, 5:14} 

# RECONSTRUCTION
config['n_iter_deconv'] = 3000
config['optimizer'] = torch.optim.Adamax # torch.optim.Adamax, torch.optim.LBFGS
config['loss_fct'] = torch.nn.MSELoss # torch.nn.L1Loss, torch.nn.MSELoss
config['alphas'] = {1:0.1, 2:0.6, 3:0.7, 4:0.8}
config['upsampling_ON'] = {1:True, 2:True, 3:True, 4:True, 5:False}

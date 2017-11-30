import torch
import os

global config
config = {}

# GPU to run on
config['device'] = 3 # 0,1,2 or 3
config['image1_path'] = os.path.join('Images', "avatar.png")
config['image2_path'] = os.path.join('Images', "joconde.png")

# VGG
config['img_size'] = 224  # 64, 128, 256, 512
config['pool_mode'] = 'avg' # 'avg' or 'max'

# PATCH-MATCH
config['patch_size'] = {1:5, 2:5, 3:3, 4:3, 5:3} # For PatchMatch
config['number_of_patches_per_zone'] = 1 # For randomSearch in PatchMatch
config['n_iter'] = 3 # For PatchMatch
config['distance_mode'] = 'bidirectional' # 'unidirectional' or 'bidirectional'
config['propagation_mode'] = 'NoNM' # 'NoNM'(Neighbor of Neighbor's Match) or 'NM'(Neighbor's Match)
config['random_search_max_step'] = {1:4, 2:4, 3:6, 4:6, 5:14} 

# RECONSTRUCTION
config['n_iter_deconv'] = 3000 # For Reconstruction
config['optimizer'] = torch.optim.Adamax # torch.optim.Adamax, torch.optim.LBFGS
config['loss_fct'] = torch.nn.MSELoss # torch.nn.L1Loss, torch.nn.MSELoss
config['alphas'] = {1:0.1, 2:0.6, 3:0.7, 4:0.8, 5:1.} # For reconstruction blending
config['upsampling_ON'] = {1:True, 2:True, 3:True, 4:True, 5:False}

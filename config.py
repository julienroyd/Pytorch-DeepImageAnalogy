import torch

global config
config = {}

# VGG
config['img_size'] = 256  # 64, 128, 256, 512
config['pool_mode'] = 'avg' # 'avg' or 'max'

# PATCH-MATCH
config['patch_size'] = 3 # For PatchMatch
config['number_of_patches_per_zone'] = 1 # For randomSearch in PatchMatch
config['n_iter'] = 3 # For PatchMatch
config['distance_mode'] = 'bidirectional' # 'unidirectional' or 'bidirectional'

# RECONSTRUCTION
config['n_iter_deconv'] = 4500 # For Reconstruction
config['optimizer'] = torch.optim.Adamax # torch.optim.Adamax, torch.optim.LBFGS
config['loss_fct'] = torch.nn.L1Loss # torch.nn.L1Loss, torch.nn.MSELoss
config['alphas'] = [0.1, 0.6, 0.7, 0.8, 1.] # For reconstruction blending
config['upsampling_ON'] = {1:True, 2:False, 3:False, 4:False, 5:False}

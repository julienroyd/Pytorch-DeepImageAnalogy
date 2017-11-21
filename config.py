global config
config = {}
config['img_size'] = 256  # 64, 128, 256, 512

config['patch_size'] = 3 # For PatchMatch
config['number_of_patches_per_zone'] = 1 # For randomSearch in PatchMatch
config['n_iter'] = 2 # For PatchMatch

config['alphas'] = [0.1, 0.6, 0.7, 0.8, 1.] # For reconstruction blending
from config import config
import utils
import DeepReconstruction
import DeepPatchMatch
from DeepVGG import VGG19, prep

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_featureMaps(A1, B1, model):
    """ Feeds content and style images through VGG and get their feature maps. A1 and B1 are PIL images."""

    img_tensor_A1 = Variable(prep(A1).unsqueeze(0), requires_grad=False)
    img_tensor_B1 = Variable(prep(B1).unsqueeze(0), requires_grad=False)

    if torch.cuda.is_available():
        img_tensor_A1 = img_tensor_A1.cuda()
        img_tensor_B1 = img_tensor_B1.cuda()

    feat_ids = {1:1, 2:6, 3:11, 4:20, 5:29}

    FeatureMaps_A1 = {L : model.forward(img_tensor_A1)[feat_ids[L]] for L in feat_ids}
    FeatureMaps_B1 = {L : model.forward(img_tensor_B1)[feat_ids[L]] for L in feat_ids}
    
    return FeatureMaps_A1, FeatureMaps_B1


print("\n\nRunning on GPU? ", torch.cuda.is_available())

print("--CONFIGS--")
for k, v in zip(config.keys(), config.values()):
    print("{0} : {1}".format(k, v))
print("-----------")

# THE SCRIPT -----------------------------------------------

A1 = Image.open(config['image1_path']).convert("RGB")
B1 = Image.open(config['image2_path']).convert("RGB")

# Builds the model (on GPU if available)
vgg = VGG19()

# Get the feature maps for reference images
FeatureMaps_A1, FeatureMaps_B1 = get_featureMaps(A1, B1, model=vgg)

# Initialize latent images representations
FeatureMaps_A2 = {}
FeatureMaps_B2 = {}

FeatureMaps_A2[5] = FeatureMaps_A1[5]
FeatureMaps_B2[5] = FeatureMaps_B1[5]

# Other initializations
NNFs_ab = {}
NNFs_ba = {}

R_A2 = {}
R_B2 = {}

# MAIN LOOP
for L in range(5,0,-1):
    print("L = {0}".format(L))
    
    # Upsampling the NNF
    if config['upsampling_ON'][L]:
        print('Upsampling ON')
        featureMapSize = FeatureMaps_A1[L].size()[-1]
        # For the current layer, NNF is initialized to upsampled version of the resulting NNF of the previous layer
        initialNNF_ab = DeepReconstruction.upsample(NNFs_ab[L+1], size=featureMapSize, mode="nearest")
        initialNNF_ba = DeepReconstruction.upsample(NNFs_ba[L+1], size=featureMapSize, mode="nearest")
    
    else:
        print('Upsampling OFF')
        initialNNF_ab = None
        initialNNF_ba = None


    # NNF Search
    NNFs_ab[L] = DeepPatchMatch.computeNNF(FeatureMaps_A1[L], FeatureMaps_B2[L], 
                                           FeatureMaps_A2[L], FeatureMaps_B1[L], 
                                           L, config, initialNNF=initialNNF_ab)
    
    NNFs_ba[L] = DeepPatchMatch.computeNNF(FeatureMaps_B1[L], FeatureMaps_A2[L], 
                                           FeatureMaps_B2[L], FeatureMaps_A1[L], 
                                           L, config, initialNNF=initialNNF_ba)

    if L > 1:
        # Reconstruction for A2
        Warped_FeatureMaps_A2 = DeepPatchMatch.warp(FeatureMaps_B1[L], NNFs_ab[L])
        
        R_A2[L-1] = DeepReconstruction.deconv(model=vgg,
                                              target=Warped_FeatureMaps_A2, 
                                              noise_size=FeatureMaps_A1[L-1].size(), 
                                              layer=L, 
                                              loss=config['loss_fct'],
                                              opt=config['optimizer'],
                                              n_iters=config['n_iter_deconv'])
        
        W_A1 = DeepReconstruction.get_weight_map(FeatureMaps_A1[L-1], config["alphas"][L-1])

        FeatureMaps_A2[L-1] = DeepReconstruction.blend(FeatureMaps_A1[L-1], R_A2[L-1], W_A1)

        # Reconstruction for B2
        Warped_FeatureMaps_B2 = DeepPatchMatch.warp(FeatureMaps_A1[L], NNFs_ba[L])

        R_B2[L-1] = DeepReconstruction.deconv(model=vgg,
                                              target=Warped_FeatureMaps_B2, 
                                              noise_size=FeatureMaps_A1[L-1].size(), 
                                              layer=L, 
                                              loss=config['loss_fct'],
                                              opt=config['optimizer'], 
                                              n_iters=config['n_iter_deconv'])
        
        W_B1 = DeepReconstruction.get_weight_map(FeatureMaps_B1[L-1], config["alphas"][L-1])

        FeatureMaps_B2[L-1] = DeepReconstruction.blend(FeatureMaps_B1[L-1], R_B2[L-1], W_B1)

print("\n--------\n--------\nOut of the main loop!")

if not os.path.exists('Results'):

    os.mkdir('Results')

if config['save_NNFs']:
    # Saves the NNFs
    utils.saveNNFs(os.path.join('Results', 'NNFs_ab.pkl'), NNFs_ab)
    utils.saveNNFs(os.path.join('Results', 'NNFs_ba.pkl'), NNFs_ba)

if config['save_FeatureMaps']:
    # Saves the FeatureMaps (result of deconvolutions)
    utils.saveFeatureMaps(os.path.join('Results', 'featureMaps_A1.pkl'), FeatureMaps_A1)
    utils.saveFeatureMaps(os.path.join('Results', 'featureMaps_A2.pkl'), FeatureMaps_A2)
    utils.saveFeatureMaps(os.path.join('Results', 'featureMaps_B1.pkl'), FeatureMaps_B1)
    utils.saveFeatureMaps(os.path.join('Results', 'featureMaps_B2.pkl'), FeatureMaps_B2)

# Saves the rsulting figure
nnf_ab = np.transpose(NNFs_ab[1].numpy(), axes=(1,2,0))
nnf_ba = np.transpose(NNFs_ba[1].numpy(), axes=(1,2,0))

A1 = np.asarray(A1.resize((config['img_size'],config['img_size']), Image.ANTIALIAS))
B1 = np.asarray(B1.resize((config['img_size'],config['img_size']), Image.ANTIALIAS))

A2 = B1[nnf_ab[:,:,0], nnf_ab[:,:,1], :]
B2 = A1[nnf_ba[:,:,0], nnf_ba[:,:,1], :]

images = [A1, A2, B2, B1]
names = ['A1', 'A2', 'B2', 'B1']

# Displays the images
plt.figure(figsize=(16, 4))
for img, title, i in zip(images, names, list(range(4))):
    plt.subplot(1,4,i+1)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
plt.show()

figname = os.path.basename(config['image1_path']).split('.')[0] + '_' + os.path.basename(config['image2_path']).split('.')[0] + '.png'
plt.savefig(os.path.join('Results', figname), bbox_inches='tight')
print('saved Feature Maps in : {0}'.format(os.path.join('Results', figname)))

print("-----DONE!-----")

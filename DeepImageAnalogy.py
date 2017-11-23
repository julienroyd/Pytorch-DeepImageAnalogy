from config import config
import DeepReconstruction
import DeepPatchMatch
import DeepVGG
from DeepVGG import Layer, Net

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os

from PIL import Image
import matplotlib.pyplot as plt

def build_model(path):
    # Instanciates the model
    vgg = DeepVGG.VGG()

    # Load the parameters saved in vgg_conv.pth
    vgg.load_state_dict(torch.load(path))

    # Freezes the model
    for param in vgg.parameters():
        param.requires_grad = False

    # Put the model on GPU    
    if torch.cuda.is_available():
        vgg.cuda()
    
    return vgg


def get_featureMaps(A1, B1, model):
    """ Feeds content and style images through VGG and get their feature maps. A1 and B1 are PIL images."""
    # Create a list of images, ordered as [content_image, style_image]
    imgs = [A1, B1]

    # Preprocess the images, converts them to autograd.Variable and put on GPU if available
    imgs_torch = [DeepVGG.prep(img) for img in imgs]
    if torch.cuda.is_available():
        imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
    else:
        imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]

    # Feed the images through VGG, get their feature maps, and put them in dictionnaries for more clarity
    Layers = [1, 2, 3, 4, 5]

    FeatureMaps_A1 = {L:F for L,F in zip(Layers, model(imgs_torch[0], ['r11','r21','r31','r41','r51']))}
    FeatureMaps_B1 = {L:F for L,F in zip(Layers, model(imgs_torch[1], ['r11','r21','r31','r41','r51']))}
    
    return FeatureMaps_A1, FeatureMaps_B1


print("Running on GPU? ", torch.cuda.is_available())
# THE SCRIPT -----------------------------------------------

A1 = Image.open(os.path.join('Images', "avatar.png"))
B1 = Image.open(os.path.join('Images', "joconde.png"))

# Builds the model (on GPU if available)
vgg = build_model(path=os.path.join('Model', 'vgg_conv.pth'))

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

# Subnets for reconstruction
subnets = []
subnets += [Net([Layer(vgg.conv1_1)])]
subnets += [Net([Layer(vgg.conv1_2),
                 vgg.pool1,
                 Layer(vgg.conv2_1)])]
subnets += [Net([Layer(vgg.conv2_2),
                 vgg.pool2,
                 Layer(vgg.conv3_1)])]
subnets += [Net([Layer(vgg.conv3_2),
                 Layer(vgg.conv3_3),
                 Layer(vgg.conv3_4),
                 vgg.pool3,
                 Layer(vgg.conv4_1)])]
subnets += [Net([Layer(vgg.conv4_2),
                 Layer(vgg.conv4_3),
                 Layer(vgg.conv4_4),
                 vgg.pool4,
                 Layer(vgg.conv5_1)])]

# MAIN LOOP
for L in range(5,0,-1):
    print("L = {0}".format(L))
    
    # NNF search
    if True: # TODO : Reactivate Upsampling
        initialNNF_ab = None
        initialNNF_ba = None
    
    else:
        initialNNF_ab = DeepReconstruction.upsample(torch.unsqueeze(NNFs_ab[L+1], 0), FeatureMaps_A1[L].size()[-1], mode="nearest")
        initialNNF_ba = DeepReconstruction.upsample(torch.unsqueeze(NNFs_ba[L+1], 0), FeatureMaps_A1[L].size()[-1], mode="nearest")
        
        
    NNFs_ab[L] = DeepPatchMatch.computeNNF(FeatureMaps_A1[L], FeatureMaps_B2[L], 
                                           FeatureMaps_A2[L], FeatureMaps_B1[L], 
                                           config, initialNNF=initialNNF_ab)
    
    NNFs_ba[L] = DeepPatchMatch.computeNNF(FeatureMaps_B1[L], FeatureMaps_A2[L], 
                                           FeatureMaps_B2[L], FeatureMaps_A1[L], 
                                           config, initialNNF=initialNNF_ba)

    if L > 1:
        # Reconstruction for A2
        Warped_FeatureMaps_A2 = DeepPatchMatch.warp(FeatureMaps_B1[L], NNFs_ab[L])
        R_A2[L-1] = DeepReconstruction.deconv(net=subnets[L-1], target=Warped_FeatureMaps_A2, 
                                                  source=FeatureMaps_B1[L-1], loss=nn.MSELoss, value=None, max_iter=300)
        FeatureMaps_A2[L-1] = DeepReconstruction.blend(FeatureMaps_A1[L-1], R_A2[L-1], DeepReconstruction.get_weight_map(FeatureMaps_A1[L-1], config["alphas"][L-1]))

        # Reconstruction for B2
        Warped_FeatureMaps_B2 = DeepPatchMatch.warp(FeatureMaps_A1[L], NNFs_ba[L])
        R_B2[L-1] = DeepReconstruction.deconv(net=subnets[L-1], target=Warped_FeatureMaps_B2, 
                                                  source=FeatureMaps_A1[L-1], loss=nn.MSELoss, value=None, max_iter=300)
        FeatureMaps_B2[L-1] = DeepReconstruction.blend(FeatureMaps_A1[L-1], R_B2[L-1], DeepReconstruction.get_weight_map(FeatureMaps_B1[L-1], config["alphas"][L-1]))
        
# NNF search
if True: # TODO : Reactivate Upsampling
    initialNNF_ab = None
    initialNNF_ba = None

else:
    initialNNF_ab = DeepReconstruction.upsample(torch.unsqueeze(NNFs_ab[L+1], 0), FeatureMaps_A1[L].size()[-1], mode="nearest")
    initialNNF_ba = DeepReconstruction.upsample(torch.unsqueeze(NNFs_ba[L+1], 0), FeatureMaps_A1[L].size()[-1], mode="nearest")

Warped_A2 = DeepPatchMatch.warp(B1, initialNNF_ab)
Warped_B2 = DeepPatchMatch.warp(A1, initialNNF_ba)
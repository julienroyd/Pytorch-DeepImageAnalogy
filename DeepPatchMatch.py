from config import config

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os


def Patch(image, i, j, m):
    """ Returns a m x m patch taken from Image, centered at coordinates i,j """
    patch = image[:, :, i-m:i+m+1, j-m:j+m+1]
    return patch


def valid(coord, coord_limit, m):
    """ Makes sure the coordinates lie in a valid range (m corrects for the zero-padding) """
    coord = max(coord, m)
    coord = min(coord, coord_limit + m - 1)

    return coord


def warp(B1, NNF_ab):
    """ Warp exprimes the content of image A but only with pixels sampled from image B """
    NNF_ab = NNF_ab.type(torch.LongTensor).cuda()
    Warp = B1[:, :, NNF_ab[0,:,:], NNF_ab[1,:,:]]
    
    return Warp


def euclideanDistance(P1, P2):
    """ Returns the squared euclidean distance between two patches """
    distance = torch.sum((P1 - P2) ** 2)
    return distance


def distance(PA1, PB2, PA2, PB1, mode="bidirectional"):
    """ Returns the bidirectional distance as described in the paper """
    
    if mode == "unidirectional":
        distance = euclideanDistance(PA1, PB2)

    elif mode == "bidirectional":
        distance = euclideanDistance(PA1, PB2) + euclideanDistance(PA2, PB1) 

    else:
        raise ValueError('The "mode" argument for function "distance" in "PatchMatch" should be either "unidirectional" or "bidirectional".')
    
    return distance


def initializeNNF(h, w, initialNNF=None):
    """ 
    Randomly initializes NNF_ab
    - NNF_ab[:,i,j] is a 2D vector representing the coordinates x,y 
      so that B[0,:,x,y] is most similar to A[0,:,i,j] 
    """
    if initialNNF is None:
        # Instanciates NNF_ab
        NNF_ab = np.zeros(shape=(2,h,w), dtype=np.int)
        
        # Fill in NNF_ab[:,:,0] contains the coordinates x (row)
        NNF_ab[0,:,:] = np.random.randint(low=0, high=h, size=(h,w))
            
        # Fill in NNF_ab[:,:,1] contains the coordinates y (column)
        NNF_ab[1,:,:] = np.random.randint(low=0, high=w, size=(h,w))
        
        NNF_ab = torch.from_numpy(NNF_ab)

    elif isinstance(initialNNF, torch.IntTensor) and initialNNF.size() == (2, w, h):
        # NNF_ab is intialized to initialNNF
        NNF_ab = initialNNF

    else:
        raise ValueError('The provided "initialNNF" should be an torch.IntTensor of size ({0},{1}) but got type {2} of size {3} '.format(h,w,type(initalNNF),tuple(initial.size())))
    
    # Turns it into an autograd.Variable and send it on the GPU
    if torch.cuda.is_available() and False:
        NNF_ab = Variable(NNF_ab).cuda()
    else:
        NNF_ab = Variable(NNF_ab)
    
    return NNF_ab


def propagate(A1, B2, A2, B1, h, w, m, NNF_ab, i, j, shift, config):

    # Extract the patches at coordinates i,j in A1 and A2
    A1_patch = Patch(A1,i,j,m)
    A2_patch = Patch(A2,i,j,m)
    
    # Extract the patch-match in B2 associated with current position
    A1_current_patchMatch = Patch(B2, NNF_ab[0,i,j], NNF_ab[1,i,j], m)
    A2_current_patchMatch = Patch(B1, NNF_ab[0,i,j], NNF_ab[1,i,j], m)
    
    # Computes the distance between current patch
    current_match = distance(A1_patch, A1_current_patchMatch, A2_patch, A2_current_patchMatch, mode=config['distance_mode'])

    if config['propagation_mode'] == 'NM':
    
        # Extract the patch-match in B associated with left neighbor and up neighbor in A
        A1_leftNeighbor_patchMatch = Patch(B2, 
                                          NNF_ab[0,i,valid(j+shift,w,m)], 
                                          NNF_ab[1,i,valid(j+shift,w,m)], 
                                          m)
        A2_leftNeighbor_patchMatch = Patch(B1, 
                                          NNF_ab[0,i,valid(j+shift,w,m)], 
                                          NNF_ab[1,i,valid(j+shift,w,m)], 
                                          m)
        
        A1_upNeighbor_patchMatch = Patch(B2, 
                                        NNF_ab[0,valid(i+shift,h,m),j], 
                                        NNF_ab[1,valid(i+shift,h,m),j], 
                                        m)
        A2_upNeighbor_patchMatch = Patch(B1, 
                                        NNF_ab[0,valid(i+shift,h,m),j], 
                                        NNF_ab[1,valid(i+shift,h,m),j], 
                                        m)

        # Computes the distance between potential matches
        left_neighbor_match = distance(A1_patch, A1_leftNeighbor_patchMatch, A2_patch, A2_leftNeighbor_patchMatch, mode=config['distance_mode'])
        up_neighbor_match = distance(A1_patch, A1_upNeighbor_patchMatch, A2_patch, A2_upNeighbor_patchMatch, mode=config['distance_mode'])

        # Looks up which match is the best. If best match is current match, nothing is changed
        best_match = np.argmin(np.array([current_match, left_neighbor_match, up_neighbor_match]))
  
        if best_match == 1:
            # New patch-match in B is the same than for left-neighbor in A
            NNF_ab[:,i,j] = NNF_ab[:,i,valid(j+shift,w,m)]

        if best_match == 2:
            # New patch-match in B is the same than for up-neighbor in A
            NNF_ab[:,i,j] = NNF_ab[:,valid(i+shift,h,m),j]
        
        
    elif config['propagation_mode'] == 'NoNM':
    
        # Extract the patch-match in B associated with left neighbor and up neighbor in A
        A1_RN_PM_LN = Patch(B2, 
                         NNF_ab[0,i,valid(j+shift,w,m)], 
                         valid(NNF_ab[1,i,valid(j+shift,w,m)]-shift,w,m), 
                         m)
        A2_RN_PM_LN = Patch(B1, 
                         NNF_ab[0,i,valid(j+shift,w,m)], 
                         valid(NNF_ab[1,i,valid(j+shift,w,m)]-shift,w,m), 
                         m)
        
        A1_DN_PM_UP = Patch(B2, 
                         valid(NNF_ab[0,valid(i+shift,h,m),j]-shift,h,m), 
                         NNF_ab[1,valid(i+shift,h,m),j], 
                         m)
        A2_DN_PM_UP = Patch(B1, 
                         valid(NNF_ab[0,valid(i+shift,h,m),j]-shift,h,m), 
                         NNF_ab[1,valid(i+shift,h,m),j], 
                         m)

        # Computes the distance between potential matches
        left_neighbor_match = distance(A1_patch, A1_RN_PM_LN, A2_patch, A2_RN_PM_LN, mode=config['distance_mode'])
        up_neighbor_match = distance(A1_patch, A1_DN_PM_UP, A2_patch, A2_DN_PM_UP, mode=config['distance_mode'])

        # Looks up which match is the best. If best match is current match, nothing is changed
        best_match = np.argmin(np.array([current_match, left_neighbor_match, up_neighbor_match]))        
        
        if best_match == 1:
            # New patch-match in B based on left-neighbor's match
            NNF_ab[0,i,j] = NNF_ab[0,i,valid(j+shift,w,m)]
            NNF_ab[1,i,j] = valid(NNF_ab[1,i,valid(j+shift,w,m)]-shift,w,m)

        if best_match == 2:
            # New patch-match in B based on up-neighbor's match
            NNF_ab[0,i,j] = valid(NNF_ab[0,valid(i+shift,h,m),j]-shift,h,m)
            NNF_ab[1,i,j] = NNF_ab[1,valid(i+shift,h,m),j]
     

    return NNF_ab


def randomSearch(A1, B2, A2, B1, h, w, m, NNF_ab, i, j, L, config):

    max_step = config['random_search_max_step'][L] // 2

    for k in range(config['number_of_patches_per_zone']):
        # The randomly sampled coordinates for the random patch-match
        [x, y] = NNF_ab[:,i,j].numpy() + np.random.randint(low=-max_step, high=max_step, size=(2,))

        # Makes sure that those coordinates lie within the limits of our image
        x = valid(x, h, m)
        y = valid(y, w, m)

        # Extract the patch around our current position in A
        A1_patch = Patch(A1,i,j,m)
        A2_patch = Patch(A2,i,j,m)

        # Extract current patch-match and random patch-match in B2
        A1_current_patchMatch = Patch(B2, NNF_ab[0,i,j], NNF_ab[1,i,j], m)
        A1_random_patchMatch = Patch(B2, x, y, m)

        # Extract current patch-match and random patch-match in B1
        A2_current_patchMatch = Patch(B1, NNF_ab[0,i,j], NNF_ab[1,i,j], m)
        A2_random_patchMatch = Patch(B1, x, y, m)

        # Computes how good both matches are
        current_match = distance(A1_patch, A1_current_patchMatch, A2_patch, A2_current_patchMatch, mode=config['distance_mode'])
        random_match = distance(A1_patch, A1_random_patchMatch, A2_patch, A2_random_patchMatch, mode=config['distance_mode'])
        
        # Looks up which match is the best. If best match is current match, nothing is changed
        best_match = np.argmin(np.array([current_match, random_match]))

        if best_match == 1:
            NNF_ab[:,i,j] = torch.from_numpy(np.array([x,y]))
                    
    return NNF_ab


def computeNNF(A1, B2, A2, B1, L, config, initialNNF=None):
    """
    Computes the NNF function from A-s and B-s
    image 1 : an autograd.Variable of shape [batch, channels, heigth, width]
    image 2 : an autograd.Variable of shape [batch, channels, heigth, width]
    """
    
    # Makes sure the two images have the same size
    if A1.size() != B2.size() and A2.size() != B1.size() and A1.size() != A2.size() :
        raise ValueError("All the images must have the same size.")
    
    # Heigth and Width of images
    [h,w] = A1.size()[2:]
     
    # Patch half-size
    m = int((config['patch_size'][L] - 1) / 2)
    
    # Randomly initializes NNF_ab
    NNF_ab = initializeNNF(h, w, initialNNF)
    
    # Zero-Pad images
    A1 = F.normalize(F.pad(A1, (m,m,m,m), mode='reflect').data).float()
    A2 = F.normalize(F.pad(A2, (m,m,m,m), mode='reflect').data).float()
    B1 = F.normalize(F.pad(B1, (m,m,m,m), mode='reflect').data).float()
    B2 = F.normalize(F.pad(B2, (m,m,m,m), mode='reflect').data).float()
    
    # Zero-Pad NNF_ab so its coordinate system as well as the values it contains remain consistent with the images
    NNF_ab += m
    NNF_ab = torch.squeeze(F.pad(NNF_ab.unsqueeze(0), (m,m,m,m), mode='constant', value=0)).data
    
    # Defines valid ranges (exclude the padded indexes)
    i_range = np.arange(h) + m
    j_range = np.arange(w) + m
    
    # Executes the PatchMatch algorithm n_iter times
    for step in range(config['n_iter']):
        if step%2 == 0:
            shift = 1
        else:
            shift = -1
        
        # For every valid pixel in the image
        for i in i_range:
            if (i+1)%100 == 0 : print("Row : {0}".format(i+1))
            for j in j_range:
                NNF_ab = propagate(A1, B2, A2, B1, h, w, m, NNF_ab, i, j, shift, config)
                NNF_ab = randomSearch(A1, B2, A2, B1, h, w, m, NNF_ab, i, j, L, config)
    
    NNF_final = NNF_ab[:, m:-m, m:-m]
    NNF_final -= m

    print("PatchMatch done!")
    
    return NNF_final

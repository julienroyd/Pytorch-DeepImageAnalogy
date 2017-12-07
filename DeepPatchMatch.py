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
        distance = euclideanDistance(PA2, PB1)

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


def propagate(A1, B2, A2, B1, h, w, m, NNF_ab, NNF_dist, i, j, shift, config):

    # Extract the patches at coordinates i,j in A1 and A2
    A1_patch = Patch(A1,i,j,m)
    A2_patch = Patch(A2,i,j,m)

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
    best_match = np.argmin(np.array([NNF_dist[i,j], left_neighbor_match, up_neighbor_match]))        
    
    if best_match == 1:
        # New patch-match in B based on left-neighbor's match
        NNF_ab[0,i,j] = NNF_ab[0,i,valid(j+shift,w,m)]
        NNF_ab[1,i,j] = valid(NNF_ab[1,i,valid(j+shift,w,m)]-shift,w,m)
        NNF_dist[i,j] = left_neighbor_match

    if best_match == 2:
        # New patch-match in B based on up-neighbor's match
        NNF_ab[0,i,j] = valid(NNF_ab[0,valid(i+shift,h,m),j]-shift,h,m)
        NNF_ab[1,i,j] = NNF_ab[1,valid(i+shift,h,m),j]
        NNF_dist[i,j] = up_neighbor_match
     

    return NNF_ab


def random_search(A1, B2, A2, B1, h, w, m, NNF_ab, NNF_dist, p_i, p_j, config, L):
    
    # helper : compute distance
    def fdist(A1, B2, A2, B1, p_i, p_j, q_i, q_j, m):
    
        # extracts patches around pixel p in A1 and A2
        A1_patch = Patch(A1, p_i, p_j, m)
        A2_patch = Patch(A2, p_i, p_j, m)
        
        # extracts patches around pixel q in A1 and A2
        B1_patch = Patch(B1, q_i, q_j, m)
        B2_patch = Patch(B2, q_i, q_j, m)
        
        # compute distance (here euclidean)
        dist = distance(A1_patch, B2_patch, A2_patch, B1_patch, mode=config['distance_mode'])

        return dist
    
    # Random Search radius
    rad = config["random_search_max_step"][L]
    
    # size over dimensions of interest (height, width)
    h, w = A1.size()[-2:]
    
    # get coordinates of current best match
    i_match, j_match = NNF_ab[:, p_i, p_j].numpy()
    
    # distance to current best match
    dist_match = NNF_dist[p_i, p_j]
    
    
    while (rad >= 1):
        
        # compute a valid search window
        i_min, j_min = max(i_match - rad, m), max(j_match - rad, m)
        i_max, j_max = min(i_match + rad, h-m), min(j_match + rad, w-m)
        
        # randomly sample a shift
        r_i, r_j = np.random.randint(i_min, i_max), np.random.randint(j_min, j_max)
        
        # compute distance to sample
        dist_random = fdist(A1, B1, A2, B2, p_i, p_j, r_i, r_j, m)
        
        if dist_random < dist_match:
            i_match, j_match, dist_match = r_i, r_j, dist_random
        
        # reduce search radius
        rad = np.floor(0.5 * rad)
        
    # update NNF_ab
    NNF_ab[:, p_i, p_j] = torch.from_numpy(np.array([i_match, j_match]))
    
    return NNF_ab


def computeNNF(A1, B2, A2, B1, L, config, initialNNF=None):
    """
    Computes the NNF function from A-s and B-s
    image 1 : an autograd.Variable of shape [batch, channels, heigth, width]
    image 2 : an autograd.Variable of shape [batch, channels, heigth, width]
    """
    
    # Makes sure the two images have the same size
    if A1.size() != B2.size() or A2.size() != B1.size() or A1.size() != A2.size() :
        raise ValueError("All the images must have the same size.")
    
    # Heigth and Width of images
    [h,w] = A1.size()[2:]
     
    # Patch half-size
    m = config['patch_size'][L] // 2
    
    # Randomly initializes NNF_ab
    NNF_ab = initializeNNF(h, w, initialNNF)
    
    # Pad and normalizes the images
    A1 = F.normalize(F.pad(A1, (m,m,m,m), mode='reflect').data).float()
    A2 = F.normalize(F.pad(A2, (m,m,m,m), mode='reflect').data).float()
    B1 = F.normalize(F.pad(B1, (m,m,m,m), mode='reflect').data).float()
    B2 = F.normalize(F.pad(B2, (m,m,m,m), mode='reflect').data).float()
    
    # Zero-Pad NNF_ab so its coordinate system as well as the values it contains remain consistent with the images
    NNF_ab += m
    NNF_ab = torch.squeeze(F.pad(NNF_ab.unsqueeze(0), (m,m,m,m), mode='constant', value=0)).data

    # Creates and initializes a NNF_dist, a matrix that saves the distance to current best match (so we don't have to compute it all the time)
    NNF_dist = torch.zeros(NNF_ab.size()[1:]).type(torch.FloatTensor)
    i_range = np.arange(h) + m
    j_range = np.arange(w) + m
    for i in i_range:
        for j in j_range:
            A1_patch = Patch(A1,i,j,m)
            A2_patch = Patch(A2,i,j,m)
            A1_current_patchMatch = Patch(B2, NNF_ab[0,i,j], NNF_ab[1,i,j], m)
            A2_current_patchMatch = Patch(B1, NNF_ab[0,i,j], NNF_ab[1,i,j], m)
            current_match = distance(A1_patch, A1_current_patchMatch, A2_patch, A2_current_patchMatch, mode=config['distance_mode'])

            NNF_dist[i,j] = current_match
    
    # Executes the PatchMatch algorithm n_iter times
    for step in range(config['n_iter']):
        if step%2 == 0:
            shift = -1

            # Defines valid ranges (exclude the padded indexes)
            i_range = np.arange(h) + m
            j_range = np.arange(w) + m
        
        else:
            shift = 1

            # Defines valid ranges (exclude the padded indexes)
            i_range = np.arange(h) + m
            j_range = np.arange(w) + m

            i_range = i_range[::-1]
            j_range = j_range[::-1]
        
        # For ever_i valid pixel in the image
        for i in i_range:
            if (i+1)%100 == 0 : print("Row : {0}".format(i+1))
            for j in j_range:
                NNF_ab = propagate(A1, B2, A2, B1, h, w, m, NNF_ab, NNF_dist, i, j, shift, config)
                NNF_ab = random_search(A1, B2, A2, B1, h, w, m, NNF_ab, NNF_dist, i, j, config, L)
    
    NNF_final = NNF_ab[:, m:-m, m:-m]
    NNF_final -= m

    print("PatchMatch done!")
    
    return NNF_final

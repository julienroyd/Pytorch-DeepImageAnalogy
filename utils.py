import os
import pickle
import numpy as np

def saveNNFs(filename, NNFs_xy):
    with open(os.path.join(filename), 'wb') as f:
        pickle.dump({1:np.transpose(NNFs_xy[1].numpy(), axes=(1,2,0)),
                     2:np.transpose(NNFs_xy[2].numpy(), axes=(1,2,0)),
                     3:np.transpose(NNFs_xy[3].numpy(), axes=(1,2,0)),
                     4:np.transpose(NNFs_xy[4].numpy(), axes=(1,2,0)),
                     5:np.transpose(NNFs_xy[5].numpy(), axes=(1,2,0))}, 
                     f, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved in : {0}'.format(os.path.join(filename)))


def loadNNFs(filename):
    with open(os.path.join(filename), 'rb') as f:
        NNFs_xy = pickle.load(f)

    return NNFs_xy


def saveFeatureMaps(filename, featureMaps_X):
    with open(os.path.join(filename), 'wb') as f:
        pickle.dump({1:np.transpose(featureMaps_X[1].data[0].cpu().numpy(), axes=(1,2,0)),
                     2:np.transpose(featureMaps_X[2].data[0].cpu().numpy(), axes=(1,2,0)),
                     3:np.transpose(featureMaps_X[3].data[0].cpu().numpy(), axes=(1,2,0)),
                     4:np.transpose(featureMaps_X[4].data[0].cpu().numpy(), axes=(1,2,0)),
                     5:np.transpose(featureMaps_X[5].data[0].cpu().numpy(), axes=(1,2,0))}, 
                     f, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved in : {0}'.format(os.path.join(filename)))


def loadFeatureMaps(filename):
    with open(os.path.join(filename), 'rb') as f:
        featureMaps_X = pickle.load(f)

    return featureMaps_X
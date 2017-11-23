import DeepVGG

import os
import pickle

import numpy as np

def saveFeatureMaps(filename, featureMaps_X):
    with open(os.path.join('Results', filename), 'wb') as f:
        pickle.dump({1:np.transpose(DeepVGG.postp(featureMaps_X[1].data[0].cpu()).numpy(), axes=(1,2,0)),
                     2:np.transpose(DeepVGG.postp(featureMaps_X[2].data[0].cpu()).numpy(), axes=(1,2,0)),
                     3:np.transpose(DeepVGG.postp(featureMaps_X[3].data[0].cpu()).numpy(), axes=(1,2,0)),
                     4:np.transpose(DeepVGG.postp(featureMaps_X[4].data[0].cpu()).numpy(), axes=(1,2,0)),
                     5:np.transpose(DeepVGG.postp(featureMaps_X[5].data[0].cpu()).numpy(), axes=(1,2,0)),}, 
                     f, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved in : {0}'.format(os.path.join('Results', filename)))


def loadFeatureMaps(filename):
    with open(os.path.join('Results', filename), 'rb') as f:
        featureMaps_X = pickle.load(f)

    return featureMaps_X
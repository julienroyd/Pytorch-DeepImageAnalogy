from config import config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# pseudo deconvolution operation implemented as a local optimisation problem
def deconv(model, target, noise_size, layer, loss, opt, n_iters=2500):

    noise = torch.randn(noise_size).float()

    if torch.cuda.is_available():
        noise = noise.cuda()

    target = target.detach()
    noise = Variable(noise, requires_grad=True)
    optimizer = opt([noise])
    loss_fn = loss()

    for i in range(n_iters):
        
        optimizer.zero_grad()
        
        output = model.forward_subnet(noise, layer)

        loss_value = loss_fn(output, target)

        loss_value.backward()

        if i % 200 == 0:
            print('Iteration: {0:d}, loss: {1:.2f}'.format(i, loss_value.data[0]))

        optimizer.step()
        noise.data.clamp_(0., 1.)

    return noise

# determines the weight map 
def get_weight_map(features, alpha, kappa=300, tau=0.05):
    # feature maps of shape [batch_size, channels, height, width]
    # normalizes features maps across the channel dimension
    x = features # torch.norm(features, p=2, dim=1)
    # removes the batch dimension (it is always 1 for 1 image)
    x = torch.squeeze(x, 0)
    
    # squares the feature maps and change dynamic range to [0, 1]
    x = x*x
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    
    return alpha * torch.sigmoid(kappa * (x - tau))

# blends x with y using the weight map W
def blend(x, y, W):
    return W * x + (1 - W) * y


def upsample(inputs, size, mode="nearest"):
    # Instanciate the new NNF
    x = Variable(inputs.unsqueeze(0))
    
    if not isinstance(inputs, torch.FloatTensor):
        x = x.float()
    
    # Upsamples using nearest neighbor method
    o = F.upsample(x, size=size, mode=mode).squeeze(0).int()
    
    # Rescale the values of the NNF to remain relevant after upsampling
    o_adapted = 2 * o
    
    return o_adapted.data
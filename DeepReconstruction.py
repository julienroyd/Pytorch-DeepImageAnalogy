from config import config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# pseudo deconvolution operation implemented as a local optimisation problem
def deconv(net, target, source, loss=nn.MSELoss, value=None, max_iter=2500): # TODO : Get rid of source parameter and ask for size and dtype instead
    optimum = value if value is not None else Variable(torch.randn(source.size()).type_as(source.data), requires_grad=True)
    loss_fn = loss()
    
    optimizer = config['optimizer']([optimum])
    
    show_iter = 50
    n_iter=[0]

    while n_iter[0] <= max_iter:

        def closure():
            t = target.detach()
            optimizer.zero_grad()
            loss = loss_fn(net(optimum), t)
            loss.backward()
            n_iter[0]+=1
            #print loss
            if n_iter[0]%show_iter == (show_iter-1):
                print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.data[0]))
            return loss

        optimizer.step(closure)
    return optimum

# determines the weight map 
def get_weight_map(features, alpha, kappa = 300, tau = 0.05):
    # feature maps of shape [batch_size, n_channels, height, width]
    # normalize features maps across the channel dimension
    x = torch.norm(features, p=2, dim=1)
    # we suppose the batch size is 1 as we use 1 image only (this may change later?)
    # thus, we remove the batch dimension
    x = torch.squeeze(x, 0)
    x = x*x
    x = x / torch.max(x)
    return alpha * torch.sigmoid(kappa * (x + tau))

# blends x with y using the weight map W: W*x + (1-W)*y
def blend(x, y, W):
    return W * x + (1 - W) * y

# upsampling
# mode = 'nearest'
def upsample(inputs, size, mode="nearest"):
    x = Variable(inputs.unsqueeze(0))
    if not isinstance(inputs, torch.FloatTensor):
        x = x.float()
    o = F.upsample(x, size=size, mode=mode).squeeze(0).int()
    o_adapted = 2 * o
    return o_adapted.data
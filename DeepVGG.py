# VGG implementation using torchvision (inspired from : https://github.com/harveyslash/Deep-Image-Analogy-PyTorch)
from config import config
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


class FeatureExtractor(nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def forward(self, x):
        outputs = []
        for module in self._modules:
            x = self._modules[module](x)
            outputs.append(x)
        return outputs


class VGG19:
    def __init__(self):
        self.cnn_temp = torchvision.models.vgg19(pretrained=True).features
        self.model = FeatureExtractor()
        
        conv_counter = 1
        relu_counter = 1
        block_counter = 1

        for i, layer in enumerate(list(self.cnn_temp)):

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(block_counter) + "_" + str(conv_counter) + "__" + str(i)
                conv_counter += 1
                self.model.add_module(name, layer)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(block_counter) + "_" + str(relu_counter) + "__" + str(i)
                relu_counter += 1
                self.model.add_module(name, nn.ReLU(inplace=False))

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(block_counter) + "__" + str(i)
                relu_counter = conv_counter = 1
                block_counter += 1
                self.model.add_module(name, nn.AvgPool2d((2,2)))

        if torch.cuda.is_available():
            self.model.cuda()
            

    def forward_subnet(self, input_tensor, L):
        
        if L == 5:
            start_layer,end_layer = 21,29 # From Conv4_2 to ReLU5_1 inclusively
        elif L == 4:
            start_layer,end_layer = 12,20 # From Conv3_2 to ReLU4_1 inclusively
        elif L == 3:
            start_layer,end_layer = 7,11 # From Conv2_2 to ReLU3_1 inclusively
        elif L == 2:
            start_layer,end_layer = 2,6 # From Conv1_2 to ReLU2_1 inclusively
        else:
            raise ValueError("Invalid layer number")

        for i, layer in enumerate(list(self.model)):
            if i >= start_layer and i <= end_layer:
                input_tensor = layer(input_tensor)
        return input_tensor

    def forward(self, img_tensor):
        features = self.model(img_tensor)

        return features



# Preprocessing
prep = transforms.Compose([transforms.Scale(config['img_size']),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #subtract imagenet mean
                          ])
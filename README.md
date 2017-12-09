# Pytorch-DeepImageAnalogy

Implementation of Deep Image Analogy algorithm [[Liao et al. 2017]](https://arxiv.org/abs/1705.01088) using Pytorch. It is meant to be as simple and easy to read as possible, to allow everyone to uderstand how the algorithm works along with the original paper.

Deep Image Analogy is a adaptation of Image Style Transfer [[Gatys et al. 2016]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html) that uses feature maps constructed by a deep CNN such as VGG [[Simonyan, Zisserman. 2015]](https://arxiv.org/abs/1409.1556) and the Randomized PatchMatch technique [[Barnes et al. 2009]](https://www.researchgate.net/profile/Eli_Shechtman/publication/220184392_PatchMatch_A_Randomized_Correspondence_Algorithm_for_Structural_Image_Editing/links/02e7e520897b12bf0f000000.pdf) to allow transfering visual attributes (color, style) from one image to another, while conserving the semantic attributes of the original image.

The following images are examples of the kind of results that we are able to get so far. There are still important artefacts to be improved, but we can definitely see that's it's doing the right thing!
![](examples.png)

### Dependencies

We use Python 3.6.1, along with the following dependencies. I assume you use a conda virtual environment. If you don't, use `pip3` instead of `pip`.

* pytorch : `pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl`
* numpy : `pip install numpy==1.13.3`
* matplotlib : `pip install matplotlib==2.0.2`
* torchvision : `pip install torchvision`

##### Python built-in dependencies

* pickle
* os

### To run

1. Edit `config.py` to choose the images you want to run on
2. Run using `python DeepImageAnalogy.py`
3. When it is done, the results will be saved in `Results/` folder. If you had `config['save_NNFs'] = True` and `config['save_FeatureMaps'] = True` in the config file, you can also open a notebook using `jupyter-notebook Visualize.ipynb` and visualize the generated Feature Maps and NNFs there.

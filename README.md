# Pytorch-DeepImageAnalogy

Implementation of Deep Image Analogy algorithm [[Liao et al. 2017]](https://arxiv.org/abs/1705.01088) using Pytorch. It is meant to be as simple and easy to read as possible, to allow everyone to uderstand how the algorithm works along with the original paper.

Deep Image Analogy is a adaptation of Image Style Transfer [[Gatys et al. 2016]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html) that uses feature maps constructed by a deep CNN such as VGG [[Simonyan, Zisserman. 2015]](https://arxiv.org/abs/1409.1556) and the Randomized PatchMatch technique [[Barnes et al. 2009]](https://www.researchgate.net/profile/Eli_Shechtman/publication/220184392_PatchMatch_A_Randomized_Correspondence_Algorithm_for_Structural_Image_Editing/links/02e7e520897b12bf0f000000.pdf) to allow transfering visual attributes (color, style) from one image to another, while conserving the semantic attributes of the original image.

The following images are examples of the kind of results that we are able to get so far. It is definitely trying to do the right thing, but that are still some artefacts that need to be improved.
![](examples.png)


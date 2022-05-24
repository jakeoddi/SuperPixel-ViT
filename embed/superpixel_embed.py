"""
This module converts an Image to its Super-Pixel embeddings.

Based on the work by Dosovitskiy et. al. https://arxiv.org/abs/2010.11929 and
the implementation in https://github.com/rwightman/pytorch-image-models/

Author: Jake Oddi
Last Modified: 04-18-2022
"""
import time
import torch
import numpy as np
from torch import nn as nn
from einops import rearrange
from skimage.segmentation import slic

class SuperPixelEmbed(nn.Module):
    """
    A class used to convert a 2D Image to its Super-Pixel Embeddings

    ...

    Attributes
    ----------


    Methods
    -------


    """
    def __init__(self, img_size=224, superpixels=196, in_chans=3, embed_dim=768, mask=None, norm_layer=None):
        super().__init__()
        # initialize device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_size = img_size
        self.superpixels = superpixels
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mask = mask


        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        


    # ABSTRACT METHOD STUB
    def forward(self, x): return 

    # To alter parameters in SLIC algorithm
    def set_sigma(): return 



class SuperPixelPadEmbed(SuperPixelEmbed):
    """
    A class used to convert a 2D Image to its Super-Pixel Embeddings

    Pads each variable-length superpixel with some set value so each is the 
    same length as the longest.
    ...

    Attributes
    ----------


    Methods
    -------


    """
    def __init__(self, img_size=224, superpixels=196, in_chans=3, embed_dim=768, mask=None, norm_layer=None):
        super().__init__()
        

#     @Override
    def forward(self, img):
        """
        Computes superpixel embeddings from an image `img`

        Parameters
        ----------
        x: torch.Tensor() representing a batch of images with shape 
        (batch_size, superpixels, embed_dim)
        """

        # TODO: use segment labels in positional embedding as well 
        # TODO: find out a way to get a fixed number of superpixels, b/c sk SLIC only gives approx

        # SLIC compactness, max_size_factor. Check # of superpixels and that each image is getting the same number

        # in: image tensor
        # create array of tensor
        # pass to _create_segments
        # 
        # shape (batch_size x num_patches x flattened_patch * num_channels)

        # TODO: need to somehow apply this accross all images in batch, parallelized as much as possible
        # for img in x
        # convert tensor to array to compute superpixels
        img_arr = img.numpy()
        # compute superpixel segmentation
        seg = self._create_segments(img_arr)
        # get patch dimension from segments tensor
        patch_dim = seg.size()[1]
        # linear layer for projecting patches to embed dim
        proj = nn.Linear(patch_dim, embed_dim)

        x = proj(x)

        if self.norm:
            x = norm(x)
        

        return x # shape (batch_size x num_patches x embed_dim)


class SuperPixelMeanEmbed(SuperPixelEmbed):
    """
    A class used to convert a 2D Image to its Super-Pixel Embeddings

    Computes the feature for each super pixel by taking the average of the 
    embedded feature of its pixels. The image is then represented by a N x C 
    matrix where N is the number of super pixels and C is the number of 
    channels

    ...

    Attributes
    ----------


    Methods
    -------


    Returns
    -------
    batch_sps: torch.Tensor() containing all superpixels in the batch of shape 
    (batch_size x superpixels x in_chans)

    """
    def __init__(self, img_size=224, superpixels=196, in_chans=3, embed_dim=32, mask=None, norm_layer=None, conv=True):
        super().__init__(
            img_size=img_size,
            superpixels=superpixels,
            in_chans=in_chans,
            embed_dim=embed_dim,
            mask=mask,
            norm_layer=norm_layer,
        )
        self.conv = conv
        self.seg_arr = []

        # define embedding layer
        # TODO: change embed dim to 8, 16, or 64
        self.conv_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1,1), stride=1)


#     @Override
    def forward(self, X, masks):
        """
        Computes superpixel embeddings for a batch of images `img`

        Parameters
        ----------
        X: torch.Tensor() representing a batch of images with shape (batch_size, in_chans, height, width)
        """
        start_time = time.time()
        # get batch size
        batch_size = X.size()[0]

        #comute pixel-wise embedding with 1x1 kernel convolution embed up to 16 dims
        if self.conv:
            x_emb = self.conv_proj(X)
        # initialize empty list to store superpixel matrices for all images in batch `i`
        batch_sps = []
        # compute segment map for each embedded image in batch
        for i, x in enumerate(x_emb):
            # get masks for image at index i
            masks_i = masks[i]
            # compute mean embedding for each superpixel
            im_sps = [x[:,mask].mean(dim=1) for mask in masks_i]
            # stack superpixels for image `x`
            im_sps = torch.stack(im_sps, dim=0)
            # add to list of images
            batch_sps.append(im_sps)
                
        # stack batches for batch `X`
        batch_sps = torch.stack(batch_sps)

#         if self.norm:   THIS CAUSED PROBLEMS, FIX LATER
#             batch_sps = norm_layer(batch_sps)
        total_time = time.time() - start_time
        print('embedding time %.2f' % total_time)
        return batch_sps 

"""
This module converts an Image to its Super-Pixel embeddings.

Based on the implementation in https://github.com/rwightman/pytorch-image-models/

Author: Jake Oddi
Last Modified: 04-11-2022
"""
from torch import nn as nn
from skimage.segmentation import slic
import numpy as np
from einops import rearrange

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
        self.img_size = img_size
        self.superpixels = superpixels
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mask = mask


        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        


    # ABSTRACT METHOD STUB
    def forward(self, x): return 



    def _create_segments(self, img_arr):
        """
        Creates segments for one image at a time. Then parallelized accross
        the entire batch.

        Parameters
        ----------
        img_arr: np.Array() of an image, with shape (in_chans x img_size x img_size)


        Returns
        -------
        segment_map: np.Array() with shape (num_patches x largest_patch_length*in_chans)

        segment_labels: np.Array() with sequence of unique superpixel labels

        largest: Int representing number of pixels in largest superpixel 
        """
        # reshape input array to be passed through segmentation
        # (img_size x img_size x in_chans) -> (in_chans x img_size x img_size)
        image_arr = np.transpose(image_arr, (1, 2, 0))
        # compute segments array
        segment_map = slic(image_arr, 
                        n_segments = self.superpixels, 
                        sigma=3, 
                        channel_axis=2,
                        #slic_zero=True
                    ) 
        # store segment labels
        segment_labels = np.unique(segment_map)

        # get length of largest superpixel
        largest = _get_largest(segment_map, segment_labels)

        # use superpixel index for regular image index - they're the same size
        # can also loop through each label and extract number using a mask
        # i'll do the former
         

        # do I have each sp channel consecutively followed by padding or each
        # sp channel separated by padding

        # loop to create tensor for each superpixel, padding zeros via: max - current_len


        # return segmented 
        return segment_map, segment_labels, largest


    def _get_largest(smap, labels): 
        """
        function for getting size of largest superpixel

        Parameters
        ----------
        smap: np.Array() mapping each pixel to its superpixel/segment (img_size x img_size)

        labels: np.Array() containing unique segment labels

        Returns
        -------
        largest: Int representing number of pixels in largest superpixel
        """
        largest = 0
        # track occurences of each superpixel
        v = [0 for i in labels]
        tracker = dict(zip(np.unique(smap), v))

        # loop through segment map and count each occurence
        for i in smap.flatten():
            # increment count of each sp label
            tracker[i]+=1
            # check if largest. If so, update largest
            if(tracker[i] > largest):
                largest = tracker[i]

        # check that no values are missing
        assert smap.flatten().shape[0] == sum(tracker.values()), 'Values are missing'

        return largest



    # def set sigma() --- METHOD STUB


class SuperPixelPadEmbed(SuperPixelEmbed):
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
        

    @Override
    def forward(self, img):
        """
        Computes superpixel embeddings from an image `img`

        Parameters
        ----------
        x: torch.Tensor() representing a batch of images with shape (batch_size, superpixels, embed_dim)
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
        seg = _create_segments(img_arr)
        # get patch dimension from segments tensor
        patch_dim = seg.size()[1]
        # linear layer for projecting patches to embed dim
        proj = nn.Linear(patch_dim, embed_dim)

        x = proj(x)

        if self.norm:
            x = norm(x)
        

        return x # shape (batch_size x num_patches x embed_dim)


class SuperPixelMeanEmbed(SuperPixelPadEmbed):
    """
    A class used to convert a 2D Image to its Super-Pixel Embeddings

    Computes the feature for each super pixel by taking the average of the 
    embedded feature of its pixels. The image is then represented by a N x C 
    matrix where N is the number of super pixels and C is the number of channels

    ...

    Attributes
    ----------


    Methods
    -------


    """
    def __init__(self, img_size=224, superpixels=196, in_chans=3, embed_dim=768, mask=None, norm_layer=None, conv=False):
        super().__init__()
        self.conv = conv
        self.seg_arr = []
        # compute size of square patch based on number of superpixels we want to compute
        self.p = self.img_size//self.superpixels**0.5
        # making sure the image contains a whole number of patches
        while self.img_size%self.p != 0:
            self.p+=1

        # get square patch dimension to compute individual pixel embedding
        self.patch_dim = self.in_chans * self.p**2

        if conv:
            self.conv_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.p, stride=self.p)
        
        # define linear embedding layer for individual pixels
        self.lin_proj = nn.Linear(in_chans*img_size**2, in_chans*img_size**2)


    @Override
    def forward(self, X):
        """
        Computes superpixel embeddings for a batch of images `img`

        Parameters
        ----------
        X: torch.Tensor() representing a batch of images with shape (batch_size, in_chans, height, width)
        """
        # define linear embedding for individual pixels
        # pix_embed = nn.Linear(patch_dim, patch_dim) - this is for embedding patch by patch

        # METHOD 2 - RESHAPE
            # reshape according to patches
            # x_emb = rearrange(X, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.p, p2 = self.p)
            # linear projection of each flattened patch
            # x_emb = pix_embed(x_emb)

            # ------ ^^ for patch-wise embedding rather than pixel-wise or sp-wise ^^^ ----
            
            # compute pixel embeddings

        batch_size = X.size()[0]

        # reshape to prepare for linear projection
        reshaped = torch.reshape(X, (batch_size, self.in_chans*self.img_size**2))
        # compute pixel embedding
        x_emb = self.lin_proj(reshaped)
        # reshape back to input dimensions
        x_emb = torch.reshape(x_emb, (batch_size, self.in_chans, self.img_size, self.img_size))
        # initialize empty list to store superpixel matrices for all images in batch `i`
        batch_sps = []

        # compute segment map for each embedded image in batch
        for i, x in enumerate(x_emb):
            # get image as array to compute superpixels
            img_arr = x.detach().numpy()
            # compute superpixel segmentation
            seg_map, seg_unique, largest_sp  = _create_segments(img_arr)
            # get segment map as tensor
            seg_map_tens = torch.from_numpy(seg_map)
            # initialize empty list to store superpixels for image `x`
            im_sps = []

            # compute mean embedding for each superpixel
            for j, m in enumerate(seg_unique):
                # compute mask with only the value of the mth superpixel
                mask = torch.eq(seg_map_tens, m)
                # use mask to get mean of pixel embeddings in superpixel m for each channel in x
                # in an image with 3 channels, this is a list of length 3
                means = [torch.mean(
                    torch.masked_select(x[c, :, :], mask)
                    ) for c in range(x.size()[0])]
                # convert list to tensor
                means = torch.Tensor(means)
                # add to list of superpixels
                im_sps.append(means)

            # stack superpixels for image `x`
            im_sps = torch.stack(im_sps, dim=0)
            # add to list of images
            batch_sps.append(im_sps)
                
        # stack batches for batch `X`
        batch_sps = torch.stack(batch_sps)

        if self.norm:
            batch_sps = norm_layer(batch_sps)
            
        # compute each superpixel feature from average of embedded 
        # component-pixel features
        # _create_segments needs to be fixed so it returns something less specific

        return x 
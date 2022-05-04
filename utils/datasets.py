"""
This module provides custom Datasets that implement superpixel segmentation

Based on:
https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html

Author: Jake Oddi
Last Modified: 05-03-2022
"""

import torch
import numpy as np
from skimage.segmentation import slic
from torchvision.datasets import CIFAR10

class CIFAR10MeanEmbed(CIFAR10):
    """CIFAR10 with Mean Embedded Superpixels Computed at Load Time"""

    def __init__(
        self,
        superpixels, 
        root='', 
        train=True, 
        download=False, 
        transform=None, 
        ):
        super().__init__(root=root, 
            train=train, 
            download=download, 
            transform=transform
        )
        self.superpixels = superpixels
        self.transform = transform

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        # set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # get image as array to compute superpixels
        img_arr = img.cpu().detach().numpy()
        # compute superpixel segmentation
        seg_map, seg_unique, largest_sp  = self._create_segments(img_arr)
        # get segment map as tensor
        seg_map_tens = torch.from_numpy(seg_map)
        # initialize empty list to store superpixels for image `x`
        masks = []

        # compute mean embedding for each superpixel
        for m in seg_unique:
            # compute mask with only the value of the mth superpixel
            mask = torch.eq(seg_map_tens, m)#.to(device)
            # ensuring the input is on device
            
            # add to list of masks
            masks.append(mask)

        # stack superpixels for image `x`
        masks = torch.stack(masks)
        # masks = torch.Tensor(tuple(masks))

        return (img, masks), target

    def _create_segments(self, img_arr):
        """
        Creates segments for one image at a time. Then parallelized accross
        the entire batch.

        Parameters
        ----------
        img_arr: np.Array() of an image, with shape 
        (in_chans x img_size x img_size)


        Returns
        -------
        segmented: torch.Tensor() with shape 
        (num_patches x largest_patch_length*in_chans)
        """
        # reshape input array to be passed through segmentation
        # (img_size x img_size x in_chans) -> (in_chans x img_size x img_size)
        img_arr = np.transpose(img_arr, (1, 2, 0))
        # compute segments array
        segment_map = slic(img_arr, 
                        n_segments = self.superpixels, 
                        sigma=3, 
                        channel_axis=2,
                        #slic_zero=True
                    ) 
        # store segment labels
        segment_labels = np.unique(segment_map)

        # get length of largest superpixel
        largest = self._get_largest(segment_map, segment_labels)
 
        return segment_map, segment_labels, largest



    def _get_largest(self, smap, labels): 
        """
        function for getting size of largest superpixel

        Parameters
        ----------
        smap: img_size x img_size array mapping each pixel to its superpixel/segment
        labels: array containing unique segment labels

        Returns
        -------
        largest: int representing size of largest superpixel
        """
        largest = 0
        # # get list of unique superpixel labels
        # unique = np.unique(segment_map)
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